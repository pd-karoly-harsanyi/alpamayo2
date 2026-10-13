#!/usr/bin/env python
"""Submit this repo to the Nebius Ray cluster as a Ray job (PD-internal).

Uploads the repo with Ray's ``working_dir`` (2-3 MB without .git) instead of git-cloning on the
worker, so the pod's deploy key never has to reach the fork; the worker then runs
``scripts/nebius_entrypoint.sh``. Modelled on pd-fm's ``scripts/submit_mirror_job.py``.

    pixi run -e launch submit-nebius probe --wait                 # env + weights, no inference
    pixi run -e launch submit-nebius smoke --wait -- --sample-indices 0 1
    pixi run -e launch submit-nebius exec --wait -- nvidia-smi
    pixi run -e launch submit-nebius run --wait --forward-nebius-credentials -- \\
        --export s3://pd-nebius-ray/alpamayo-eval/w5s_3s --save-viz   # PD export, both arms

Resources default to one GPU on the ``h100`` pool (the 1-GPU H100 worker group); the 8-GPU
node is ``--resource h100x8=1 --num-gpus 8``.

SECURITY: Ray stores ``runtime_env.env_vars`` in the job spec and this cluster's dashboard
serves it UNAUTHENTICATED on port 8265. ``--forward-hf-token`` therefore exposes the token to
anyone on the VPN for as long as the job record lives. The entrypoint persists it once into the
cluster's HF cache, so forward it on the first run only. It is needed because the pod's own
cluster token can read the model but is not authorized for the gated PhysicalAI-AV dataset.
``--forward-nebius-credentials`` has the same exposure and is what every pd-fm Nebius job does
today (pods have no ambient object-store identity); the key is scoped to one bucket and rotatable,
and the switch goes away once the `nebius-object-storage` Secret lands in the worker pods.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import time
import uuid
from pathlib import Path

DEFAULT_ADDRESS = "http://ray-nebius.internal.paralleldomain.com:8265"
REPO_ROOT = Path(__file__).resolve().parents[1]
# Everything the worker does not need; .git alone is most of the checkout.
# Code only. Data (exports/) never rides in the Ray package: it lives in the Nebius bucket.
UPLOAD_EXCLUDES = [".git", ".pixi", ".venv", "outputs", "notebooks", "exports", "**/__pycache__", "*.png"]
NEBIUS_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
NEBIUS_REGION = "eu-north1"


def _hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        token_file = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "token"
        if token_file.is_file():
            token = token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("--forward-hf-token: no HF_TOKEN in env and no ~/.cache/huggingface/token")
    return token


def _nebius_credential_env(profile: str) -> dict[str, str]:
    """Resolve the Nebius key from a local AWS profile into the job env a pd-fm Nebius job gets.

    Same shape as pd-fm's ``_remote.aws_credential_env`` + ``ray_nebius.yaml`` env: the key as
    ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY`` and the endpoint/region pinned so every
    ``s3://`` in the job resolves to the Nebius bucket. Never the ``default`` (AWS SSO) profile.
    """
    import boto3

    if profile == "default":
        raise SystemExit("--forward-nebius-credentials must name the Nebius profile, never `default` (AWS SSO)")
    creds = boto3.Session(profile_name=profile).get_credentials()
    if creds is None:
        raise SystemExit(f"no credentials in AWS profile {profile!r} (expected the Nebius key under ~/.aws/credentials)")
    frozen = creds.get_frozen_credentials()
    env = {
        "AWS_ACCESS_KEY_ID": frozen.access_key,
        "AWS_SECRET_ACCESS_KEY": frozen.secret_key,
        "AWS_ENDPOINT_URL_S3": NEBIUS_ENDPOINT,
        "AWS_DEFAULT_REGION": NEBIUS_REGION,
    }
    if frozen.token:
        env["AWS_SESSION_TOKEN"] = frozen.token
    return env


def _parse_resources(items: list[str]) -> dict[str, float]:
    resources: dict[str, float] = {}
    for item in items:
        name, _, value = item.partition("=")
        if not name or not value:
            raise SystemExit(f"--resource expects name=value, got {item!r}")
        resources[name] = float(value)
    return resources


def _client(address: str):
    from ray.job_submission import JobSubmissionClient

    # The cluster may run a different ray than the pinned client; skip the strict check
    # (same workaround pd-fm and pd-ncc use).
    JobSubmissionClient._check_connection_and_version = lambda *a, **k: None  # type: ignore[method-assign]
    return JobSubmissionClient(address)


def wait_for_job(client, job_id: str, poll_s: float = 10.0) -> int:
    """Block until the job is terminal, streaming new log output. Returns an exit code."""
    from ray.job_submission import JobStatus

    terminal = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.STOPPED}
    last_status = None
    seen = 0
    while True:
        status = client.get_job_status(job_id)
        if status != last_status:
            print(f"--- ray job {job_id}: {status}", flush=True)
            last_status = status
        try:
            logs = client.get_job_logs(job_id)
            if len(logs) > seen:
                print(logs[seen:], end="", flush=True)
                seen = len(logs)
        except Exception:  # logs may be briefly unavailable; keep polling
            pass
        if status in terminal:
            return 0 if status == JobStatus.SUCCEEDED else 1
        time.sleep(poll_s)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["probe", "smoke", "run", "exec"], help="Entrypoint mode.")
    parser.add_argument(
        "--forward-nebius-credentials",
        nargs="?",
        const="nebius",
        default=None,
        metavar="PROFILE",
        help="Forward the Nebius object-store key from this local AWS profile (default: nebius) as job env, "
        "pinned to the Nebius endpoint. Needed for s3:// exports and result upload. See SECURITY.",
    )
    parser.add_argument("args", nargs="*", help="Passed through to the entrypoint after the mode.")
    parser.add_argument("--address", default=DEFAULT_ADDRESS, help=f"Ray dashboard (default: {DEFAULT_ADDRESS}).")
    parser.add_argument("--num-gpus", type=float, default=1, help="entrypoint_num_gpus (default: 1).")
    parser.add_argument("--num-cpus", type=float, default=None, help="entrypoint_num_cpus (default: unset).")
    parser.add_argument(
        "--resource",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Custom Ray resource tag; repeatable. Default: h100=1 (the 1-GPU H100 pool).",
    )
    parser.add_argument("--forward-hf-token", action="store_true", help="Forward the HF token as job env (first run only; see SECURITY).")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE", help="Extra job env var; repeatable.")
    parser.add_argument("--submission-id", default=None, help="Override the generated submission id.")
    parser.add_argument("--wait", action="store_true", help="Stream logs until the job finishes.")
    parser.add_argument("--dry-run", action="store_true", help="Print the job spec and exit.")
    args = parser.parse_args(argv)

    resources = _parse_resources(args.resource) or {"h100": 1}
    submission_id = args.submission_id or f"alpamayo-{args.mode}-{uuid.uuid4().hex[:8]}"

    env_vars = {"ALPAMAYO_EVAL_RUN_ID": submission_id, "PYTHONUNBUFFERED": "1"}
    if args.forward_hf_token:
        # Marker for the entrypoint: the pod env already carries a cluster HF_TOKEN (model access
        # only), so a forwarded HF_TOKEN is indistinguishable by name; this flag says it is ours.
        env_vars["ALPAMAYO_EVAL_TOKEN_FORWARDED"] = "1"
    for item in args.env:
        key, _, value = item.partition("=")
        if not key or not value:
            raise SystemExit(f"--env expects KEY=VALUE, got {item!r}")
        env_vars[key] = value
    if args.forward_hf_token:
        env_vars["HF_TOKEN"] = _hf_token()

    entrypoint = " ".join(shlex.quote(part) for part in ["bash", "scripts/nebius_entrypoint.sh", args.mode, *args.args])
    if args.forward_nebius_credentials:
        env_vars.update(_nebius_credential_env(args.forward_nebius_credentials))
    runtime_env = {"working_dir": str(REPO_ROOT), "excludes": UPLOAD_EXCLUDES, "env_vars": env_vars}

    print(f"submission : {submission_id}")
    print(f"entrypoint : {entrypoint}")
    print(f"resources  : gpus={args.num_gpus} cpus={args.num_cpus} tags={resources}")
    print(f"env        : {sorted(k for k in env_vars)}")
    if args.dry_run:
        return 0

    client = _client(args.address)
    job_id = client.submit_job(
        entrypoint=entrypoint,
        submission_id=submission_id,
        runtime_env=runtime_env,
        entrypoint_num_gpus=args.num_gpus,
        entrypoint_num_cpus=args.num_cpus,
        entrypoint_resources=resources,
    )
    print(f"submitted  : {job_id}")
    print(f"dashboard  : {args.address}/#/jobs/{job_id}")
    if args.wait:
        return wait_for_job(client, job_id)
    print(f"logs       : ray job logs -f --address {args.address} {job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Small boto3 helpers shared by the exporter, the mirror script and the cluster runner.

Clients are always built explicitly (profile and endpoint as arguments) rather than from the
process-global ``AWS_*`` environment, so one process can talk to AWS and Nebius at once (the same
reason pd-fm's ``mirror_datasets.py`` does it). On the Ray worker there is exactly one store, the
Nebius bucket, and ``client_from_env()`` builds that one from the forwarded job env.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

NEBIUS_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
NEBIUS_REGION = "eu-north1"
NEBIUS_BUCKET = "pd-nebius-ray"
NEBIUS_EVAL_PREFIX = "alpamayo-eval"  # s3://pd-nebius-ray/alpamayo-eval/<export name>/...
ARTIFACT_TMP_DIR = "tmp_alpamayo_eval"  # extra folder inside the source replipair artifact


def parse_s3(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3 uri: {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    return bucket, key.strip("/")


def is_s3(uri: str | os.PathLike) -> bool:
    return str(uri).startswith("s3://")


def client(profile: str | None = None, endpoint_url: str | None = None, region: str | None = None):
    import boto3

    from botocore.config import Config

    session = boto3.Session(profile_name=profile)
    if endpoint_url is None and profile == "nebius":
        endpoint_url, region = NEBIUS_ENDPOINT, region or NEBIUS_REGION
    # Laptop links drop: retry each request up to 10 times with backoff and allow slow reads, so a mirror or
    # staging survives a flaky connection instead of dying on the first ReadTimeout.
    config = Config(retries={"max_attempts": 10, "mode": "adaptive"}, connect_timeout=30, read_timeout=120, max_pool_connections=32)
    return session.client("s3", endpoint_url=endpoint_url, region_name=region, config=config)


def client_from_env():
    """The single store a Ray worker sees: credentials and endpoint from the forwarded job env."""
    return client(
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL_S3"),
        region=os.environ.get("AWS_DEFAULT_REGION"),
    )


def list_objects(s3, bucket: str, prefix: str) -> Iterator[tuple[str, int]]:
    """Yield ``(key, size)`` under ``prefix`` (``prefix`` is treated as a directory)."""
    prefix = prefix.rstrip("/") + "/" if prefix else ""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"], int(obj["Size"])


def object_size(s3, bucket: str, key: str) -> int | None:
    try:
        return int(s3.head_object(Bucket=bucket, Key=key)["ContentLength"])
    except s3.exceptions.ClientError as exc:
        if exc.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


def download_prefix(s3, uri: str, local_dir: Path, log=print) -> int:
    """Mirror ``uri`` (a directory) into ``local_dir``; skip files already present with the same size."""
    bucket, prefix = parse_s3(uri)
    n = 0
    for key, size in list_objects(s3, bucket, prefix):
        rel = key[len(prefix) :].lstrip("/")
        dst = local_dir / rel
        if dst.is_file() and dst.stat().st_size == size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f"{dst.name}.part-{os.getpid()}")  # per-process: concurrent stagers must not collide
        s3.download_file(bucket, key, str(tmp))
        os.replace(tmp, dst)
        n += 1
        if n % 50 == 0:
            log(f"  downloaded {n} files ...")
    return n


def upload_dir(s3, local_dir: Path, uri: str, log=print) -> int:
    """Upload every file under ``local_dir`` to ``uri`` (a directory); skip same-size objects."""
    bucket, prefix = parse_s3(uri)
    n = 0
    for path in sorted(p for p in local_dir.rglob("*") if p.is_file()):
        key = f"{prefix}/{path.relative_to(local_dir).as_posix()}"
        if object_size(s3, bucket, key) == path.stat().st_size:
            continue
        s3.upload_file(str(path), bucket, key)
        n += 1
        if n % 50 == 0:
            log(f"  uploaded {n} files ...")
    return n


def copy_between_stores(src, dst, src_uri: str, dst_uri: str, log=print, workers: int = 16) -> tuple[int, int]:
    """Stream every object under ``src_uri`` to ``dst_uri`` across two clients; skip same-size objects.

    Returns ``(copied, skipped)``. Objects stream through memory (frames are < 1 MB), ``workers``
    at a time; boto3 clients are thread-safe.
    """
    from concurrent.futures import ThreadPoolExecutor

    src_bucket, src_prefix = parse_s3(src_uri)
    dst_bucket, dst_prefix = parse_s3(dst_uri)
    existing = {key: size for key, size in list_objects(dst, dst_bucket, dst_prefix)}
    todo = []
    skipped = 0
    for key, size in list_objects(src, src_bucket, src_prefix):
        dst_key = f"{dst_prefix}/{key[len(src_prefix):].lstrip('/')}"
        if existing.get(dst_key) == size:
            skipped += 1
        else:
            todo.append((key, dst_key))

    def _copy(pair: tuple[str, str]) -> None:
        import time as _time

        for attempt in range(5):  # on top of botocore's per-request retries: a whole-object retry
            try:
                return _copy_once(pair)
            except Exception as exc:  # noqa: BLE001
                if attempt == 4:
                    raise
                log(f"  retry {attempt + 1} for {pair[0]}: {type(exc).__name__}")
                _time.sleep(2 * (attempt + 1))

    def _copy_once(pair: tuple[str, str]) -> None:
        key, dst_key = pair
        body = src.get_object(Bucket=src_bucket, Key=key)["Body"].read()
        dst.put_object(Bucket=dst_bucket, Key=dst_key, Body=body)

    copied = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(_copy, todo):
            copied += 1
            if copied % 100 == 0:
                log(f"  copied {copied}/{len(todo)} files ...")
    return copied, skipped

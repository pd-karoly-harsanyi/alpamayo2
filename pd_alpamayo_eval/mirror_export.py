"""Laptop-side copy between the artifact bucket (AWS) and the Nebius bucket, in either direction.

Nebius pods cannot reach AWS, and no AWS credential should ride in a Ray job, so exports are
mirrored from the laptop with two explicit profiles (same two-client idea as pd-fm's
``mirror_datasets.py``). Objects stream through memory one at a time; same-size objects are
skipped, so reruns are incremental.

    # export -> Nebius (default profile reads AWS, `nebius` profile writes Nebius)
    pixi run -e launch mirror-export \\
        s3://paralleldomain-pipeline/artifacts/replica_replipair_pdncc/<uid>/tmp_alpamayo_eval/w5s_3s \\
        s3://pd-nebius-ray/alpamayo-eval/w5s_3s

    # run results -> back next to the export in the artifact
    pixi run -e launch mirror-export \\
        s3://pd-nebius-ray/alpamayo-eval/w5s_3s/runs/<run id> \\
        s3://paralleldomain-pipeline/artifacts/replica_replipair_pdncc/<uid>/tmp_alpamayo_eval/w5s_3s/runs/<run id>

The profile for each side is inferred from the bucket (``pd-nebius-ray`` -> ``nebius``, anything
else -> ``default``) unless given explicitly.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.s3util import NEBIUS_BUCKET, client, copy_between_stores, list_objects, parse_s3  # noqa: E402


def _profile_for(uri: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    bucket, _ = parse_s3(uri)
    return "nebius" if bucket == NEBIUS_BUCKET else "default"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("src", help="s3:// directory to copy from")
    parser.add_argument("dst", help="s3:// directory to copy to")
    parser.add_argument("--src-profile", default=None)
    parser.add_argument("--dst-profile", default=None)
    parser.add_argument("--dry-run", action="store_true", help="List what would be copied.")
    args = parser.parse_args()

    src_profile, dst_profile = _profile_for(args.src, args.src_profile), _profile_for(args.dst, args.dst_profile)
    src, dst = client(src_profile), client(dst_profile)
    print(f"{args.src} [{src_profile}] -> {args.dst} [{dst_profile}]")
    src_bucket, src_prefix = parse_s3(args.src)
    objects = list(list_objects(src, src_bucket, src_prefix))
    total_mb = sum(size for _, size in objects) / 1e6
    print(f"source: {len(objects)} objects, {total_mb:.0f} MB")
    if not objects:
        print("nothing to copy (is the source a directory prefix?)")
        return 1
    if args.dry_run:
        return 0
    t = time.time()
    copied, skipped = copy_between_stores(src, dst, args.src, args.dst)
    print(f"copied {copied}, skipped {skipped} (already present), {time.time() - t:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())

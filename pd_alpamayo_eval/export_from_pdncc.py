"""Build an Alpamayo eval export (EXPORT_FORMAT.md) from a pdncc location bundle.

Runs in pdncc's pixi env (it needs the bundle/registry stack), not in this repo's env:

    cd ~/workspace/pd-neural-content-creation && pixi run python \\
        ~/workspace/alpamayo2/pd_alpamayo_eval/export_from_pdncc.py \\
        --bundle-id ce48f271-05b1-4f04-a1d1-af1472473ed9 --name w5s_3s --start-s 5 --duration-s 3

Reads the bundle's pdncc replipair artifact (``<camera>/original`` = real arm, ``<camera>/rendered``,
optional ``variations/<type>/<camera>/rendered``) and the original-data MCAP (10 Hz ego poses,
camera calibration), picks the instants on the ego pose grid whose four context frames exist for
all seven cameras, and writes the export next to its source: by default into
``<replipair artifact>/tmp_alpamayo_eval/<name>/`` in the artifact bucket, frames copied
server-side (same bucket, no download, bytes untouched). ``--out <dir>`` writes a local copy
instead. Nothing outside the new ``tmp_alpamayo_eval/<name>/`` folder is ever written.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.export_common import (  # noqa: E402
    ARMS,
    CANONICAL_CAMERA_NAMES,
    MANIFEST_NAME,
    build_manifest,
    frame_path,
    pose_record,
    select_instants,
    write_manifest,
)
from pd_alpamayo_eval.s3util import ARTIFACT_TMP_DIR, client, is_s3, object_size, parse_s3  # noqa: E402

ARM_TO_REPLIPAIR_DIR = {"real": "original", "rendered": "rendered"}
_TS_RE = re.compile(r"__(\d+)\.jpg$")


def _list_frames(camera_dir) -> dict[int, object]:
    """``{t_ns: AnyPath}`` for one ``<camera>/<original|rendered>`` directory."""
    if not camera_dir.is_dir():
        return {}
    out = {}
    for p in camera_dir.iterdir():
        m = _TS_RE.search(p.name)
        if m:
            out[int(m.group(1))] = p
    return out


def _calibration_record(calibration, cam_to_ego) -> dict:
    rec: dict = {"width": int(calibration.width), "height": int(calibration.height)}
    v = getattr(calibration, "distortion_model", None)
    if v is not None:
        rec["distortion_model"] = str(getattr(v, "name", v))
    for attr in ("K", "D"):
        v = getattr(calibration, attr, None)
        if v is not None:
            rec[attr] = np.asarray(v, dtype=float).tolist()
    q = cam_to_ego.quaternion
    rec["cam_to_ego"] = {
        **pose_record(cam_to_ego.translation, (q.w, q.x, q.y, q.z)),
        "source_axes": str(cam_to_ego.source_coordinate_system),
        "target_axes": str(cam_to_ego.target_coordinate_system),
    }
    return rec


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle-id", required=True)
    parser.add_argument("--name", required=True, help="Export name, e.g. w5s_3s; becomes the folder name.")
    parser.add_argument(
        "--out",
        default=None,
        help=f"Destination: a local directory, or an s3:// prefix. Default: <replipair artifact>/{ARTIFACT_TMP_DIR}/<name>.",
    )
    parser.add_argument("--start-s", type=float, default=0.0, help="Window start, seconds after the first ego pose.")
    parser.add_argument("--duration-s", type=float, default=None, help="Window length in seconds (default: to the end).")
    parser.add_argument("--max-instants", type=int, default=None, help="Cap on the number of instants exported.")
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS, help="Required arms (define the instants).")
    parser.add_argument(
        "--no-variations",
        action="store_true",
        help="Skip the optional weather/time-of-day arms (`variations/<type>/<camera>/rendered`, exported as `rendered_<type>`).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Select instants and print, write nothing.")
    args = parser.parse_args()

    from neural_content_creation.common.location_bundle import LocationBundle

    bundle = LocationBundle.from_artifact_id(args.bundle_id)
    original = bundle.original_data_artifact
    replipair = bundle.pdncc_replipair_artifact
    if replipair is None:
        raise SystemExit("bundle has no pdncc replipair artifact")
    sessions = original.sessions
    if len(sessions) != 1:
        raise SystemExit(f"expected one capture session, got {[s.session_id for s in sessions]}")
    session = sessions[0]
    out = args.out or f"{str(replipair.path).rstrip('/')}/{ARTIFACT_TMP_DIR}/{args.name}"
    print(f"session {session.session_id}\nreplipair {replipair.path}\nexport -> {out}")

    # Ego poses: (t_ns, ego_to_world Transformation) in FLU, 10 Hz for PAI-AV bundles.
    ego_poses = {}
    for t_ns, pose in session.iter_ego_poses():
        q = pose.quaternion
        ego_poses[int(t_ns)] = pose_record(pose.translation, (q.w, q.x, q.y, q.z))
    pose_ts = sorted(ego_poses)
    print(f"ego poses: {len(pose_ts)}, span {(pose_ts[-1] - pose_ts[0]) / 1e9:.1f} s")

    # Frames per arm per camera from the replipair listing.
    frames: dict[str, dict[str, dict[int, object]]] = {}
    for arm in args.arms:
        frames[arm] = {}
        for camera in CANONICAL_CAMERA_NAMES:
            frames[arm][camera] = _list_frames(replipair.path / camera / ARM_TO_REPLIPAIR_DIR[arm])
            print(f"  {arm:16s} {camera:28s} {len(frames[arm][camera])} frames")
    common_ts = {
        camera: set.intersection(*(set(frames[arm][camera]) for arm in args.arms)) for camera in CANONICAL_CAMERA_NAMES
    }
    # Optional arms: appearance variations rendered over the same poses (not every bundle has them).
    variations_dir = replipair.path / "variations"
    if not args.no_variations and variations_dir.is_dir():
        for vdir in sorted(variations_dir.iterdir()):
            if not vdir.is_dir():
                continue
            arm = f"rendered_{vdir.name}"
            frames[arm] = {camera: _list_frames(vdir / camera / "rendered") for camera in CANONICAL_CAMERA_NAMES}
            print(f"  {arm:16s} " + " ".join(str(len(frames[arm][c])) for c in CANONICAL_CAMERA_NAMES) + " frames")
    optional_arms = [arm for arm in frames if arm not in args.arms]

    window_start = pose_ts[0] + int(args.start_s * 1e9)
    window_end = pose_ts[-1] if args.duration_s is None else window_start + int(args.duration_s * 1e9)
    candidates = [t for t in pose_ts if window_start <= t <= window_end]
    instants = select_instants(
        candidates,
        common_ts,
        pose_ts,
        required_arms=args.arms,
        optional_arm_frames={arm: frames[arm] for arm in optional_arms},
    )
    if args.max_instants is not None:
        instants = instants[: args.max_instants]
    print(f"instants: {len(instants)} of {len(candidates)} candidates in window")
    for arm in optional_arms:
        print(f"  {arm}: complete on {sum(arm in inst['arms'] for inst in instants)} instants")
    if not instants:
        raise SystemExit("no instant has four context frames on all seven cameras; widen the window")
    if args.dry_run:
        for inst in instants:
            print("  t0_ns", inst["t0_ns"], inst["arms"])
        return
    arms_present = list(args.arms) + [arm for arm in optional_arms if any(arm in inst["arms"] for inst in instants)]

    cameras = {}
    cam_to_egos = session.camera_to_egos_by_camera_name
    calibrations = session.camera_calibrations
    calibrations = calibrations() if callable(calibrations) else calibrations
    by_frame_id = {str(getattr(c, "frame_id", "")): c for c in calibrations}
    for camera in CANONICAL_CAMERA_NAMES:
        cal = by_frame_id.get(camera)
        if cal is not None and camera in cam_to_egos:
            cameras[camera] = _calibration_record(cal, cam_to_egos[camera])

    manifest = build_manifest(
        scene=session.session_id,
        ego_poses=ego_poses,
        instants=instants,
        arms=arms_present,
        cameras=cameras,
        source={
            "bundle_id": args.bundle_id,
            "replipair_artifact": str(replipair.path),
            "original_data_artifact": str(original.path),
            "export_uri": str(out),
            "window": {"start_s": args.start_s, "duration_s": args.duration_s},
        },
    )
    needed = sorted({(arm, camera, t) for inst in instants for camera, ts in inst["frames"].items() for arm in inst["arms"] for t in ts})

    if is_s3(out):
        s3 = client()  # the artifact bucket: default (SSO) profile
        dst_bucket, dst_prefix = parse_s3(out)
        s3.put_object(Bucket=dst_bucket, Key=f"{dst_prefix}/{MANIFEST_NAME}", Body=json.dumps(manifest, indent=1).encode())
        print(f"copying {len(needed)} frames server-side to {out} ...")
        copied = 0
        for i, (arm, camera, t) in enumerate(needed, 1):
            src_bucket, src_key = parse_s3(str(frames[arm][camera][t]))
            dst_key = f"{dst_prefix}/{arm}/{camera}/{t}.jpg"
            if object_size(s3, dst_bucket, dst_key) is not None:
                continue
            s3.copy_object(CopySource={"Bucket": src_bucket, "Key": src_key}, Bucket=dst_bucket, Key=dst_key)
            copied += 1
            if i % 50 == 0:
                print(f"  {i}/{len(needed)}")
        print(f"done: {len(instants)} instants, {len(needed)} frames ({copied} new) -> {out}")
    else:
        out_dir = Path(out)
        write_manifest(out_dir, manifest)
        print(f"copying {len(needed)} frames to {out_dir} ...")
        for i, (arm, camera, t) in enumerate(needed, 1):
            dst = frame_path(out_dir, arm, camera, t)
            if dst.exists():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            with frames[arm][camera][t].open("rb") as src, dst.open("wb") as fh:
                fh.write(src.read())
            if i % 50 == 0:
                print(f"  {i}/{len(needed)}")
        total_mb = sum(p.stat().st_size for p in out_dir.rglob("*.jpg")) / 1e6
        print(f"done: {len(instants)} instants, {len(needed)} frames, {total_mb:.0f} MB -> {out_dir}")


if __name__ == "__main__":
    main()

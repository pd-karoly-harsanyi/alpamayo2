"""Pure numpy/stdlib pieces shared by the pdncc exporter, the loader and the tests.

The export contract itself is documented in EXPORT_FORMAT.md. This module owns the two things
both sides must agree on: how context frames are matched to an instant, and the manifest schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MANIFEST_VERSION = 1
MANIFEST_NAME = "manifest.json"
ARMS = ("real", "rendered")
CANONICAL_CAMERA_NAMES = (
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_rear_left_70fov",
    "camera_rear_tele_30fov",
    "camera_rear_right_70fov",
    "camera_front_tele_30fov",
)
STEP_NS = 100_000_000  # the model's 10 Hz
NUM_CONTEXT_FRAMES = 4  # t0 - 0.3, -0.2, -0.1, 0 s
NUM_HISTORY_STEPS = 16
NUM_FUTURE_STEPS = 64
DEFAULT_FRAME_TOLERANCE_NS = 20_000_000  # 30 fps source: worst case 16.7 ms off the grid


def nearest(sorted_ts: np.ndarray, target: int) -> int:
    """Nearest value of a sorted int64 array to ``target``."""
    i = int(np.searchsorted(sorted_ts, target))
    candidates = sorted_ts[max(i - 1, 0) : i + 1]
    return int(candidates[np.argmin(np.abs(candidates - target))])


def match_context_frames(
    frame_ts_by_camera: dict[str, Sequence[int]],
    t0_ns: int,
    tolerance_ns: int = DEFAULT_FRAME_TOLERANCE_NS,
) -> dict[str, list[int]] | None:
    """Pick, per camera, the frame nearest to each of t0 - 0.3/0.2/0.1/0 s.

    Returns ``None`` when any camera has no frame within ``tolerance_ns`` of a target, or when
    two targets resolve to the same frame (source frame rate below 10 Hz).
    """
    picked: dict[str, list[int]] = {}
    for camera, ts in frame_ts_by_camera.items():
        arr = np.asarray(sorted(int(t) for t in ts), dtype=np.int64)
        if arr.size == 0:
            return None
        frames = []
        for k in range(NUM_CONTEXT_FRAMES - 1, -1, -1):
            target = t0_ns - k * STEP_NS
            best = nearest(arr, target)
            if abs(best - target) > tolerance_ns:
                return None
            frames.append(best)
        if len(set(frames)) != NUM_CONTEXT_FRAMES:
            return None
        picked[camera] = frames
    return picked


def select_instants(
    candidate_t0_ns: Sequence[int],
    frame_ts_by_camera: dict[str, Sequence[int]],
    pose_ts_ns: Sequence[int],
    tolerance_ns: int = DEFAULT_FRAME_TOLERANCE_NS,
    required_arms: Sequence[str] = ARMS,
    optional_arm_frames: dict[str, dict[str, Sequence[int]]] | None = None,
) -> list[dict[str, Any]]:
    """Keep the candidates for which all cameras have context frames and the pose log covers the history.

    ``frame_ts_by_camera`` holds the timestamps present in every required arm (their intersection).
    ``optional_arm_frames`` (``{arm: {camera: timestamps}}``, e.g. weather variations) never
    decides whether an instant exists; an optional arm is listed in the instant's ``arms`` only when
    it has the exact same 28 frames.
    """
    pose_ts = np.asarray(sorted(int(t) for t in pose_ts_ns), dtype=np.int64)
    history_span = (NUM_HISTORY_STEPS - 1) * STEP_NS
    optional = {arm: {cam: set(int(t) for t in ts) for cam, ts in cams.items()} for arm, cams in (optional_arm_frames or {}).items()}
    instants = []
    for t0 in sorted(int(t) for t in candidate_t0_ns):
        if t0 - history_span < pose_ts[0] or t0 > pose_ts[-1]:
            continue
        frames = match_context_frames(frame_ts_by_camera, t0, tolerance_ns)
        if frames is None:
            continue
        arms = list(required_arms)
        for arm, cams in optional.items():
            if all(set(ts) <= cams.get(cam, set()) for cam, ts in frames.items()):
                arms.append(arm)
        instants.append({"t0_ns": t0, "frames": frames, "arms": arms})
    return instants


def pose_record(xyz: Sequence[float], quat_wxyz: Sequence[float]) -> dict[str, float]:
    x, y, z = (float(v) for v in xyz)
    qw, qx, qy, qz = (float(v) for v in quat_wxyz)
    return {"x": x, "y": y, "z": z, "qw": qw, "qx": qx, "qy": qy, "qz": qz}


def build_manifest(
    scene: str,
    ego_poses: dict[int, dict[str, float]],
    instants: list[dict[str, Any]],
    arms: Sequence[str],
    ego_pose_convention: dict[str, str] | None = None,
    cameras: dict[str, Any] | None = None,
    source: dict[str, Any] | None = None,
    nav_text: str | None = None,
) -> dict[str, Any]:
    pose_ts = sorted(ego_poses)
    return {
        "version": MANIFEST_VERSION,
        "scene": scene,
        "arms": list(arms),
        "time_origin_ns": pose_ts[0],
        "ego_pose_convention": ego_pose_convention or {"direction": "ego_to_world", "axes": "FLU"},
        "ego_poses": {str(t): ego_poses[t] for t in pose_ts},
        "instants": instants,
        "nav_text": nav_text,
        "cameras": cameras or {},
        "source": source or {},
    }


def frame_path(export_dir: Path, arm: str, camera: str, t_ns: int) -> Path:
    return export_dir / arm / camera / f"{t_ns}.jpg"


def write_manifest(export_dir: Path, manifest: dict[str, Any]) -> Path:
    export_dir.mkdir(parents=True, exist_ok=True)
    path = export_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def read_manifest(export_dir: Path) -> dict[str, Any]:
    manifest = json.loads((export_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if manifest.get("version") != MANIFEST_VERSION:
        raise ValueError(f"{export_dir}: manifest version {manifest.get('version')} != {MANIFEST_VERSION}")
    return manifest

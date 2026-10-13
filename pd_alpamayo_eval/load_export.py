"""Load one instant of an Alpamayo eval export as the dict ``load_physical_aiavdataset()`` returns.

Same keys, shapes, dtypes and conventions as NVIDIA's loader (``src/alpamayo2_super/
load_physical_aiavdataset.py``), so ``select_task_input`` / ``helper.prepare_model_inputs`` /
``sample_trajectories_from_data`` run untouched on it. Differences: images come from JPEG files
instead of the PhysicalAI-AV devkit, ego poses come from the manifest's pose log (interpolated to
the 10 Hz grid: linear xyz, slerp rotation), and ``camera_calibrations`` is omitted (overlays
fall back to BEV). The GT future is included only when the pose log covers 6.4 s past t0.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES
from pd_alpamayo_eval.export_common import (
    CANONICAL_CAMERA_NAMES,
    NUM_CONTEXT_FRAMES,
    NUM_FUTURE_STEPS,
    NUM_HISTORY_STEPS,
    STEP_NS,
    frame_path,
    read_manifest,
)

# Rows map a vector in the named convention onto FLU (x forward, y left, z up).
AXES_TO_FLU = {
    "FLU": np.eye(3),
    "RFU": np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
}


class EgoTrajectory:
    """Dense ego-to-world pose log in FLU, sampled by interpolation."""

    def __init__(self, manifest: dict[str, Any]):
        poses = manifest["ego_poses"]
        convention = manifest.get("ego_pose_convention", {})
        ts = np.array(sorted(int(t) for t in poses), dtype=np.int64)
        if ts.size < 2:
            raise ValueError("ego_poses needs at least two poses")
        xyz = np.array([[poses[str(t)][k] for k in ("x", "y", "z")] for t in ts], dtype=np.float64)
        quat_xyzw = np.array([[poses[str(t)][k] for k in ("qx", "qy", "qz", "qw")] for t in ts], dtype=np.float64)
        rot = Rotation.from_quat(quat_xyzw)
        if convention.get("direction", "ego_to_world") == "world_to_ego":
            rot = rot.inv()
            xyz = -rot.apply(xyz)
        axes = convention.get("axes", "FLU")
        if axes not in AXES_TO_FLU:
            raise ValueError(f"unsupported ego_pose_convention.axes {axes!r}; expected one of {list(AXES_TO_FLU)}")
        c = AXES_TO_FLU[axes]
        if not np.array_equal(c, np.eye(3)):
            xyz = xyz @ c.T
            rot = Rotation.from_matrix(c @ rot.as_matrix() @ c.T)
        self.ts = ts
        self.xyz = xyz
        self.rot = rot
        self._t_rel = (ts - ts[0]).astype(np.float64) * 1e-9
        self._slerp = Slerp(self._t_rel, rot)

    def covers(self, t_ns: int) -> bool:
        return int(self.ts[0]) <= int(t_ns) <= int(self.ts[-1])

    def sample(self, t_ns: np.ndarray) -> tuple[np.ndarray, Rotation]:
        t = np.asarray(t_ns, dtype=np.int64)
        if t.min() < self.ts[0] or t.max() > self.ts[-1]:
            raise ValueError(
                f"timestamps {t.min()}..{t.max()} outside the pose log {self.ts[0]}..{self.ts[-1]}"
            )
        t_rel = (t - self.ts[0]).astype(np.float64) * 1e-9
        xyz = np.stack([np.interp(t_rel, self._t_rel, self.xyz[:, i]) for i in range(3)], axis=-1)
        return xyz, self._slerp(t_rel)


@functools.lru_cache(maxsize=8)
def _manifest(export_dir: str) -> dict[str, Any]:
    return read_manifest(Path(export_dir))


@functools.lru_cache(maxsize=8)
def _trajectory(export_dir: str) -> EgoTrajectory:
    return EgoTrajectory(_manifest(export_dir))


def list_instants(export_dir: str | Path) -> list[int]:
    return [int(inst["t0_ns"]) for inst in _manifest(str(Path(export_dir)))["instants"]]


def list_arms(export_dir: str | Path, t0_ns: int | None = None) -> list[str]:
    """Arms of the export, or of one instant (optional arms such as weather variations may be missing)."""
    manifest = _manifest(str(Path(export_dir)))
    if t0_ns is None:
        return list(manifest["arms"])
    for inst in manifest["instants"]:
        if int(inst["t0_ns"]) == int(t0_ns):
            return list(inst.get("arms", manifest["arms"]))
    raise KeyError(f"t0_ns {t0_ns} is not an instant of {export_dir}")


def _load_frames(paths: list[Path]) -> torch.Tensor:
    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as im:
            frames.append(np.asarray(im.convert("RGB")))
    shapes = {f.shape for f in frames}
    if len(shapes) != 1:
        raise ValueError(f"context frames differ in shape: {shapes} ({paths[0].parent})")
    return rearrange(torch.from_numpy(np.stack(frames)), "t h w c -> t c h w")


def load_export(
    export_dir: str | Path,
    t0_ns: int,
    arm: str = "real",
    include_future: bool = True,
    time_step: float = 0.1,
) -> dict[str, Any]:
    """Return NVIDIA's inference dict for instant ``t0_ns`` of ``export_dir`` using arm ``arm``."""
    export_dir = Path(export_dir)
    manifest = _manifest(str(export_dir))
    if arm not in manifest["arms"]:
        raise ValueError(f"arm {arm!r} not in export arms {manifest['arms']}")
    instants = {int(inst["t0_ns"]): inst for inst in manifest["instants"]}
    if int(t0_ns) not in instants:
        raise KeyError(f"t0_ns {t0_ns} is not an instant of {export_dir}")
    instant = instants[int(t0_ns)]
    if arm not in instant.get("arms", manifest["arms"]):
        raise ValueError(f"arm {arm!r} is not available for instant {t0_ns} (has {instant.get('arms')})")
    t0_ns = int(t0_ns)
    origin_ns = int(manifest["time_origin_ns"])
    trajectory = _trajectory(str(export_dir))

    history_ts = t0_ns + np.arange(-(NUM_HISTORY_STEPS - 1), 1, dtype=np.int64) * STEP_NS
    future_ts = t0_ns + np.arange(1, NUM_FUTURE_STEPS + 1, dtype=np.int64) * STEP_NS
    history_xyz, history_rot = trajectory.sample(history_ts)
    t0_xyz = history_xyz[-1].copy()
    t0_rot = history_rot[-1]
    t0_rot_inv = t0_rot.inv()
    history_xyz_local = t0_rot_inv.apply(history_xyz - t0_xyz)
    history_rot_local = (t0_rot_inv * history_rot).as_matrix()

    future_available = include_future and trajectory.covers(int(future_ts[-1]))
    if future_available:
        future_xyz, future_rot = trajectory.sample(future_ts)
        future_xyz_local = t0_rot_inv.apply(future_xyz - t0_xyz)
        future_rot_local = (t0_rot_inv * future_rot).as_matrix()

    image_frames_list, timestamps_list, camera_indices_list = [], [], []
    for camera in CANONICAL_CAMERA_NAMES:
        frame_ts = [int(t) for t in instant["frames"][camera]]
        if len(frame_ts) != NUM_CONTEXT_FRAMES:
            raise ValueError(f"{camera}: expected {NUM_CONTEXT_FRAMES} context frames, got {len(frame_ts)}")
        image_frames_list.append(_load_frames([frame_path(export_dir, arm, camera, t) for t in frame_ts]))
        timestamps_list.append(torch.tensor([(t - origin_ns) // 1000 for t in frame_ts], dtype=torch.int64))
        camera_indices_list.append(CAMERA_NAMES_TO_INDICES[camera])

    image_frames = torch.stack(image_frames_list, dim=0)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
    absolute_timestamps = torch.stack(timestamps_list, dim=0)
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    absolute_timestamps = absolute_timestamps[sort_order]
    camera_names = [CANONICAL_CAMERA_NAMES[i] for i in sort_order.tolist()]

    camera_tmin_tensor = absolute_timestamps.min()
    camera_tmin = int(camera_tmin_tensor.item())
    relative_timestamps = (absolute_timestamps - camera_tmin_tensor).float() * 1e-6
    ego_t0_us = (t0_ns - origin_ns) // 1000
    t0_quat = t0_rot.as_quat()  # xyzw
    t0_inv_quat_wxyz = np.array([t0_quat[3], -t0_quat[0], -t0_quat[1], -t0_quat[2]], dtype=np.float32)

    data: dict[str, Any] = {
        "image_frames": image_frames,
        "camera_indices": camera_indices,
        "camera_names": camera_names,
        "ego_available": torch.tensor(True),
        "ego_t0": torch.tensor([ego_t0_us], dtype=torch.int64),
        "ego_t0_relative": torch.tensor([(ego_t0_us - camera_tmin) * 1e-6], dtype=torch.float32),
        "ego_t0_frame_idx": torch.tensor([NUM_CONTEXT_FRAMES - 1], dtype=torch.int64),
        "prediction_start_offset": torch.zeros(1, dtype=torch.float32),
        "ego_history_tvals": torch.arange(-NUM_HISTORY_STEPS + 1, 1, dtype=torch.float32) * float(time_step),
        "ego_history_xyz": torch.from_numpy(history_xyz_local).float().unsqueeze(0).unsqueeze(0),
        "ego_history_rot": torch.from_numpy(history_rot_local).float().unsqueeze(0).unsqueeze(0),
        "ego_future_tvals": torch.arange(1, NUM_FUTURE_STEPS + 1, dtype=torch.float32) * float(time_step),
        "ego_future_available": future_available,
        "relative_timestamps": relative_timestamps,
        "absolute_timestamps": absolute_timestamps,
        "camera_tmin": camera_tmin,
        "ego_t0_xyz": torch.from_numpy(t0_xyz).float().view(1, 1, 3),
        "ego_t0_inv_quat": torch.from_numpy(t0_inv_quat_wxyz).float().view(1, 1, 4),
        "t0_us": ego_t0_us,
        "t0_ns": t0_ns,
        "clip_id": f"{manifest['scene']}/{arm}",
        "scene": manifest["scene"],
        "arm": arm,
        "export_dir": str(export_dir),
        "nav_text": manifest.get("nav_text"),
    }
    if future_available:
        data["ego_future_xyz"] = torch.from_numpy(future_xyz_local).float().unsqueeze(0).unsqueeze(0)
        data["ego_future_rot"] = torch.from_numpy(future_rot_local).float().unsqueeze(0).unsqueeze(0)
    return data

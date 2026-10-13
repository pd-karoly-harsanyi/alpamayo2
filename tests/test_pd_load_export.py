"""CPU tests for the PD export loader: a synthetic export must load as NVIDIA's input dict.

Run with ``pixi run -e dev test-pd`` (PYTHONPATH=src:. so ``alpamayo2_super`` imports without
the model environment).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from pd_alpamayo_eval.export_common import (
    CANONICAL_CAMERA_NAMES,
    STEP_NS,
    build_manifest,
    frame_path,
    match_context_frames,
    pose_record,
    select_instants,
    write_manifest,
)
from pd_alpamayo_eval.load_export import list_instants, load_export

SPEED = 5.0  # m/s
HEADING = math.radians(30.0)
IMG_W, IMG_H = 48, 27
CAMERA_OFFSET_NS = {name: i * 3_000_000 for i, name in enumerate(CANONICAL_CAMERA_NAMES)}  # unsynchronized ring


def _straight_line_poses_flu(t_ns: np.ndarray) -> dict[int, dict[str, float]]:
    """Ego driving straight at constant heading: world xyz and yaw rotation, FLU, ego-to-world."""
    poses = {}
    for t in t_ns:
        s = SPEED * (t - t_ns[0]) * 1e-9
        xyz = (s * math.cos(HEADING), s * math.sin(HEADING), 0.0)
        q = Rotation.from_euler("z", HEADING).as_quat()  # xyzw
        poses[int(t)] = pose_record(xyz, (q[3], q[0], q[1], q[2]))
    return poses


def _to_rfu(poses_flu: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """Re-express the same ego-to-world poses with world and ego axes in RFU (x right, y forward)."""
    c = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # FLU -> RFU
    out = {}
    for t, p in poses_flu.items():
        xyz = c @ np.array([p["x"], p["y"], p["z"]])
        r = Rotation.from_quat([p["qx"], p["qy"], p["qz"], p["qw"]]).as_matrix()
        q = Rotation.from_matrix(c @ r @ c.T).as_quat()
        out[t] = pose_record(xyz, (q[3], q[0], q[1], q[2]))
    return out


def _make_export(root: Path, axes: str, arms=("real", "rendered"), fps: float = 30.0, seconds: float = 4.0) -> Path:
    rng = np.random.default_rng(0)
    pose_ts = np.arange(0, int(12e9) + 1, STEP_NS, dtype=np.int64) + 1_700_000_000_000_000_000
    poses = _straight_line_poses_flu(pose_ts)
    if axes == "RFU":
        poses = _to_rfu(poses)
    frame_ts = {
        cam: [int(pose_ts[0] + off + k * 1e9 / fps) for k in range(int(seconds * fps))]
        for cam, off in CAMERA_OFFSET_NS.items()
    }
    instants = select_instants(pose_ts, frame_ts, pose_ts)
    manifest = build_manifest(
        scene="synthetic",
        ego_poses=poses,
        instants=instants,
        arms=arms,
        ego_pose_convention={"direction": "ego_to_world", "axes": axes},
    )
    write_manifest(root, manifest)
    for inst in instants:
        for cam, ts in inst["frames"].items():
            for t in ts:
                for arm in arms:
                    p = frame_path(root, arm, cam, t)
                    if not p.exists():
                        p.parent.mkdir(parents=True, exist_ok=True)
                        Image.fromarray(rng.integers(0, 255, (IMG_H, IMG_W, 3), dtype=np.uint8)).save(p, quality=95)
    return root


@pytest.fixture(params=["FLU", "RFU"])
def export_dir(tmp_path, request):
    return _make_export(tmp_path / f"export_{request.param}", axes=request.param)


def test_instant_selection_needs_history_and_all_frames(tmp_path):
    export = _make_export(tmp_path / "e", axes="FLU", seconds=4.0)
    instants = list_instants(export)
    # history needs 1.5 s of poses before t0, context frames need 0.3 s of frames before t0
    assert instants[0] - 1_700_000_000_000_000_000 == 15 * STEP_NS
    # frames stop at 4 s (+ the largest camera offset), so no instant beyond that
    assert all(t - 1_700_000_000_000_000_000 <= 4 * 10 * STEP_NS for t in instants)
    assert len(instants) > 20


def test_match_context_frames_rejects_sparse_or_missing():
    t0 = 10 * STEP_NS
    sparse = {cam: list(range(0, 20 * STEP_NS, 2 * STEP_NS)) for cam in CANONICAL_CAMERA_NAMES}  # 5 Hz
    assert match_context_frames(sparse, t0) is None
    dense = {cam: list(range(0, 20 * STEP_NS, STEP_NS // 3)) for cam in CANONICAL_CAMERA_NAMES}
    assert match_context_frames(dense, t0) is not None
    missing = dict(dense)
    missing["camera_rear_tele_30fov"] = [0]
    assert match_context_frames(missing, t0) is None


def test_load_export_matches_nvidia_contract(export_dir):
    t0 = list_instants(export_dir)[5]
    data = load_export(export_dir, t0, arm="real")

    assert data["image_frames"].shape == (7, 4, 3, IMG_H, IMG_W)
    assert data["image_frames"].dtype == torch.uint8
    assert data["camera_indices"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert data["camera_names"] == list(CANONICAL_CAMERA_NAMES)
    assert data["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert data["ego_history_rot"].shape == (1, 1, 16, 3, 3)
    assert data["ego_future_xyz"].shape == (1, 1, 64, 3)
    assert data["ego_future_rot"].shape == (1, 1, 64, 3, 3)
    assert data["absolute_timestamps"].shape == (7, 4)
    assert data["absolute_timestamps"].dtype == torch.int64
    assert data["ego_t0_frame_idx"].item() == 3

    # Ego frame at t0: last history point is the origin with identity rotation; the car moved
    # straight ahead, so history is along -x (behind) and future along +x (ahead), y ~ 0 (FLU).
    hist = data["ego_history_xyz"][0, 0].numpy()
    fut = data["ego_future_xyz"][0, 0].numpy()
    assert np.allclose(hist[-1], 0, atol=1e-6)
    assert np.allclose(data["ego_history_rot"][0, 0, -1].numpy(), np.eye(3), atol=1e-6)
    assert np.allclose(hist[:, 0], -SPEED * 0.1 * np.arange(15, -1, -1), atol=1e-4)
    assert np.allclose(hist[:, 1], 0, atol=1e-4)
    assert np.allclose(fut[:, 0], SPEED * 0.1 * np.arange(1, 65), atol=1e-4)
    assert np.allclose(fut[:, 1], 0, atol=1e-4)

    # Per-camera timestamps: four frames ~100 ms apart ending near t0, in clip-relative microseconds.
    ts = data["absolute_timestamps"].numpy()
    assert np.all(np.diff(ts, axis=1) > 0)
    assert np.allclose(np.diff(ts, axis=1), 100_000, atol=20_000)
    assert np.all(np.abs(ts[:, -1] - data["ego_t0"].item()) <= 20_000)
    assert data["camera_tmin"] == int(ts.min())
    assert np.isclose(data["relative_timestamps"].numpy().min(), 0.0)


def test_select_task_input_accepts_export(export_dir):
    from alpamayo2_super.input_profiles import select_task_input

    t0 = list_instants(export_dir)[0]
    for arm in ("real", "rendered"):
        selected = select_task_input(load_export(export_dir, t0, arm=arm), "trajectory")
        assert selected["camera_indices"].tolist() == [0, 1, 2, 3, 5, 6]
        assert selected["image_frames"].shape[:2] == (6, 4)
        assert selected["ego_t0_frame_idx"].item() == 3


def test_future_omitted_when_pose_log_ends(tmp_path):
    export = _make_export(tmp_path / "short", axes="FLU", seconds=11.5)
    last = list_instants(export)[-1]  # within 6.4 s of the 12 s pose log end
    data = load_export(export, last)
    assert data["ego_future_available"] is False
    assert "ego_future_xyz" not in data


def test_optional_arm_listed_only_where_complete(tmp_path):
    from pd_alpamayo_eval.load_export import list_arms

    pose_ts = np.arange(0, int(12e9) + 1, STEP_NS, dtype=np.int64) + 1_700_000_000_000_000_000
    frame_ts = {cam: [int(pose_ts[0] + off + k * 1e9 / 30) for k in range(120)] for cam, off in CAMERA_OFFSET_NS.items()}
    # the variation only covers the first 2.5 s of frames
    variation = {cam: [t for t in ts if t < pose_ts[0] + 2_500_000_000] for cam, ts in frame_ts.items()}
    instants = select_instants(pose_ts, frame_ts, pose_ts, optional_arm_frames={"rendered_snowy": variation})
    with_var = [i for i in instants if "rendered_snowy" in i["arms"]]
    without = [i for i in instants if "rendered_snowy" not in i["arms"]]
    assert with_var and without
    assert all(i["t0_ns"] < pose_ts[0] + 2_500_000_000 for i in with_var)
    assert all(i["arms"][:2] == ["real", "rendered"] for i in instants)

    root = tmp_path / "v"
    write_manifest(root, build_manifest("s", _straight_line_poses_flu(pose_ts), instants, ["real", "rendered", "rendered_snowy"]))
    assert list_arms(root) == ["real", "rendered", "rendered_snowy"]
    assert list_arms(root, with_var[0]["t0_ns"]) == ["real", "rendered", "rendered_snowy"]
    assert list_arms(root, without[0]["t0_ns"]) == ["real", "rendered"]
    with pytest.raises(ValueError):
        load_export(root, without[0]["t0_ns"], arm="rendered_snowy")


def test_unknown_arm_or_instant(export_dir):
    t0 = list_instants(export_dir)[0]
    with pytest.raises(ValueError):
        load_export(export_dir, t0, arm="blind")
    with pytest.raises(KeyError):
        load_export(export_dir, t0 + 1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for assembling the inference input dict from arrays."""

import numpy as np
import pytest
import torch

from alpamayo2_super.common.constants import CAMERA_NAMES
from alpamayo2_super.inference_sample import inference_sample_from_arrays
from alpamayo2_super.input_profiles import assert_task_input, select_task_input

NUM_FRAMES, H, W = 4, 8, 8
NUM_HISTORY, NUM_FUTURE = 16, 64
T0_US = 5_100_000

# key -> (dtype, shape); None entries are python scalars / lists checked separately
CONTRACT = {
    "image_frames": (torch.uint8, (None, NUM_FRAMES, 3, H, W)),
    "camera_indices": (torch.int64, (None,)),
    "ego_available": (torch.bool, ()),
    "ego_t0": (torch.int64, (1,)),
    "ego_t0_relative": (torch.float32, (1,)),
    "ego_t0_frame_idx": (torch.int64, (1,)),
    "prediction_start_offset": (torch.float32, (1,)),
    "ego_history_tvals": (torch.float32, (NUM_HISTORY,)),
    "ego_history_xyz": (torch.float32, (1, 1, NUM_HISTORY, 3)),
    "ego_history_rot": (torch.float32, (1, 1, NUM_HISTORY, 3, 3)),
    "ego_future_tvals": (torch.float32, (NUM_FUTURE,)),
    "ego_future_xyz": (torch.float32, (1, 1, NUM_FUTURE, 3)),
    "ego_future_rot": (torch.float32, (1, 1, NUM_FUTURE, 3, 3)),
    "relative_timestamps": (torch.float32, (None, NUM_FRAMES)),
    "absolute_timestamps": (torch.int64, (None, NUM_FRAMES)),
    "ego_t0_xyz": (torch.float32, (1, 1, 3)),
    "ego_t0_inv_quat": (torch.float32, (1, 1, 4)),
}
SCALAR_KEYS = {"camera_names", "camera_tmin", "t0_us", "clip_id"}


def _arrays(camera_indices, with_future=True):
    rng = np.random.default_rng(0)
    n = len(camera_indices)
    frames = torch.from_numpy(rng.integers(0, 256, (n, NUM_FRAMES, 3, H, W), dtype=np.uint8))
    offsets = np.arange(n)[:, None] * 3_000 + np.arange(NUM_FRAMES)[None] * 100_000
    timestamps = torch.from_numpy(T0_US - 300_000 + offsets).to(torch.int64)
    history = np.zeros((NUM_HISTORY, 3))
    history[:, 0] = -0.5 * np.arange(NUM_HISTORY - 1, -1, -1)
    future = np.zeros((NUM_FUTURE, 3))
    future[:, 0] = 0.5 * np.arange(1, NUM_FUTURE + 1)
    kwargs = {
        "image_frames": frames,
        "camera_indices": torch.tensor(camera_indices, dtype=torch.int64),
        "image_timestamps_us": timestamps,
        "t0_us": T0_US,
        "t0_xyz": np.array([10.0, 20.0, 0.5]),
        "t0_quat_xyzw": np.array([0.0, 0.0, np.sin(0.25), np.cos(0.25)]),
        "ego_history_xyz_local": history,
        "ego_history_rot_local": np.repeat(np.eye(3)[None], NUM_HISTORY, axis=0),
        "clip_id": "synthetic/real",
    }
    if with_future:
        kwargs["ego_future_xyz_local"] = future
        kwargs["ego_future_rot_local"] = np.repeat(np.eye(3)[None], NUM_FUTURE, axis=0)
    return kwargs


def test_contract_keys_dtypes_and_shapes():
    data = inference_sample_from_arrays(**_arrays(list(range(7))))
    assert set(data) == set(CONTRACT) | SCALAR_KEYS
    for key, (dtype, shape) in CONTRACT.items():
        value = data[key]
        assert isinstance(value, torch.Tensor), key
        assert value.dtype == dtype, key
        assert len(value.shape) == len(shape), key
        for actual, expected in zip(value.shape, shape, strict=True):
            assert expected is None or actual == expected, key
    assert data["camera_names"] == list(CAMERA_NAMES)
    assert data["camera_indices"].tolist() == list(range(7))
    assert isinstance(data["camera_tmin"], int)
    assert data["camera_tmin"] == int(data["absolute_timestamps"].min())
    assert data["t0_us"] == T0_US and data["clip_id"] == "synthetic/real"
    assert data["ego_available"].item() is True
    assert data["ego_t0_frame_idx"].item() == NUM_FRAMES - 1
    assert data["ego_t0"].item() == T0_US
    assert np.isclose(data["ego_t0_relative"].item(), (T0_US - data["camera_tmin"]) * 1e-6)
    assert np.isclose(data["relative_timestamps"].min().item(), 0.0)
    np.testing.assert_allclose(
        data["relative_timestamps"].numpy(),
        (data["absolute_timestamps"].numpy() - data["camera_tmin"]) * 1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        data["ego_history_tvals"].numpy(), np.arange(-NUM_HISTORY + 1, 1) * 0.1, atol=1e-6
    )
    np.testing.assert_allclose(
        data["ego_future_tvals"].numpy(), np.arange(1, NUM_FUTURE + 1) * 0.1, atol=1e-6
    )
    np.testing.assert_allclose(data["ego_t0_xyz"].numpy().reshape(3), [10.0, 20.0, 0.5])
    # inverse of the t0 quaternion in wxyz order: [w, -x, -y, -z]
    s, c = np.sin(0.25), np.cos(0.25)
    np.testing.assert_allclose(
        data["ego_t0_inv_quat"].numpy().reshape(4), [c, 0.0, 0.0, -s], atol=1e-6
    )
    assert data["prediction_start_offset"].item() == 0.0


def test_full_ring_selects_as_trajectory_task_input():
    data = inference_sample_from_arrays(**_arrays(list(range(7))))
    selected = select_task_input(data, "trajectory")
    assert_task_input(selected, "trajectory")
    assert selected["camera_indices"].tolist() == [0, 1, 2, 3, 5, 6]
    assert selected["image_frames"].shape[:2] == (6, NUM_FRAMES)


def test_future_omitted_when_not_given():
    data = inference_sample_from_arrays(**_arrays(list(range(7)), with_future=False))
    assert "ego_future_xyz" not in data and "ego_future_rot" not in data
    assert data["ego_future_tvals"].shape == (NUM_FUTURE,)
    assert set(data) == (set(CONTRACT) | SCALAR_KEYS) - {"ego_future_xyz", "ego_future_rot"}
    # the model input path accepts a sample without the future
    select_task_input(data, "trajectory")


def test_unsorted_cameras_are_sorted_together():
    kwargs = _arrays([6, 1, 0])
    data = inference_sample_from_arrays(**kwargs)
    assert data["camera_indices"].tolist() == [0, 1, 6]
    assert data["camera_names"] == [CAMERA_NAMES[0], CAMERA_NAMES[1], CAMERA_NAMES[6]]
    order = [2, 1, 0]
    assert torch.equal(data["image_frames"], kwargs["image_frames"][order])
    assert torch.equal(data["absolute_timestamps"], kwargs["image_timestamps_us"][order])


@pytest.mark.parametrize(
    "key, bad",
    [
        ("image_timestamps_us", torch.zeros(7, NUM_FRAMES + 1, dtype=torch.int64)),
        ("camera_indices", torch.tensor([0, 1])),
        ("camera_indices", torch.tensor([0, 1, 2, 3, 4, 5, 9])),
        ("ego_history_rot_local", np.repeat(np.eye(3)[None], NUM_HISTORY - 1, axis=0)),
        ("ego_future_xyz_local", np.zeros((NUM_FUTURE, 2))),
        ("ego_future_rot_local", None),
        ("t0_xyz", np.zeros(2)),
        ("t0_quat_xyzw", np.zeros(3)),
    ],
)
def test_shape_errors_name_the_key(key, bad):
    kwargs = _arrays(list(range(7)))
    kwargs[key] = bad
    with pytest.raises(ValueError, match=key.split("_us")[0].split("_local")[0]):
        inference_sample_from_arrays(**kwargs)

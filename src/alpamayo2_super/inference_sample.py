# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assemble the Alpamayo 2 Super inference input dict from plain arrays.

This is the one definition of the input contract consumed by
:func:`alpamayo2_super.input_profiles.select_task_input` and
:func:`alpamayo2_super.helper.prepare_model_inputs`. :func:`load_physical_aiavdataset` builds
its output through it; code that feeds frames and poses from another source should too.
"""

from typing import Any

import numpy as np
import torch

from alpamayo2_super.common.constants import CAMERA_INDICES_TO_NAMES


def _as_array(value: Any, key: str, shape: tuple[int | None, ...]) -> np.ndarray:
    """Return ``value`` as a float array of ``shape`` (``None`` = any size), or raise."""
    array = np.asarray(value.detach().cpu() if hasattr(value, "detach") else value)
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape, strict=True)
    ):
        wanted = tuple("*" if dim is None else dim for dim in shape)
        raise ValueError(f"{key} must have shape {wanted}, got {tuple(array.shape)}")
    return array


def inference_sample_from_arrays(
    *,
    image_frames: torch.Tensor,
    camera_indices: torch.Tensor,
    image_timestamps_us: torch.Tensor,
    t0_us: int,
    t0_xyz: np.ndarray,
    t0_quat_xyzw: np.ndarray,
    ego_history_xyz_local: np.ndarray,
    ego_history_rot_local: np.ndarray,
    ego_future_xyz_local: np.ndarray | None = None,
    ego_future_rot_local: np.ndarray | None = None,
    time_step: float = 0.1,
    num_future_steps: int = 64,
    clip_id: str,
) -> dict[str, Any]:
    """Build the inference input dict ``load_physical_aiavdataset`` returns, from arrays.

    Args:
        image_frames: (N_cameras, num_frames, 3, H, W) uint8 frames per camera, oldest first.
        camera_indices: (N_cameras,) camera indices (values of ``CAMERA_NAMES_TO_INDICES``), in
            the order of ``image_frames``. The output is sorted by camera index.
        image_timestamps_us: (N_cameras, num_frames) int64 frame timestamps, clip-relative
            microseconds.
        t0_us: Clip-relative timestamp (microseconds) of the prediction instant.
        t0_xyz: (3,) world position of the ego at ``t0_us``.
        t0_quat_xyzw: (4,) world orientation of the ego at ``t0_us``, scipy ``xyzw`` order.
        ego_history_xyz_local: (num_history_steps, 3) history positions in the ego frame at t0,
            last row is t0 (zeros).
        ego_history_rot_local: (num_history_steps, 3, 3) history rotations in the ego frame.
        ego_future_xyz_local: (num_future_steps, 3) future positions in the ego frame, or
            ``None`` when the future is unavailable (the keys are then omitted).
        ego_future_rot_local: (num_future_steps, 3, 3) future rotations; given with the future
            positions or not at all.
        time_step: Trajectory step in seconds (``tvals`` are multiples of it).
        num_future_steps: Length of ``ego_future_tvals`` when no future is given.
        clip_id: Identifier recorded in the sample.

    Returns:
        The dict documented on
        :func:`alpamayo2_super.load_physical_aiavdataset.load_physical_aiavdataset` (without
        ``camera_calibrations``).

    Raises:
        ValueError: On an inconsistent shape, naming the offending key.
    """
    if not isinstance(image_frames, torch.Tensor) or image_frames.ndim != 5:
        raise ValueError("image_frames must be a tensor of shape (N_cameras, num_frames, 3, H, W)")
    num_cameras, num_frames = int(image_frames.shape[0]), int(image_frames.shape[1])
    camera_indices = torch.as_tensor(camera_indices, dtype=torch.int64).reshape(-1)
    if camera_indices.numel() != num_cameras:
        raise ValueError(
            f"camera_indices has {camera_indices.numel()} entries for {num_cameras} cameras"
        )
    unknown = [int(c) for c in camera_indices.tolist() if int(c) not in CAMERA_INDICES_TO_NAMES]
    if unknown:
        raise ValueError(f"camera_indices contains unknown camera ids {unknown}")
    if len(set(camera_indices.tolist())) != num_cameras:
        raise ValueError("camera_indices must not repeat a camera")
    absolute_timestamps = torch.as_tensor(image_timestamps_us, dtype=torch.int64)
    if tuple(absolute_timestamps.shape) != (num_cameras, num_frames):
        raise ValueError(
            f"image_timestamps_us must have shape {(num_cameras, num_frames)}, "
            f"got {tuple(absolute_timestamps.shape)}"
        )

    t0_xyz_arr = _as_array(t0_xyz, "t0_xyz", (3,))
    t0_quat = _as_array(t0_quat_xyzw, "t0_quat_xyzw", (4,))
    history_xyz = _as_array(ego_history_xyz_local, "ego_history_xyz_local", (None, 3))
    num_history_steps = history_xyz.shape[0]
    history_rot = _as_array(
        ego_history_rot_local, "ego_history_rot_local", (num_history_steps, 3, 3)
    )
    if (ego_future_xyz_local is None) != (ego_future_rot_local is None):
        raise ValueError(
            "ego_future_xyz_local and ego_future_rot_local must be given together or not at all"
        )
    future_xyz = future_rot = None
    if ego_future_xyz_local is not None:
        future_xyz = _as_array(ego_future_xyz_local, "ego_future_xyz_local", (None, 3))
        num_future_steps = future_xyz.shape[0]
        future_rot = _as_array(
            ego_future_rot_local, "ego_future_rot_local", (num_future_steps, 3, 3)
        )

    # Sort by camera index for consistent ordering (idempotent on sorted input).
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    absolute_timestamps = absolute_timestamps[sort_order]

    camera_tmin_tensor = absolute_timestamps.min()
    camera_tmin = int(camera_tmin_tensor.item())
    relative_timestamps = (absolute_timestamps - camera_tmin_tensor).float() * 1e-6
    ego_t0_us = int(t0_us)

    ego_history_tvals = torch.arange(-num_history_steps + 1, 1, dtype=torch.float32) * float(
        time_step
    )
    ego_future_tvals = torch.arange(1, num_future_steps + 1, dtype=torch.float32) * float(time_step)
    t0_inv_quat_wxyz = np.array(
        [t0_quat[3], -t0_quat[0], -t0_quat[1], -t0_quat[2]],
        dtype=np.float32,
    )

    def _traj(array: np.ndarray) -> torch.Tensor:
        """(T, ...) -> float32 (1, 1, T, ...): batch and trajectory-group dims."""
        return torch.from_numpy(np.ascontiguousarray(array)).float().unsqueeze(0).unsqueeze(0)

    data: dict[str, Any] = {
        "image_frames": image_frames,  # (N_cameras, num_frames, 3, H, W)
        "camera_indices": camera_indices,  # (N_cameras,)
        "camera_names": [CAMERA_INDICES_TO_NAMES[idx] for idx in camera_indices.tolist()],
        "ego_available": torch.tensor(True),
        "ego_t0": torch.tensor([ego_t0_us], dtype=torch.int64),
        "ego_t0_relative": torch.tensor([(ego_t0_us - camera_tmin) * 1e-6], dtype=torch.float32),
        "ego_t0_frame_idx": torch.tensor([num_frames - 1], dtype=torch.int64),
        "prediction_start_offset": torch.zeros(1, dtype=torch.float32),
        "ego_history_tvals": ego_history_tvals,
        "ego_history_xyz": _traj(history_xyz),  # (1, 1, num_history_steps, 3)
        "ego_history_rot": _traj(history_rot),  # (1, 1, num_history_steps, 3, 3)
        "ego_future_tvals": ego_future_tvals,
    }
    if future_xyz is not None and future_rot is not None:
        data["ego_future_xyz"] = _traj(future_xyz)  # (1, 1, num_future_steps, 3)
        data["ego_future_rot"] = _traj(future_rot)  # (1, 1, num_future_steps, 3, 3)
    data.update(
        {
            "relative_timestamps": relative_timestamps,  # (N_cameras, num_frames)
            "absolute_timestamps": absolute_timestamps,  # (N_cameras, num_frames)
            "camera_tmin": camera_tmin,
            "ego_t0_xyz": torch.from_numpy(np.asarray(t0_xyz_arr)).float().view(1, 1, 3),
            "ego_t0_inv_quat": torch.from_numpy(t0_inv_quat_wxyz).float().view(1, 1, 4),
            "t0_us": t0_us,
            "clip_id": clip_id,
        }
    )
    return data

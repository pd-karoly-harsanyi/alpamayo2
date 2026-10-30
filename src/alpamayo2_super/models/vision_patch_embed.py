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

"""Run the Qwen-VL vision patch embedding as a GEMM instead of a conv3d."""

import logging
from collections.abc import Callable

import torch

logger = logging.getLogger(__name__)


def patch_embed_as_linear(patch_embed: torch.nn.Module) -> Callable[[], None]:
    """Replace a vision patch embedding's conv3d forward by the identical linear projection.

    The patch embedding (``Qwen3VLVisionPatchEmbed`` and relatives) is a ``nn.Conv3d`` whose
    kernel equals its stride, one patch of ``(temporal_patch_size, patch_size, patch_size)``, so
    every output element is the dot product of one flattened patch with one flattened kernel plus
    the bias: exactly ``F.linear`` with the conv weight viewed as
    ``(embed_dim, in_channels * temporal_patch_size * patch_size**2)``. Same parameters, same
    maths, same output up to float summation order. cuDNN's conv3d for this shape has been
    observed to take ~100 s per prefill on recent GPUs (torch 2.9.1 / cuDNN 9.10) where the GEMM
    takes microseconds.

    Only the instance ``forward`` is replaced; parameters and ``state_dict`` keys are unchanged,
    and the weights are read at call time, so the replacement may be applied before loading.

    Args:
        patch_embed: Module with ``proj`` (``nn.Conv3d``, bias optional), ``in_channels``,
            ``temporal_patch_size``, ``patch_size`` and ``embed_dim`` attributes.

    Returns:
        A callable that restores the original conv forward.
    """
    proj = patch_embed.proj
    patch_numel = (
        int(patch_embed.in_channels)
        * int(patch_embed.temporal_patch_size)
        * int(patch_embed.patch_size) ** 2
    )
    embed_dim = int(patch_embed.embed_dim)
    if tuple(proj.kernel_size) != tuple(proj.stride):
        raise ValueError(
            "patch_embed_as_linear requires kernel_size == stride, "
            f"got {tuple(proj.kernel_size)} and {tuple(proj.stride)}"
        )
    if proj.weight.numel() != embed_dim * patch_numel:
        raise ValueError(
            f"conv weight {tuple(proj.weight.shape)} is not one patch of {patch_numel} values "
            f"per output channel ({embed_dim})"
        )

    def forward(hidden_states: torch.Tensor) -> torch.Tensor:
        weight = proj.weight.reshape(embed_dim, patch_numel)
        hidden_states = hidden_states.reshape(-1, patch_numel).to(dtype=weight.dtype)
        return torch.nn.functional.linear(hidden_states, weight, proj.bias)

    original_forward = patch_embed.__dict__.get("forward")
    patch_embed.forward = forward

    def restore() -> None:
        if original_forward is None:
            patch_embed.__dict__.pop("forward", None)
        else:
            patch_embed.forward = original_forward

    return restore


def apply_patch_embed_as_linear(vlm: torch.nn.Module) -> bool:
    """Apply :func:`patch_embed_as_linear` to ``vlm.model.visual.patch_embed`` when present.

    Returns ``True`` when the replacement was applied; logs and returns ``False`` for a VLM
    class without that layout.
    """
    visual = getattr(getattr(vlm, "model", None), "visual", None)
    patch_embed = getattr(visual, "patch_embed", None)
    if patch_embed is None or not isinstance(getattr(patch_embed, "proj", None), torch.nn.Conv3d):
        logger.info(
            "%s has no model.visual.patch_embed conv3d; leaving the patch embedding as is",
            type(vlm).__name__,
        )
        return False
    patch_embed_as_linear(patch_embed)
    return True

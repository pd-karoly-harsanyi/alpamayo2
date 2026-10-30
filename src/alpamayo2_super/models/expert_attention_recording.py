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

"""Where the trajectory decision looked: the diffusion expert's attention over the VLM cache.

Alpamayo 2 Super's expert gets no pooled scene embedding: its trajectory tokens attend directly
into the VLM's key/value cache, which still holds every image patch token of the input frames,
the prompt and the generated chain-of-causation sentence. :class:`ExpertAttentionRecorder`
captures those weights without changing what the model computes. While recording, the expert
decoder's attention implementation is switched to a registered transformers attention function
(``AttentionInterface.register``) that calls the original implementation unchanged and, on the
side, recomputes the softmax weights from the same queries, keys and mask (for ``eager`` it takes
the weights the original call returns). Predictions are identical to runs made without recording.

What is kept per layer is attention x ``||value||``, a token's actual contribution to the output,
averaged over batch, heads, trajectory tokens and diffusion steps and normalised to sum 1: raw
attention weights pile onto attention sinks (fixed low-information tokens with near-zero values)
and read as blobs on sky and vegetation. Value weighting leaves one positional sink, the first
token of every image, which the recorder stores unmasked; readers decide how to treat it.

:meth:`ExpertAttentionRecorder.summarize` maps the contributions onto the images:

- ``attn_img`` (2, T_img) float16: contribution per image token over the images in prompt order;
  row 0 the mean over all layers, row 1 over the last quarter of the layers.
- ``attn_mass`` (L, 4) float32: per layer, the mass on [images, other prompt tokens, generated
  sentence, the expert's own tokens]; ``attn_image_mass`` (L, N_img) float32 per image.

and returns beside them the layout: which slice of ``attn_img`` is which camera, frame and grid,
and how many layers count as late.
"""

import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, AttentionMaskInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, AttentionInterface

from alpamayo2_super.common.constants import CAMERA_INDICES_TO_NAMES

logger = logging.getLogger(__name__)

RECORDING_ATTN_IMPLEMENTATION = "alpamayo2_recording"
LATE_FRACTION = 0.25  # "late layers" = the last quarter, where the decision forms

# Active recorders, keyed by id() of every attention module and config object they cover. The
# registered attention and mask functions below are global to transformers, so they find the
# recorder responsible for the module/config they are called with here.
_ACTIVE: dict[int, "ExpertAttentionRecorder"] = {}


def _attention_modules(decoder: torch.nn.Module) -> dict[int, int]:
    """``id(module) -> layer index`` for every attention module of a transformers decoder."""
    found: dict[int, int] = {}
    for module in decoder.modules():
        if type(module).__name__.endswith("Attention") and hasattr(module, "layer_idx"):
            found[id(module)] = int(module.layer_idx)
    if not found:
        raise RuntimeError("no attention modules with layer_idx found in the expert decoder")
    return found


def attention_weights(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    attention_mask: torch.Tensor | None,
    kwargs: dict[str, Any],
) -> torch.Tensor:
    """Softmax weights (b, h, Lq, Lk) as the sdpa/eager path computes them, in float32.

    Handles grouped-query attention, the ``scaling`` kwarg, bool and additive 4D masks sliced to
    the key length, and the ``is_causal`` kwarg the expert passes for non-causal attention.
    """
    groups = query.shape[1] // key.shape[1]
    if groups > 1:  # grouped-query attention: every kv head serves ``groups`` query heads
        key = key.repeat_interleave(groups, dim=1)
    scaling = kwargs.get("scaling")
    if scaling is None:
        scaling = getattr(module, "scaling", None) or query.shape[-1] ** -0.5
    weights = torch.matmul(query.float(), key.float().transpose(-1, -2)) * float(scaling)
    key_len = key.shape[-2]
    if attention_mask is not None:
        mask = attention_mask[..., :key_len]  # transformers slices a longer mask the same way
        if mask.dtype == torch.bool:
            weights = weights.masked_fill(~mask, float("-inf"))
        else:
            weights = weights + mask.float()
    else:
        is_causal = kwargs.get("is_causal")
        if is_causal is None:
            is_causal = query.shape[2] > 1 and getattr(module, "is_causal", True)
        if is_causal:
            query_len = query.shape[2]
            causal = torch.ones(query_len, key_len, dtype=torch.bool, device=weights.device).tril(
                diagonal=key_len - query_len
            )
            weights = weights.masked_fill(~causal, float("-inf"))
    return torch.softmax(weights, dim=-1)


def _recording_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Registered attention function: run the base implementation unchanged, then record."""
    recorder = _ACTIVE.get(id(module))
    if recorder is None:
        raise RuntimeError(
            f"{RECORDING_ATTN_IMPLEMENTATION} attention called for {type(module).__name__} "
            "without an active ExpertAttentionRecorder"
        )
    if recorder.base_impl == "eager":
        base = getattr(sys.modules[type(module).__module__], "eager_attention_forward", None)
        if base is None:
            raise RuntimeError(
                f"{type(module).__module__} has no eager_attention_forward to record from"
            )
    else:
        base = ALL_ATTENTION_FUNCTIONS[recorder.base_impl]
    out = base(module, query, key, value, attention_mask, *args, **kwargs)
    weights = None
    if recorder.base_impl == "eager" and isinstance(out, tuple) and len(out) > 1:
        weights = out[1]
    recorder._record(
        module=module,
        query=query,
        key=key,
        value=value,
        attention_mask=attention_mask,
        kwargs=kwargs,
        weights=weights,
    )
    return out


def _recording_mask(*args: Any, config: Any = None, **kwargs: Any) -> Any:
    """Registered mask function: build the mask the base implementation would build."""
    recorder = _ACTIVE.get(id(config))
    if recorder is None:
        raise RuntimeError(
            f"{RECORDING_ATTN_IMPLEMENTATION} mask requested for a config without an active "
            "ExpertAttentionRecorder"
        )
    base_mask = ALL_MASK_ATTENTION_FUNCTIONS._global_mapping.get(recorder.base_impl)
    if base_mask is None:  # implementations without a mask function get none (as transformers)
        return None
    return base_mask(*args, config=config, **kwargs)


AttentionInterface.register(RECORDING_ATTN_IMPLEMENTATION, _recording_attention_forward)
AttentionMaskInterface.register(RECORDING_ATTN_IMPLEMENTATION, _recording_mask)


class ExpertAttentionRecorder:
    """Accumulate the expert's attention over the cache, per layer, while the model samples.

    Args:
        model: An ``Alpamayo2Super`` (or anything with ``model.expert.expert``, the transformers
            decoder inside the expert, and optionally ``model.vlm.config`` for ``summarize``).
    """

    def __init__(self, model: Any) -> None:
        self.decoder: torch.nn.Module = model.expert.expert
        self.layers = _attention_modules(self.decoder)
        self.num_layers = max(self.layers.values()) + 1
        self.base_impl: str = "eager"
        vlm = getattr(model, "vlm", None)
        self.vlm_config = getattr(vlm, "config", None)
        self.reset()

    def reset(self) -> None:
        """Drop everything recorded so far."""
        self.sums: list[torch.Tensor | None] = [None] * self.num_layers
        self.counts: list[int] = [0] * self.num_layers
        self.query_len: int | None = None

    def _configs(self) -> list[Any]:
        """Config objects whose ``_attn_implementation`` the expert's attention reads."""
        configs: dict[int, Any] = {id(self.decoder.config): self.decoder.config}
        for module in self.decoder.modules():
            if id(module) in self.layers and getattr(module, "config", None) is not None:
                configs.setdefault(id(module.config), module.config)
        return list(configs.values())

    @contextmanager
    def recording(self) -> Iterator["ExpertAttentionRecorder"]:
        """Record every expert attention call made inside the block; restore on exit.

        Supports ``eager`` (weights taken from the original call) and every registry
        implementation (``sdpa``, ``flash_attention_2``, ...; weights recomputed on the side).
        """
        self.reset()
        configs = self._configs()
        base_impl = configs[0]._attn_implementation or "eager"
        if base_impl == RECORDING_ATTN_IMPLEMENTATION:
            raise RuntimeError("ExpertAttentionRecorder.recording() does not nest")
        if base_impl != "eager" and base_impl not in ALL_ATTENTION_FUNCTIONS:
            raise RuntimeError(f"unknown attention implementation {base_impl!r}")
        self.base_impl = base_impl
        previous = {id(config): config._attn_implementation for config in configs}
        keys = list(self.layers) + [id(config) for config in configs]
        if any(key in _ACTIVE for key in keys):
            raise RuntimeError("another ExpertAttentionRecorder is active on this expert")
        for key in keys:
            _ACTIVE[key] = self
        for config in configs:
            config._attn_implementation_internal = RECORDING_ATTN_IMPLEMENTATION
        try:
            yield self
        finally:
            for config in configs:
                config._attn_implementation_internal = previous[id(config)]
            for key in keys:
                _ACTIVE.pop(key, None)

    def _record(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kwargs: dict[str, Any],
        weights: torch.Tensor | None = None,
    ) -> None:
        layer = self.layers[id(module)]
        with torch.no_grad():
            if weights is None:
                w = attention_weights(
                    module=module,
                    query=query,
                    key=key,
                    attention_mask=attention_mask,
                    kwargs=kwargs,
                )
            else:
                w = weights[..., : key.shape[-2]]
            w = w.float()
            value_norm = value.float().norm(dim=-1)  # (b, h_kv, Lk)
            groups = w.shape[1] // value_norm.shape[1]
            if groups > 1:
                value_norm = value_norm.repeat_interleave(groups, dim=1)
            contribution = (w * value_norm[:, :, None, : w.shape[-1]]).mean(dim=(0, 1, 2))
            self.query_len = int(query.shape[2])
            previous = self.sums[layer]
            if previous is None or previous.shape != contribution.shape:
                self.sums[layer] = contribution.clone()
                self.counts[layer] = 1
            else:
                previous += contribution
                self.counts[layer] += 1

    def per_layer(self) -> np.ndarray:
        """(L, K) float32 contribution per key per layer, each row normalised to sum 1.

        ``K`` is the shortest recorded key length (cache length + expert tokens).
        """
        rows = []
        for total, count in zip(self.sums, self.counts, strict=True):
            if total is None or count == 0:
                raise RuntimeError(
                    "a layer recorded nothing; was the expert run inside recording()?"
                )
            row = (total / count).cpu().numpy()
            rows.append(row / (row.sum() or 1.0))
        k = min(len(row) for row in rows)
        return np.stack([row[:k] for row in rows]).astype(np.float32)

    def summarize(
        self,
        *,
        tokenized_data: dict[str, Any],
        camera_indices: Any,
        num_frames: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Map the recorded attention onto the input images: ``(arrays, layout)``.

        Args:
            tokenized_data: The processor output the VLM was prefilled with (``input_ids`` and
                ``image_grid_thw`` are read).
            camera_indices: Camera index per camera in prompt order.
            num_frames: Frames per camera.

        Returns:
            ``arrays`` with ``attn_img`` (2, T_img) float16, ``attn_mass`` (L, 4) float32 and
            ``attn_image_mass`` (L, N_img) float32, and ``layout`` with ``images``
            (``[{image, camera, frame, gh, gw, offset}]``, ``offset`` indexing the ``attn_img``
            axis), ``late_layers``, ``num_layers`` and ``query_len``.
        """
        if self.vlm_config is None:
            raise RuntimeError("summarize needs the VLM config (model.vlm.config)")
        layout = image_token_layout(
            vlm_config=self.vlm_config,
            tokenized_data=tokenized_data,
            camera_indices=camera_indices,
            num_frames=num_frames,
        )
        return summarize_contributions(
            per_layer=self.per_layer(),
            query_len=self.query_len or 0,
            prompt_len=int(tokenized_data["input_ids"].shape[1]),
            layout=layout,
        )


def summarize_contributions(
    per_layer: np.ndarray,
    query_len: int,
    prompt_len: int,
    layout: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reduce ``(L, K)`` per-key contributions to the image grids and the mass summary.

    Args:
        per_layer: (L, K) contributions, rows summing to 1
            (:meth:`ExpertAttentionRecorder.per_layer`).
        query_len: Number of expert tokens (the tail of the key axis).
        prompt_len: Prompt length; keys in ``[prompt_len, K - query_len)`` are the sampled sentence.
        layout: :func:`image_token_layout` output; its ``start``/``end`` are replaced by ``offset``.
    """
    a = np.asarray(per_layer, dtype=np.float32)
    num_layers, k_total = a.shape
    cache_len = k_total - int(query_len)
    is_image = np.zeros(k_total, dtype=bool)
    for item in layout:
        is_image[item["start"] : item["end"]] = True
    is_self = np.zeros(k_total, dtype=bool)
    is_self[cache_len:] = True
    is_generated = np.zeros(k_total, dtype=bool)
    is_generated[prompt_len:cache_len] = True
    is_prompt_other = ~(is_image | is_self | is_generated)
    masks = (is_image, is_prompt_other, is_generated, is_self)
    mass = np.stack([a[:, m].sum(1) for m in masks], axis=1)
    late = max(1, round(num_layers * LATE_FRACTION))
    if layout:
        image_mass = np.stack([a[:, it["start"] : it["end"]].sum(1) for it in layout], axis=1)
        grids_all = np.concatenate([a[:, it["start"] : it["end"]].mean(0) for it in layout])
        grids_late = np.concatenate([a[-late:, it["start"] : it["end"]].mean(0) for it in layout])
    else:
        image_mass = np.zeros((num_layers, 0), dtype=np.float32)
        grids_all = grids_late = np.zeros(0, dtype=np.float32)
    images = []
    offset = 0
    for it in layout:
        entry = {key: value for key, value in it.items() if key not in ("start", "end")}
        entry["offset"] = offset
        offset += it["end"] - it["start"]
        images.append(entry)
    arrays = {
        "attn_img": np.stack([grids_all, grids_late]).astype(np.float16),
        "attn_mass": mass.astype(np.float32),
        "attn_image_mass": image_mass.astype(np.float32),
    }
    layout_out = {
        "images": images,
        "late_layers": late,
        "num_layers": int(num_layers),
        "query_len": int(query_len),
    }
    return arrays, layout_out


def image_token_layout(
    vlm_config: Any,
    tokenized_data: dict[str, Any],
    camera_indices: Any,
    num_frames: int,
) -> list[dict[str, Any]]:
    """Which prompt positions are which image, in prompt order.

    The prompt holds the images camera-major (camera 0 frames 0..F-1, camera 1 ...; see
    ``prepare_model_inputs``), each a contiguous run of ``image_token_id`` placeholders whose
    count is the vision grid divided by the spatial merge.

    Returns:
        ``[{"image", "camera", "frame", "gh", "gw", "start", "end"}]`` with ``camera`` the
        camera name and ``[start, end)`` the run of positions in ``input_ids``.
    """
    image_token_id = getattr(vlm_config, "image_token_id", None)
    if image_token_id is None:
        image_token_id = vlm_config.get_text_config().image_token_id
    image_token_id = int(image_token_id)
    vision_config = getattr(vlm_config, "vision_config", None)
    merge = int(getattr(vision_config, "spatial_merge_size", 2))
    input_ids = torch.as_tensor(tokenized_data["input_ids"])[0].detach().cpu()
    positions = torch.nonzero(input_ids == image_token_id).flatten().tolist()
    grid = torch.as_tensor(tokenized_data["image_grid_thw"]).detach().cpu().tolist()
    if hasattr(camera_indices, "detach"):
        camera_indices = camera_indices.detach().cpu()
    cameras = [int(c) for c in np.asarray(camera_indices).reshape(-1)]
    layout: list[dict[str, Any]] = []
    cursor = 0
    for n, (t, h, w) in enumerate(grid):
        gh, gw = int(h) // merge, int(w) // merge
        count = int(t) * gh * gw
        run = positions[cursor : cursor + count]
        if len(run) != count or run[-1] - run[0] + 1 != count:
            raise RuntimeError(
                f"image {n}: expected {count} contiguous image tokens, got {len(run)}"
            )
        camera_pos = n // int(num_frames)
        camera = cameras[camera_pos] if camera_pos < len(cameras) else -1
        layout.append(
            {
                "image": n,
                "camera": CAMERA_INDICES_TO_NAMES.get(camera, str(camera)),
                "frame": n % int(num_frames),
                "gh": gh,
                "gw": gw,
                "start": run[0],
                "end": run[-1] + 1,
            }
        )
        cursor += count
    if cursor != len(positions):
        raise RuntimeError(f"{len(positions) - cursor} image tokens not covered by image_grid_thw")
    return layout

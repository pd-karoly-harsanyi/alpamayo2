"""Where the trajectory decision looked: the diffusion expert's attention over the camera images.

Alpamayo 2 Super's expert does not get a pooled scene embedding. Its 64 trajectory tokens attend
directly into the VLM's key/value cache, which still holds every image patch token of the 24 input
frames (6 cameras x 4 times) plus the prompt and the generated Chain-of-Causation sentence. So "what
the trajectory looked at" is a set of attention weights over image patches, and each patch maps back to
a 28 x 28 px block of a known camera frame.

``ExpertAttentionRecorder`` captures those weights WITHOUT changing how the model computes: it wraps the
attention function the expert dispatches to (``ALL_ATTENTION_FUNCTIONS[impl]`` for sdpa/flash, or the
modeling file's ``eager_attention_forward`` for eager), calls the original unchanged, and computes the
softmax weights on the side from the same queries, keys and mask. Predictions therefore stay
bit-identical to runs made without recording. The weights are averaged over batch, heads, the 64
trajectory tokens and all diffusion steps, per layer, and ``summarize`` maps them onto the images:

- ``attn_img`` (2, T) float16: attention per image token, concatenated over the N images in prompt
  order; row 0 = mean over all layers, row 1 = mean over the last quarter of layers (where the
  decision forms). ``attn_layout`` (JSON) says which slice is which (camera, frame, grid h x w).
- ``attn_mass`` (L, 4) float32: per layer, the attention mass on [images, other prompt tokens,
  generated sentence, the expert's own tokens]. ``attn_image_mass`` (L, N): per layer and image.

Usage (``run_export.py``)::

    rec = ExpertAttentionRecorder(model)
    with rec.recording():
        pred = model.sample_trajectories_from_data(...)
    npz.update(rec.summarize(model, model_inputs["tokenized_data"], data["camera_indices"], n_frames))
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch

LATE_FRACTION = 0.25  # "late layers" = last quarter


def _attention_modules(inner: torch.nn.Module) -> dict[int, int]:
    """id(module) -> layer index for every attention module of a HF decoder."""
    found = {}
    for m in inner.modules():
        if m.__class__.__name__.endswith("Attention") and hasattr(m, "layer_idx"):
            found[id(m)] = int(m.layer_idx)
    if not found:
        raise RuntimeError("no attention modules with layer_idx found in the expert")
    return found


def attention_weights(module, query: torch.Tensor, key: torch.Tensor, attention_mask, kwargs: dict) -> torch.Tensor:
    """Softmax weights (b, h, Lq, Lk) as the sdpa/eager path computes them, in float32."""
    groups = query.shape[1] // key.shape[1]
    if groups > 1:  # grouped-query attention: every kv head serves `groups` query heads
        key = key.repeat_interleave(groups, dim=1)
    scaling = kwargs.get("scaling")
    if scaling is None:
        scaling = getattr(module, "scaling", None) or query.shape[-1] ** -0.5
    w = torch.matmul(query.float(), key.float().transpose(-1, -2)) * float(scaling)
    lk = key.shape[-2]
    if attention_mask is not None:
        m = attention_mask[..., :lk]  # transformers slices a longer mask the same way
        if m.dtype == torch.bool:
            w = w.masked_fill(~m, float("-inf"))
        else:
            w = w + m.float()
    else:
        is_causal = kwargs.get("is_causal")
        if is_causal is None:
            is_causal = query.shape[2] > 1 and getattr(module, "is_causal", True)
        if is_causal:
            lq = query.shape[2]
            causal = torch.ones(lq, lk, dtype=torch.bool, device=w.device).tril(diagonal=lk - lq)
            w = w.masked_fill(~causal, float("-inf"))
    return torch.softmax(w, dim=-1)


class ExpertAttentionRecorder:
    """Accumulates the expert's attention over the cache, per layer, while the model samples."""

    def __init__(self, model):
        self.inner = model.expert.expert  # the HF decoder inside the expert
        impl = getattr(self.inner.config, "_attn_implementation", None)
        self.impl = impl or "eager"
        self.layers = _attention_modules(self.inner)
        self.num_layers = max(self.layers.values()) + 1
        self.reset()

    def reset(self) -> None:
        self.sums: list[torch.Tensor | None] = [None] * self.num_layers
        self.counts = [0] * self.num_layers
        self.query_len: int | None = None

    def _record(self, module, query, key, attention_mask, kwargs, weights: torch.Tensor | None = None) -> None:
        layer = self.layers[id(module)]
        with torch.no_grad():
            w = weights if weights is not None else attention_weights(module, query, key, attention_mask, kwargs)
            w = w[..., : key.shape[-2]] if weights is not None else w
            v = w.float().mean(dim=(0, 1, 2))  # (Lk,): mean over batch, heads, trajectory tokens
            self.query_len = int(query.shape[2])
            if self.sums[layer] is None or self.sums[layer].shape != v.shape:
                self.sums[layer] = v.clone()
                self.counts[layer] = 1
            else:
                self.sums[layer] += v
                self.counts[layer] += 1

    @contextmanager
    def recording(self):
        """Patch the attention function the expert uses; restore it on exit."""
        self.reset()
        if self.impl == "eager":
            yield from self._record_eager()
        else:
            yield from self._record_registry()

    def _record_registry(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        orig = ALL_ATTENTION_FUNCTIONS[self.impl]
        recorder = self

        def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
            out = orig(module, query, key, value, attention_mask, *args, **kwargs)
            if id(module) in recorder.layers:
                recorder._record(module, query, key, attention_mask, kwargs)
            return out

        ALL_ATTENTION_FUNCTIONS[self.impl] = wrapped
        try:
            yield self
        finally:
            ALL_ATTENTION_FUNCTIONS[self.impl] = orig

    def _record_eager(self):
        # Eager attention is referenced as a module-level name inside the modeling file, not via the
        # registry; the function returns the weights itself, so no recomputation is needed.
        mods = {sys.modules[type(m).__module__] for m in self.inner.modules() if id(m) in self.layers}
        originals = {}
        recorder = self

        def make(orig):
            def wrapped(module, query, key, value, attention_mask, *args, **kwargs):
                out = orig(module, query, key, value, attention_mask, *args, **kwargs)
                if id(module) in recorder.layers:
                    weights = out[1] if isinstance(out, tuple) and len(out) > 1 and out[1] is not None else None
                    recorder._record(module, query, key, attention_mask, kwargs, weights)
                return out

            return wrapped

        for mod in mods:
            if hasattr(mod, "eager_attention_forward"):
                originals[mod] = mod.eager_attention_forward
                mod.eager_attention_forward = make(mod.eager_attention_forward)
        if not originals:
            raise RuntimeError("expert uses eager attention but no eager_attention_forward found to wrap")
        try:
            yield self
        finally:
            for mod, fn in originals.items():
                mod.eager_attention_forward = fn

    def per_layer(self) -> np.ndarray:
        """(L, K) mean attention over the keys, per layer; K = cache length + expert tokens."""
        rows = []
        for s, c in zip(self.sums, self.counts):
            if s is None:
                raise RuntimeError("a layer recorded nothing; was the expert run inside recording()?")
            rows.append((s / c).cpu().numpy())
        k = min(len(r) for r in rows)
        return np.stack([r[:k] for r in rows]).astype(np.float32)

    def summarize(self, model, tokenized_data: dict, camera_indices, n_frames: int) -> dict[str, Any]:
        """Map the recorded attention onto the input images. Returns arrays to store in the record's npz."""
        layout = image_token_layout(model, tokenized_data, camera_indices, n_frames)
        a = self.per_layer()  # (L, K)
        n_layers, k_total = a.shape
        q = self.query_len or 0
        prompt_len = int(tokenized_data["input_ids"].shape[1])
        cache_len = k_total - q
        is_image = np.zeros(k_total, dtype=bool)
        for item in layout:
            is_image[item["start"] : item["end"]] = True
        is_self = np.zeros(k_total, dtype=bool)
        is_self[cache_len:] = True
        is_generated = np.zeros(k_total, dtype=bool)
        is_generated[prompt_len:cache_len] = True
        is_prompt_other = ~(is_image | is_self | is_generated)
        mass = np.stack([a[:, m].sum(1) for m in (is_image, is_prompt_other, is_generated, is_self)], axis=1)
        image_mass = np.stack([a[:, it["start"] : it["end"]].sum(1) for it in layout], axis=1) if layout else np.zeros((n_layers, 0))
        late = max(1, int(round(n_layers * LATE_FRACTION)))
        grids_all = np.concatenate([a[:, it["start"] : it["end"]].mean(0) for it in layout]) if layout else np.zeros(0)
        grids_late = np.concatenate([a[-late:, it["start"] : it["end"]].mean(0) for it in layout]) if layout else np.zeros(0)
        offset = 0
        for it in layout:
            it["offset"] = offset
            offset += it["end"] - it["start"]
            del it["start"], it["end"]
        return {
            "attn_img": np.stack([grids_all, grids_late]).astype(np.float16),
            "attn_layout": np.array(json.dumps({"images": layout, "late_layers": late, "num_layers": n_layers, "query_len": q})),
            "attn_mass": mass.astype(np.float32),
            "attn_image_mass": image_mass.astype(np.float32),
        }


def image_token_layout(model, tokenized_data: dict, camera_indices, n_frames: int) -> list[dict]:
    """Which sequence positions are which image: [{image, camera, frame, gh, gw, start, end}] in prompt order.

    The prompt holds the images camera-major (``image_frames.flatten(0, 1)``: camera 0 frames 0..3,
    camera 1 ...), each expanded to a contiguous run of ``image_token_id`` placeholders whose count is
    the vision grid divided by the spatial merge (2 x 2 patches of 14 px -> one 28 px token)."""
    from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES

    index_to_name = {v: k for k, v in CAMERA_NAMES_TO_INDICES.items()}
    vcfg = model.vlm.config
    image_token_id = int(getattr(vcfg, "image_token_id", None) or vcfg.get_text_config().image_token_id)
    merge = int(getattr(vcfg.vision_config, "spatial_merge_size", 2))
    input_ids = tokenized_data["input_ids"][0].detach().cpu()
    positions = torch.nonzero(input_ids == image_token_id).flatten().tolist()
    grid = tokenized_data["image_grid_thw"].detach().cpu().tolist()
    cams = [int(c) for c in np.asarray(camera_indices.detach().cpu() if hasattr(camera_indices, "detach") else camera_indices).reshape(-1)]
    layout, cursor = [], 0
    for n, (t, h, w) in enumerate(grid):
        gh, gw = h // merge, w // merge
        count = int(t) * gh * gw
        run = positions[cursor : cursor + count]
        if len(run) != count or run[-1] - run[0] + 1 != count:
            raise RuntimeError(f"image {n}: expected {count} contiguous image tokens, got {len(run)}")
        cam = cams[n // n_frames] if n // n_frames < len(cams) else -1
        layout.append({"image": n, "camera": index_to_name.get(cam, str(cam)), "frame": n % n_frames,
                       "gh": gh, "gw": gw, "start": run[0], "end": run[-1] + 1})
        cursor += count
    if cursor != len(positions):
        raise RuntimeError(f"{len(positions) - cursor} image tokens not covered by image_grid_thw")
    return layout


def heatmap_for(rec: dict, camera: str, frame: int | None = None, late: bool = True) -> np.ndarray | None:
    """(gh, gw) attention grid of one camera frame from a loaded record, or None if not stored."""
    if rec.get("attn_img") is None or rec.get("attn_layout") is None:
        return None
    layout = json.loads(str(rec["attn_layout"]))
    images = layout["images"]
    if frame is None:
        frame = max(it["frame"] for it in images)  # the frame at t0
    for it in images:
        if it["camera"] == camera and it["frame"] == frame:
            flat = np.asarray(rec["attn_img"], dtype=np.float32)[1 if late else 0]
            return flat[it["offset"] : it["offset"] + it["gh"] * it["gw"]].reshape(it["gh"], it["gw"])
    return None


def camera_shares(rec: dict, late: bool = True) -> dict[str, float] | None:
    """Share of the image attention per camera (all four frames), from a loaded record."""
    if rec.get("attn_image_mass") is None or rec.get("attn_layout") is None:
        return None
    layout = json.loads(str(rec["attn_layout"]))
    m = np.asarray(rec["attn_image_mass"], dtype=np.float32)
    rows = m[-layout["late_layers"] :] if late else m
    per_image = rows.mean(0)
    shares: dict[str, float] = {}
    for it in layout["images"]:
        shares[it["camera"]] = shares.get(it["camera"], 0.0) + float(per_image[it["image"]])
    total = sum(shares.values()) or 1.0
    return {k: v / total for k, v in shares.items()}


def text_share(rec: dict, late: bool = True) -> dict[str, float] | None:
    """Attention mass on [images, prompt, sentence, self], averaged over the chosen layers."""
    if rec.get("attn_mass") is None or rec.get("attn_layout") is None:
        return None
    layout = json.loads(str(rec["attn_layout"]))
    m = np.asarray(rec["attn_mass"], dtype=np.float32)
    rows = m[-layout["late_layers"] :] if late else m
    v = rows.mean(0)
    return {"images": float(v[0]), "prompt": float(v[1]), "sentence": float(v[2]), "self": float(v[3])}

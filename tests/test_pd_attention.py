"""The attention recorder reproduces eager attention weights exactly and leaves outputs untouched.

Runs on CPU with a tiny randomly initialised Qwen2 decoder (same transformers version as the model
env), in the way the expert is driven: a prefilled cache, new tokens appended with a 4D additive mask.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
transformers = pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2Model  # noqa: E402

from pd_alpamayo_eval.attention import ExpertAttentionRecorder, heatmap_for, image_token_layout  # noqa: E402

N_PREFIX, N_NEW = 23, 8


def _tiny(impl: str) -> Qwen2Model:
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=100, max_position_embeddings=256, attn_implementation=impl)
    return Qwen2Model(cfg).eval()


def _drive(model: Qwen2Model, record: ExpertAttentionRecorder | None, output_attentions=False):
    torch.manual_seed(1)
    prefix = torch.randn(1, N_PREFIX, 32)
    new = torch.randn(1, N_NEW, 32)
    with torch.no_grad():
        cache = model(inputs_embeds=prefix, use_cache=True).past_key_values
        # 4D additive mask like the expert's: the last 5 cache positions masked for the new tokens
        mask = torch.zeros(1, 1, N_NEW, N_PREFIX + N_NEW)
        mask[..., N_PREFIX - 5 : N_PREFIX] = torch.finfo(torch.float32).min
        pos = torch.arange(N_PREFIX, N_PREFIX + N_NEW)[None]
        kwargs = dict(inputs_embeds=new, past_key_values=cache, attention_mask=mask, position_ids=pos, use_cache=True,
                      output_attentions=output_attentions)
        if record is not None:
            with record.recording():
                out = model(**kwargs)
        else:
            out = model(**kwargs)
    return out


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_recorder_matches_eager_and_keeps_outputs(impl):
    model = _tiny(impl)
    fake = SimpleNamespace(expert=SimpleNamespace(expert=model))
    rec = ExpertAttentionRecorder(fake)
    assert rec.num_layers == 3
    plain = _drive(model, None)
    recorded = _drive(model, rec)
    assert torch.equal(plain.last_hidden_state, recorded.last_hidden_state), "recording changed the output"
    # ground truth: the same weights, eager, output_attentions
    ref_model = _tiny("eager")
    ref_model.load_state_dict(model.state_dict())
    ref = _drive(ref_model, None, output_attentions=True)
    per_layer = rec.per_layer()
    assert per_layer.shape == (3, N_PREFIX + N_NEW)
    for layer, att in enumerate(ref.attentions):
        expected = att.float().mean(dim=(0, 1, 2)).numpy()
        np.testing.assert_allclose(per_layer[layer], expected, atol=1e-5)
    assert np.allclose(per_layer.sum(1), 1.0, atol=1e-4)
    assert np.allclose(per_layer[:, N_PREFIX - 5 : N_PREFIX], 0.0)  # masked positions get nothing


def test_recorder_averages_over_steps():
    model = _tiny("sdpa")
    rec = ExpertAttentionRecorder(SimpleNamespace(expert=SimpleNamespace(expert=model)))
    torch.manual_seed(2)
    prefix = torch.randn(1, N_PREFIX, 32)
    with torch.no_grad(), rec.recording():
        cache = model(inputs_embeds=prefix, use_cache=True).past_key_values
        rec.reset()  # the prefill is not what we want; the expert's steps are
        for _ in range(4):
            cache.crop(N_PREFIX)
            model(inputs_embeds=torch.randn(1, N_NEW, 32), past_key_values=cache, use_cache=True,
                  position_ids=torch.arange(N_PREFIX, N_PREFIX + N_NEW)[None])
    assert rec.counts == [4, 4, 4]
    assert rec.per_layer().shape == (3, N_PREFIX + N_NEW)


def test_image_token_layout_and_heatmap():
    image_token_id, merge = 7, 2
    grid = torch.tensor([[1, 4, 6], [1, 4, 6], [1, 2, 4]])  # 3 images: 6, 6, 2 tokens
    ids = [1, 2] + [7] * 6 + [3] + [7] * 6 + [4, 5] + [7] * 2 + [6]
    tok = {"input_ids": torch.tensor([ids]), "image_grid_thw": grid}
    vcfg = SimpleNamespace(image_token_id=image_token_id, vision_config=SimpleNamespace(spatial_merge_size=merge))
    model = SimpleNamespace(vlm=SimpleNamespace(config=vcfg))
    import alpamayo2_super.common.constants as c  # noqa: F401  (camera name map)

    cams = torch.tensor([0, 1])  # 2 cameras x ... n_frames=2 -> image n: camera n//2, frame n%2
    layout = image_token_layout(model, tok, cams, n_frames=2)
    assert [(it["start"], it["end"], it["gh"], it["gw"]) for it in layout] == [(2, 8, 2, 3), (9, 15, 2, 3), (17, 19, 1, 2)]
    assert [it["frame"] for it in layout] == [0, 1, 0]
    # a fake record with the first image's late-layer grid = 0..5
    flat = np.arange(14, dtype=np.float32)
    rec = {"attn_img": np.stack([flat, flat]), "attn_layout": json.dumps({"images": [
        {"image": 0, "camera": layout[0]["camera"], "frame": 0, "gh": 2, "gw": 3, "offset": 0},
        {"image": 1, "camera": layout[1]["camera"], "frame": 1, "gh": 2, "gw": 3, "offset": 6},
        {"image": 2, "camera": layout[2]["camera"], "frame": 0, "gh": 1, "gw": 2, "offset": 12}], "late_layers": 1, "num_layers": 3, "query_len": 8})}
    hm = heatmap_for(rec, layout[1]["camera"], frame=1)
    assert hm.shape == (2, 3) and hm[0, 0] == 6 and hm[1, 2] == 11
    assert heatmap_for(rec, "nope") is None

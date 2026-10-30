# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The expert attention recorder reproduces eager weights exactly and leaves outputs untouched.

Runs on CPU with a tiny randomly initialised Qwen2 decoder driven the way the expert is driven: a
prefilled cache, new tokens appended with a 4D additive mask, non-causal.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alpamayo2_super.models import expert_attention_recording as ear
from alpamayo2_super.models.expert_attention_recording import (
    RECORDING_ATTN_IMPLEMENTATION,
    ExpertAttentionRecorder,
    image_token_layout,
    summarize_contributions,
)

transformers = pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2Model  # noqa: E402

N_PREFIX, N_NEW, HIDDEN, LAYERS = 23, 8, 32, 3


def _tiny(impl: str) -> Qwen2Model:
    torch.manual_seed(0)
    cfg = Qwen2Config(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=100,
        max_position_embeddings=256,
        attn_implementation=impl,
    )
    return Qwen2Model(cfg).eval()


def _fake_model(decoder: Qwen2Model) -> SimpleNamespace:
    return SimpleNamespace(expert=SimpleNamespace(expert=decoder))


def _drive(model: Qwen2Model, recorder: ExpertAttentionRecorder | None, output_attentions=False):
    torch.manual_seed(1)
    prefix = torch.randn(1, N_PREFIX, HIDDEN)
    new = torch.randn(1, N_NEW, HIDDEN)
    with torch.no_grad():
        cache = model(inputs_embeds=prefix, use_cache=True).past_key_values
        # 4D additive mask like the expert's: the last 5 cache positions masked for the new tokens
        mask = torch.zeros(1, 1, N_NEW, N_PREFIX + N_NEW)
        mask[..., N_PREFIX - 5 : N_PREFIX] = torch.finfo(torch.float32).min
        pos = torch.arange(N_PREFIX, N_PREFIX + N_NEW)[None]
        kwargs = {
            "inputs_embeds": new,
            "past_key_values": cache,
            "attention_mask": mask,
            "position_ids": pos,
            "use_cache": True,
            "is_causal": False,
            "output_attentions": output_attentions,
        }
        if recorder is not None:
            with recorder.recording():
                out = model(**kwargs)
        else:
            out = model(**kwargs)
    return out


def _value_norms(model: Qwen2Model, out) -> list[torch.Tensor]:
    """(b, h, Lk) value norms per layer from the cache, expanded to the query heads."""
    norms = []
    groups = model.config.num_attention_heads // model.config.num_key_value_heads
    for layer in out.past_key_values.layers:
        norm = layer.values.float().norm(dim=-1)
        norms.append(norm.repeat_interleave(groups, dim=1))
    return norms


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_recorded_weights_match_eager_and_outputs_are_identical(impl):
    model = _tiny(impl)
    recorder = ExpertAttentionRecorder(_fake_model(model))
    assert recorder.num_layers == LAYERS
    plain = _drive(model, None)
    recorded = _drive(model, recorder)
    assert torch.equal(plain.last_hidden_state, recorded.last_hidden_state), (
        "recording changed the output"
    )
    assert recorder.base_impl == impl
    assert model.config._attn_implementation == impl, "implementation not restored"
    assert not ear._ACTIVE

    # ground truth: eager weights of the same model via output_attentions, value weighted by hand
    ref_model = _tiny("eager")
    ref_model.load_state_dict(model.state_dict())
    ref = _drive(ref_model, None, output_attentions=True)
    value_norms = _value_norms(ref_model, ref)
    per_layer = recorder.per_layer()
    assert per_layer.shape == (LAYERS, N_PREFIX + N_NEW)
    assert recorder.counts == [1] * LAYERS
    assert recorder.query_len == N_NEW
    for layer, att in enumerate(ref.attentions):
        expected = (att.float() * value_norms[layer][:, :, None, :]).mean(dim=(0, 1, 2)).numpy()
        expected = expected / expected.sum()
        np.testing.assert_allclose(per_layer[layer], expected, atol=1e-5)
    assert np.allclose(per_layer.sum(1), 1.0, atol=1e-4)
    assert np.allclose(per_layer[:, N_PREFIX - 5 : N_PREFIX], 0.0)  # masked positions get nothing


def test_recording_without_a_mask_keeps_outputs_identical():
    """Mask creation goes through the registered mask function; the result must not change."""
    model = _tiny("sdpa")
    recorder = ExpertAttentionRecorder(_fake_model(model))
    torch.manual_seed(3)
    x = torch.randn(1, N_PREFIX, HIDDEN)
    with torch.no_grad():
        plain = model(inputs_embeds=x).last_hidden_state
        with recorder.recording():
            recorded = model(inputs_embeds=x).last_hidden_state
    assert torch.equal(plain, recorded)
    per_layer = recorder.per_layer()
    assert per_layer.shape == (LAYERS, N_PREFIX)
    # causal: the first query attends only to the first key, so key 0 collects mass from everyone
    assert per_layer[:, 0].min() > 0


def test_value_weighting_on_hand_built_tensors():
    class _Module:
        scaling = 1.0

    model = _tiny("sdpa")
    recorder = ExpertAttentionRecorder(_fake_model(model))
    module = next(m for m in model.modules() if id(m) in recorder.layers)
    # one batch, one head, two queries, three keys; weights given explicitly
    weights = torch.tensor([[[[0.5, 0.25, 0.25], [0.0, 1.0, 0.0]]]])
    value = torch.zeros(1, 1, 3, 4)
    value[0, 0, 0, 0] = 2.0  # ||v0|| = 2
    value[0, 0, 1, 1] = 1.0  # ||v1|| = 1
    value[0, 0, 2, :] = 0.0  # ||v2|| = 0
    query = torch.zeros(1, 1, 2, 4)
    key = torch.zeros(1, 1, 3, 4)
    recorder._record(module, query, key, value, None, {}, weights=weights)
    layer = recorder.layers[id(module)]
    expected = torch.tensor([(0.5 * 2 + 0.0 * 2) / 2, (0.25 * 1 + 1.0 * 1) / 2, 0.0])
    assert torch.allclose(recorder.sums[layer], expected)
    assert recorder.counts[layer] == 1


def test_recorder_averages_over_steps_and_resets_on_a_key_length_change():
    model = _tiny("sdpa")
    recorder = ExpertAttentionRecorder(_fake_model(model))
    torch.manual_seed(2)
    prefix = torch.randn(1, N_PREFIX, HIDDEN)
    with torch.no_grad():
        cache = model(inputs_embeds=prefix, use_cache=True).past_key_values
        pos = torch.arange(N_PREFIX, N_PREFIX + N_NEW)[None]
        steps = []
        with recorder.recording():
            for _ in range(2):
                cache.crop(N_PREFIX)
                x = torch.randn(1, N_NEW, HIDDEN)
                model(inputs_embeds=x, past_key_values=cache, use_cache=True, position_ids=pos)
                steps.append([s.clone() for s in recorder.sums])
    assert recorder.counts == [2, 2, 2]
    per_layer = recorder.per_layer()
    assert per_layer.shape == (LAYERS, N_PREFIX + N_NEW)
    first_step = steps[0][0]
    mean_row = (recorder.sums[0] / 2).numpy()
    assert not np.allclose(first_step.numpy(), mean_row)  # second step changed the mean
    np.testing.assert_allclose(per_layer[0], mean_row / mean_row.sum(), atol=1e-6)

    # a different key length resets the layer instead of mixing lengths
    module = next(m for m in model.modules() if recorder.layers.get(id(m)) == 0)
    recorder._record(
        module,
        torch.zeros(1, 4, 2, 8),
        torch.zeros(1, 2, 5, 8),
        torch.ones(1, 2, 5, 8),
        None,
        {"is_causal": False},
    )
    assert recorder.counts[0] == 1 and recorder.sums[0].shape == (5,)


def test_recording_does_not_nest_and_restores_on_error():
    model = _tiny("sdpa")
    recorder = ExpertAttentionRecorder(_fake_model(model))
    with pytest.raises(RuntimeError, match="nest"), recorder.recording(), recorder.recording():
        pass
    assert model.config._attn_implementation == "sdpa"
    with pytest.raises(ValueError), recorder.recording():
        assert model.config._attn_implementation == RECORDING_ATTN_IMPLEMENTATION
        raise ValueError("boom")
    assert model.config._attn_implementation == "sdpa"
    assert not ear._ACTIVE


def _vlm_config(image_token_id: int = 7, merge: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        image_token_id=image_token_id, vision_config=SimpleNamespace(spatial_merge_size=merge)
    )


def test_image_token_layout():
    grid = torch.tensor([[1, 4, 4], [1, 4, 4]])  # merge 2 -> 4 tokens each
    ids = [1, 2] + [7] * 4 + [3] + [7] * 4 + [5]
    tokenized = {"input_ids": torch.tensor([ids]), "image_grid_thw": grid}
    layout = image_token_layout(_vlm_config(), tokenized, torch.tensor([1]), num_frames=2)
    assert [(it["start"], it["end"], it["gh"], it["gw"]) for it in layout] == [
        (2, 6, 2, 2),
        (7, 11, 2, 2),
    ]
    assert [it["image"] for it in layout] == [0, 1]
    assert [it["frame"] for it in layout] == [0, 1]
    assert [it["camera"] for it in layout] == ["camera_front_wide_120fov"] * 2

    # camera-major image order: image n is camera n // num_frames
    two_cams = image_token_layout(_vlm_config(), tokenized, torch.tensor([0, 6]), num_frames=1)
    assert [it["camera"] for it in two_cams] == [
        "camera_cross_left_120fov",
        "camera_front_tele_30fov",
    ]
    assert [it["frame"] for it in two_cams] == [0, 0]


def test_image_token_layout_rejects_non_contiguous_and_uncovered_runs():
    grid = torch.tensor([[1, 4, 4]])
    broken = {"input_ids": torch.tensor([[7, 7, 1, 7, 7]]), "image_grid_thw": grid}
    with pytest.raises(RuntimeError, match="contiguous"):
        image_token_layout(_vlm_config(), broken, torch.tensor([0]), num_frames=1)
    extra = {"input_ids": torch.tensor([[7, 7, 7, 7, 7]]), "image_grid_thw": grid}
    with pytest.raises(RuntimeError, match="not covered"):
        image_token_layout(_vlm_config(), extra, torch.tensor([0]), num_frames=1)


def test_summary_masks_and_image_concatenation():
    num_layers, prompt_len, generated, query_len = 8, 12, 3, 4
    k_total = prompt_len + generated + query_len
    layout = [
        {"image": 0, "camera": "a", "frame": 0, "gh": 1, "gw": 3, "start": 2, "end": 5},
        {"image": 1, "camera": "a", "frame": 1, "gh": 1, "gw": 2, "start": 6, "end": 8},
    ]
    rng = np.random.default_rng(0)
    per_layer = rng.random((num_layers, k_total)).astype(np.float32)
    per_layer /= per_layer.sum(1, keepdims=True)
    arrays, out_layout = summarize_contributions(per_layer, query_len, prompt_len, layout)

    assert arrays["attn_img"].shape == (2, 5) and arrays["attn_img"].dtype == np.float16
    assert arrays["attn_mass"].shape == (num_layers, 4) and arrays["attn_mass"].dtype == np.float32
    assert arrays["attn_image_mass"].shape == (num_layers, 2)
    assert arrays["attn_image_mass"].dtype == np.float32
    np.testing.assert_allclose(arrays["attn_mass"].sum(1), 1.0, atol=1e-5)
    np.testing.assert_allclose(
        arrays["attn_mass"][:, 0], per_layer[:, 2:5].sum(1) + per_layer[:, 6:8].sum(1), atol=1e-6
    )
    np.testing.assert_allclose(
        arrays["attn_mass"][:, 2], per_layer[:, prompt_len : prompt_len + generated].sum(1)
    )
    np.testing.assert_allclose(arrays["attn_mass"][:, 3], per_layer[:, -query_len:].sum(1))
    expected_all = np.concatenate([per_layer[:, 2:5].mean(0), per_layer[:, 6:8].mean(0)])
    late = out_layout["late_layers"]
    assert late == 2
    expected_late = np.concatenate([per_layer[-late:, 2:5].mean(0), per_layer[-late:, 6:8].mean(0)])
    np.testing.assert_allclose(arrays["attn_img"][0], expected_all, rtol=2e-3)
    np.testing.assert_allclose(arrays["attn_img"][1], expected_late, rtol=2e-3)
    assert out_layout["num_layers"] == num_layers and out_layout["query_len"] == query_len
    assert out_layout["images"] == [
        {"image": 0, "camera": "a", "frame": 0, "gh": 1, "gw": 3, "offset": 0},
        {"image": 1, "camera": "a", "frame": 1, "gh": 1, "gw": 2, "offset": 3},
    ]
    assert "start" in layout[0]  # the caller's layout is left untouched


def test_summarize_end_to_end_on_the_tiny_model():
    model = _tiny("sdpa")
    fake = _fake_model(model)
    fake.vlm = SimpleNamespace(config=_vlm_config())
    recorder = ExpertAttentionRecorder(fake)
    _drive(model, recorder)
    # prompt of N_PREFIX + 1 tokens with two 4-token images; the cache holds the prompt minus the
    # last token plus nothing generated (prefill is input_ids[:, :-1]) -> cache_len = N_PREFIX
    ids = [1] * 3 + [7] * 4 + [2] * 5 + [7] * 4 + [3] * (N_PREFIX + 1 - 16)
    tokenized = {"input_ids": torch.tensor([ids]), "image_grid_thw": torch.tensor([[1, 4, 4]] * 2)}
    arrays, layout = recorder.summarize(
        tokenized_data=tokenized, camera_indices=torch.tensor([1]), num_frames=2
    )
    assert arrays["attn_img"].shape == (2, 8)
    assert arrays["attn_mass"].shape == (LAYERS, 4)
    assert layout["query_len"] == N_NEW and layout["num_layers"] == LAYERS
    assert [it["offset"] for it in layout["images"]] == [0, 4]

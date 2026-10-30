# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test of the ``diffusion_seed`` and ``record_attention`` wiring in trajectory sampling.

The VLM is stubbed (prefill + generation replaced by a fixed sentence and a cache built by a tiny
Qwen3-VL text decoder standing in for the expert); everything after the sentence, from the expert
mask to the diffusion sampler and ``extra``, runs the real code.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GenerationConfig

pytest.importorskip("hydra")
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel  # noqa: E402

from alpamayo2_super.diffusion.flow_matching import FlowMatching  # noqa: E402
from alpamayo2_super.models import alpamayo2_super as model_module  # noqa: E402
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402
from alpamayo2_super.models.expert_attention_recording import (  # noqa: E402
    RECORDING_ATTN_IMPLEMENTATION,
)

HIDDEN, N_TOKENS, ACTION_DIM, EOS, IMAGE_TOKEN, PAD = 32, 4, 2, 99, 7, 0
PROMPT = [1, 2] + [IMAGE_TOKEN] * 4 + [3] + [IMAGE_TOKEN] * 4 + [4, 5]
GENERATED = [11, 12, EOS]


class _ActionSpace:
    def get_action_space_dims(self) -> tuple[int, int]:
        return (N_TOKENS, ACTION_DIM)

    def action_to_traj(self, action, hist_xyz, hist_rot):
        """``hist_xyz`` is the last history waypoint per sample, (b, T, 3) here (T = 16)."""
        anchor = hist_xyz.reshape(hist_xyz.shape[0], -1, 3)[:, -1]
        xyz = torch.cat([action, torch.zeros_like(action[..., :1])], dim=-1) + anchor[:, None]
        rot = hist_rot.reshape(hist_rot.shape[0], -1, 3, 3)[:, -1][:, None]
        return xyz, rot.expand(-1, N_TOKENS, -1, -1)


class _ActionInProj(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(ACTION_DIM + 1, HIDDEN)

    def forward(self, x, t):
        t = t.reshape(x.shape[0], 1, 1).expand(-1, x.shape[1], 1)
        return self.proj(torch.cat([x, t], dim=-1))


class _Stub:
    """Just enough of ``Alpamayo2Super`` for ``sample_trajectories_from_data``."""

    def __init__(self, attn_implementation: str) -> None:
        torch.manual_seed(0)
        config = Qwen3VLTextConfig(
            hidden_size=HIDDEN,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=128,
            max_position_embeddings=512,
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
            attn_implementation=attn_implementation,
        )
        decoder = Qwen3VLTextModel(config).eval()
        self.decoder = decoder
        self.config = SimpleNamespace(
            enable_expert=True,
            traj_ids={"history_id0": 100, "future_id0": 110, "future_start": EOS},
            traj_vocab_size=10,
            tokens_per_future_traj=8,
        )
        vlm_config = SimpleNamespace(
            image_token_id=IMAGE_TOKEN, vision_config=SimpleNamespace(spatial_merge_size=2)
        )
        self.vlm = SimpleNamespace(generation_config=GenerationConfig(), config=vlm_config)
        self.tokenizer = SimpleNamespace(pad_token_id=PAD)
        self.history_traj_tokenizer = self.future_traj_tokenizer = None
        self.expert = SimpleNamespace(
            expert=decoder,
            config=SimpleNamespace(expert_non_causal_attention=True),
            action_space=_ActionSpace(),
            diffusion=FlowMatching(x_dims=(N_TOKENS, ACTION_DIM), num_inference_steps=3),
            action_in_proj=_ActionInProj(),
            action_out_proj=torch.nn.Linear(HIDDEN, ACTION_DIM),
        )
        self._diffusion_expert_cuda_graph = None
        self._expert_attention_recorder = None
        self.sample_calls = 0

    def _generate_with_shared_prefill(
        self, tokenized_data, generation_config, n_samples_total, **kw
    ):
        del generation_config, kw
        input_ids = tokenized_data["input_ids"]
        sequences = torch.cat(
            [input_ids, torch.tensor([GENERATED], dtype=input_ids.dtype)], dim=1
        ).repeat(n_samples_total, 1)
        torch.manual_seed(1)
        embeds = torch.randn(n_samples_total, sequences.shape[1], HIDDEN)
        with torch.no_grad():
            cache = self.decoder(inputs_embeds=embeds, use_cache=True).past_key_values
        return SimpleNamespace(
            sequences=sequences,
            past_key_values=cache,
            rope_deltas=torch.zeros(n_samples_total, 1, dtype=torch.long),
        )


def _data() -> dict:
    return {
        "tokenized_data": {
            "input_ids": torch.tensor([PROMPT]),
            "attention_mask": torch.ones(1, len(PROMPT), dtype=torch.long),
            "image_grid_thw": torch.tensor([[1, 4, 4], [1, 4, 4]]),
        },
        "ego_history_xyz": torch.zeros(1, 1, 16, 3),
        "ego_history_rot": torch.eye(3).expand(1, 1, 16, 3, 3).clone(),
        "camera_indices": torch.tensor([1]),
        "num_frames": 2,
    }


@pytest.fixture(autouse=True)
def _stub_text_functions(monkeypatch):
    monkeypatch.setattr(model_module, "fuse_traj_tokens", lambda *args: args[2])
    monkeypatch.setattr(
        model_module,
        "extract_text_tokens",
        lambda tokenizer, sequences, task: {"cot": ["sentence"] * sequences.shape[0]},
    )


def _sample(stub, **kwargs):
    kwargs.setdefault("diffusion_kwargs", {"inference_step": 3})
    return Alpamayo2Super.sample_trajectories_from_data(
        stub, data=_data(), num_traj_samples=2, **kwargs
    )


def test_diffusion_seed_fixes_the_trajectory_regardless_of_the_global_stream():
    stub = _Stub("sdpa")
    torch.manual_seed(0)
    torch.randn(100)
    a, _, _ = _sample(stub, diffusion_seed=7)
    torch.manual_seed(999)
    b, _, _ = _sample(stub, diffusion_seed=7)
    c, _, _ = _sample(stub, diffusion_seed=8)
    assert a.shape == (1, 1, 2, N_TOKENS, 3)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)

    # the seeded generator draws what a global reseed right before the noise draw would draw
    # (the monkeypatch pdncc used before ``diffusion_seed`` existed)
    original = stub.expert.diffusion.sample

    def reseed_then_sample(*args, **kwargs):
        torch.manual_seed(7)
        return original(*args, **kwargs)

    stub.expert.diffusion.sample = reseed_then_sample
    d, _, _ = _sample(stub)
    stub.expert.diffusion.sample = original
    assert torch.equal(a, d)
    # and the diffusion phase leaves the global stream where the VLM left it when seeded
    states = {}

    def snapshot_then_sample(*args, **kwargs):
        states["at_noise_draw"] = torch.get_rng_state()
        return original(*args, **kwargs)

    stub.expert.diffusion.sample = snapshot_then_sample
    _sample(stub, diffusion_seed=7)
    assert torch.equal(torch.get_rng_state(), states["at_noise_draw"])
    _sample(stub)
    assert not torch.equal(torch.get_rng_state(), states["at_noise_draw"])
    stub.expert.diffusion.sample = original


def test_diffusion_seed_rejects_an_explicit_generator():
    stub = _Stub("sdpa")
    with pytest.raises(ValueError, match="generator"):
        _sample(stub, diffusion_seed=1, diffusion_kwargs={"generator": torch.Generator()})


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_record_attention_keeps_predictions_identical_and_returns_the_summary(impl):
    stub = _Stub(impl)
    plain = _sample(stub, diffusion_seed=5, return_extra=True)
    recorded = _sample(stub, diffusion_seed=5, return_extra=True, record_attention=True)
    assert torch.equal(plain[0], recorded[0]) and torch.equal(plain[1], recorded[1])
    assert "attention" not in plain[3]
    assert stub.decoder.config._attn_implementation == impl

    extra = recorded[3]
    assert extra["cot"].shape == (1, 1, 2)
    attention = extra["attention"]
    arrays, layout = attention["arrays"], attention["layout"]
    assert set(arrays) == {"attn_img", "attn_mass", "attn_image_mass"}
    assert arrays["attn_img"].shape == (2, 8) and arrays["attn_img"].dtype == np.float16
    assert arrays["attn_mass"].shape == (2, 4) and arrays["attn_mass"].dtype == np.float32
    assert arrays["attn_image_mass"].shape == (2, 2)
    np.testing.assert_allclose(arrays["attn_mass"].sum(1), 1.0, atol=1e-5)
    assert layout["num_layers"] == 2 and layout["query_len"] == N_TOKENS
    assert layout["late_layers"] == 1
    assert [(it["camera"], it["frame"], it["offset"]) for it in layout["images"]] == [
        ("camera_front_wide_120fov", 0, 0),
        ("camera_front_wide_120fov", 1, 4),
    ]
    # the cache holds prompt + sentence; the masked expert steps all landed in the recorder
    recorder = stub._expert_attention_recorder
    assert recorder.counts == [3, 3]
    assert recorder.per_layer().shape == (2, len(PROMPT) + len(GENERATED) + N_TOKENS)
    assert recorder.base_impl == impl


def test_record_attention_requires_return_extra_and_layout_inputs():
    stub = _Stub("sdpa")
    with pytest.raises(ValueError, match="return_extra"):
        _sample(stub, record_attention=True)
    data = _data()
    del data["num_frames"]
    with pytest.raises(ValueError, match="num_frames"):
        Alpamayo2Super.sample_trajectories_from_data(
            stub, data=data, return_extra=True, record_attention=True
        )
    stub._diffusion_expert_cuda_graph = object()
    with pytest.raises(ValueError, match="CUDA graph"):
        _sample(stub, return_extra=True, record_attention=True)


def test_recording_is_confined_to_the_diffusion_phase(monkeypatch):
    stub = _Stub("sdpa")
    seen = {}
    original = stub.expert.diffusion.sample

    def spy(*args, **kwargs):
        seen["impl_during_sampling"] = stub.decoder.config._attn_implementation
        return original(*args, **kwargs)

    monkeypatch.setattr(stub.expert.diffusion, "sample", spy)
    _sample(stub, return_extra=True, record_attention=True)
    assert seen["impl_during_sampling"] == RECORDING_ATTN_IMPLEMENTATION
    assert stub.decoder.config._attn_implementation == "sdpa"

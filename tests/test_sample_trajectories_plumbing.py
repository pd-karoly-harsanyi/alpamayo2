# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU test of the ``diffusion_seed`` and ``record_attention`` wiring in trajectory sampling,
on the stub model of ``alpamayo2_stub``."""

import numpy as np
import pytest
import torch

pytest.importorskip("hydra")
from alpamayo2_stub import (  # noqa: E402
    GENERATED,
    N_TOKENS,
    PROMPT,
    StubAlpamayo2Super,
    model_inputs,
    stub_text_functions,
)

from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super  # noqa: E402
from alpamayo2_super.models.expert_attention_recording import (  # noqa: E402
    RECORDING_ATTN_IMPLEMENTATION,
)


@pytest.fixture(autouse=True)
def _stub_text_functions(monkeypatch):
    stub_text_functions(monkeypatch)


def _sample(stub, **kwargs):
    kwargs.setdefault("diffusion_kwargs", {"inference_step": 3})
    return Alpamayo2Super.sample_trajectories_from_data(
        stub, data=model_inputs(), num_traj_samples=2, **kwargs
    )


def test_diffusion_seed_fixes_the_trajectory_regardless_of_the_global_stream():
    stub = StubAlpamayo2Super("sdpa")
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
    stub = StubAlpamayo2Super("sdpa")
    with pytest.raises(ValueError, match="generator"):
        _sample(stub, diffusion_seed=1, diffusion_kwargs={"generator": torch.Generator()})


@pytest.mark.parametrize("impl", ["sdpa", "eager"])
def test_record_attention_keeps_predictions_identical_and_returns_the_summary(impl):
    stub = StubAlpamayo2Super(impl)
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
    stub = StubAlpamayo2Super("sdpa")
    with pytest.raises(ValueError, match="return_extra"):
        _sample(stub, record_attention=True)
    data = model_inputs()
    del data["num_frames"]
    with pytest.raises(ValueError, match="num_frames"):
        Alpamayo2Super.sample_trajectories_from_data(
            stub, data=data, return_extra=True, record_attention=True
        )
    stub._diffusion_expert_cuda_graph = object()
    with pytest.raises(ValueError, match="CUDA graph"):
        _sample(stub, return_extra=True, record_attention=True)


def test_recording_is_confined_to_the_diffusion_phase(monkeypatch):
    stub = StubAlpamayo2Super("sdpa")
    seen = {}
    original = stub.expert.diffusion.sample

    def spy(*args, **kwargs):
        seen["impl_during_sampling"] = stub.decoder.config._attn_implementation
        return original(*args, **kwargs)

    monkeypatch.setattr(stub.expert.diffusion, "sample", spy)
    _sample(stub, return_extra=True, record_attention=True)
    assert seen["impl_during_sampling"] == RECORDING_ATTN_IMPLEMENTATION
    assert stub.decoder.config._attn_implementation == "sdpa"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for seeding the flow-matching sampler's initial noise through a generator."""

import torch

from alpamayo2_super.diffusion.flow_matching import FlowMatching


def _sampler() -> FlowMatching:
    return FlowMatching(x_dims=(64, 2), num_inference_steps=3)


def _zero_step(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    del t
    return torch.zeros_like(x)


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_same_generator_seed_gives_same_noise():
    sampler = _sampler()
    a = sampler.sample(batch_size=8, step_fn=_zero_step, generator=_generator(7))
    b = sampler.sample(batch_size=8, step_fn=_zero_step, generator=_generator(7))
    assert a.shape == (8, 64, 2)
    assert torch.equal(a, b)


def test_different_generator_seeds_differ():
    sampler = _sampler()
    a = sampler.sample(batch_size=8, step_fn=_zero_step, generator=_generator(7))
    b = sampler.sample(batch_size=8, step_fn=_zero_step, generator=_generator(8))
    assert not torch.equal(a, b)


def test_generator_is_independent_of_the_global_stream():
    sampler = _sampler()
    torch.manual_seed(0)
    torch.randn(1000)  # consume part of the global stream, as sentence sampling does
    a = sampler.sample(batch_size=8, step_fn=_zero_step, generator=_generator(7))
    torch.manual_seed(123)
    b = sampler.sample(batch_size=8, step_fn=_zero_step, generator=_generator(7))
    assert torch.equal(a, b)


def test_no_generator_consumes_the_global_stream_as_before():
    sampler = _sampler()
    torch.manual_seed(5)
    expected = torch.randn(8, 64, 2)
    torch.manual_seed(5)
    sampled = sampler.sample(batch_size=8, step_fn=_zero_step)
    assert torch.equal(sampled, expected)
    # and the global stream advanced: the next call differs
    assert not torch.equal(sampler.sample(batch_size=8, step_fn=_zero_step), expected)


def test_generator_matches_a_global_reseed_on_cpu():
    """A fresh generator seeded with ``s`` draws what a global ``manual_seed(s)`` draws first."""
    torch.manual_seed(5)
    a = torch.randn(3, 4)
    b = torch.randn(3, 4, generator=torch.Generator().manual_seed(5))
    assert torch.equal(a, b)

    sampler = _sampler()
    torch.manual_seed(11)
    global_noise = sampler.sample(batch_size=4, step_fn=_zero_step, temperature=0.6)
    seeded = sampler.sample(
        batch_size=4, step_fn=_zero_step, temperature=0.6, generator=_generator(11)
    )
    assert torch.equal(global_noise, seeded)


def test_temperature_scales_the_seeded_noise():
    sampler = _sampler()
    unit = sampler.sample(batch_size=2, step_fn=_zero_step, generator=_generator(3))
    scaled = sampler.sample(
        batch_size=2, step_fn=_zero_step, generator=_generator(3), temperature=0.5
    )
    assert torch.allclose(scaled, unit * 0.5)

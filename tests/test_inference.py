# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of ``predict_trajectories`` on the stub model: the task selection and tokenization
are replaced by the stub's ready-made model inputs, everything from the sampler on is real."""

import numpy as np
import pytest

pytest.importorskip("hydra")
from alpamayo2_stub import (  # noqa: E402
    N_TOKENS,
    StubAlpamayo2Super,
    model_inputs,
    stub_text_functions,
)

from alpamayo2_super import inference  # noqa: E402

SAMPLE = {"any": "inference sample"}


@pytest.fixture(autouse=True)
def _stub_input_path(monkeypatch):
    stub_text_functions(monkeypatch)
    monkeypatch.setattr(inference, "select_task_input", lambda sample, task: sample)
    monkeypatch.setattr(
        inference.helper, "prepare_model_inputs", lambda data, config, tokenizer: model_inputs()
    )


def _predict(stub, **kwargs):
    kwargs.setdefault("num_samples", 2)
    kwargs.setdefault("seed", 7)
    kwargs.setdefault("diffusion_steps", 3)
    return inference.predict_trajectories(stub, SAMPLE, **kwargs)


def test_returns_paired_numpy_trajectories_and_sentences():
    prediction = _predict(StubAlpamayo2Super("sdpa"))
    assert prediction.trajectories_xyz.shape == (2, N_TOKENS, 3)
    assert prediction.trajectories_xyz.dtype == np.float32
    assert prediction.reasoning_texts == ["sentence", "sentence"]
    assert prediction.attention_arrays is None and prediction.attention_layout is None


def test_the_seed_reproduces_the_call_and_recording_does_not_change_it():
    stub = StubAlpamayo2Super("sdpa")
    first = _predict(stub, seed=7)
    again = _predict(stub, seed=7)
    other = _predict(stub, seed=8)
    recorded = _predict(stub, seed=7, record_attention=True)
    assert np.array_equal(first.trajectories_xyz, again.trajectories_xyz)
    assert not np.array_equal(first.trajectories_xyz, other.trajectories_xyz)
    assert np.array_equal(first.trajectories_xyz, recorded.trajectories_xyz)
    assert recorded.attention_arrays["attn_img"].shape == (2, 8)
    assert [it["offset"] for it in recorded.attention_layout["images"]] == [0, 4]


def test_a_sample_count_mismatch_is_an_error(monkeypatch):
    stub = StubAlpamayo2Super("sdpa")
    original = stub.sample_trajectories_from_data

    def one_trajectory_short(**kwargs):
        pred_xyz, pred_rot, logprob, extra = original(**kwargs)
        return pred_xyz[:, :, :1], pred_rot, logprob, extra

    monkeypatch.setattr(stub, "sample_trajectories_from_data", one_trajectory_short)
    with pytest.raises(RuntimeError, match="1 trajectories and 2 sentences for 2 samples"):
        _predict(stub)

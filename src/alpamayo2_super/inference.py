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

"""Run Alpamayo 2 Super on one inference sample and get plain arrays back.

The entry point for a caller that feeds its own frames and poses through
:func:`alpamayo2_super.inference_sample.inference_sample_from_arrays`: it selects the task's
input profile, tokenizes, moves the inputs to the model's device, seeds the sentence and the
diffusion sampling, and unpacks the prediction, so the caller never touches the task profiles,
the processor, the device moves or the ``extra`` dict.
"""

import dataclasses
import importlib.metadata
from typing import Any

import numpy as np
import torch

from alpamayo2_super import helper
from alpamayo2_super.input_profiles import select_task_input
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

# NVIDIA's sampling recipe for the chain-of-causation sentence and the diffusion expert.
DEFAULT_TOP_P = 0.98
DEFAULT_TEMPERATURE = 0.6
DEFAULT_DIFFUSION_STEPS = 10


@dataclasses.dataclass(frozen=True)
class TrajectoryPrediction:
    """One call's output as numpy; sample ``i`` of every field belongs together."""

    trajectories_xyz: np.ndarray  # (num_samples, num_future_steps, 3) float32, ego frame at t0
    reasoning_texts: list[str]  # the chain-of-causation sentence per trajectory
    # ``extra["attention"]`` of ``sample_trajectories_from_data``; None unless recorded.
    attention_arrays: dict[str, np.ndarray] | None
    attention_layout: dict[str, Any] | None


def load_model(checkpoint: str, device: str = "cuda:0") -> Alpamayo2Super:
    """Load the checkpoint at ``checkpoint`` (a directory or a Hugging Face id) in bfloat16."""
    return Alpamayo2Super.from_pretrained(checkpoint, dtype=torch.bfloat16, device_map=device)


def model_environment(model: Alpamayo2Super) -> dict[str, Any]:
    """What a results record should note about the model: package version and attention kernel."""
    return {
        "alpamayo2_super": importlib.metadata.version("alpamayo2-super"),
        "expert_attention_implementation": model.expert.expert.config._attn_implementation,
    }


def predict_trajectories(
    model: Alpamayo2Super,
    sample: dict[str, Any],
    *,
    num_samples: int,
    seed: int,
    diffusion_steps: int = DEFAULT_DIFFUSION_STEPS,
    record_attention: bool = False,
    top_p: float = DEFAULT_TOP_P,
    temperature: float = DEFAULT_TEMPERATURE,
    task: str = "trajectory",
) -> TrajectoryPrediction:
    """Sample ``num_samples`` trajectories, each with its sentence, for one inference sample.

    Args:
        model: A loaded :class:`Alpamayo2Super`.
        sample: An :func:`inference_sample_from_arrays` dict (the full camera ring; ``task``
            picks the cameras).
        num_samples: Trajectories to sample.
        seed: Seeds the global torch RNG (the sentence sampling) and the diffusion noise
            (``diffusion_seed``), so a repeated call reproduces its output and two calls with
            the same seed on different frames diffuse from the same noise.
        diffusion_steps: Inference steps of the diffusion expert.
        record_attention: Also return the expert's attention over the image tokens
            (:mod:`alpamayo2_super.models.expert_attention_recording`).
        top_p: Nucleus threshold for the sentence.
        temperature: Sampling temperature for the sentence.
        task: Input profile to select (see ``input_profiles.TASK_INPUT_PROFILES``).

    Raises:
        RuntimeError: When the model returns a sample count other than ``num_samples``, so
            trajectories and sentences can never be mis-paired silently.
    """
    device = model.device
    data = select_task_input(sample, task)
    model_inputs = helper.to_device(
        helper.prepare_model_inputs(data, model.config, model.tokenizer), device=device
    )
    torch.manual_seed(seed)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        pred_xyz, _pred_rot, _logprob, extra = model.sample_trajectories_from_data(
            data=model_inputs,
            top_p=top_p,
            temperature=temperature,
            num_traj_samples=num_samples,
            diffusion_kwargs={"inference_step": diffusion_steps},
            return_extra=True,
            diffusion_seed=seed,
            record_attention=record_attention,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    trajectories = pred_xyz[0, 0].float().cpu().numpy()  # one input, one trajectory set
    texts = [str(text) for text in np.asarray(extra["cot"], dtype=object).reshape(-1).tolist()]
    if not (trajectories.shape[0] == num_samples == len(texts)):
        raise RuntimeError(
            f"the model returned {trajectories.shape[0]} trajectories and {len(texts)} "
            f"sentences for {num_samples} samples"
        )
    attention = extra.get("attention") if record_attention else None
    return TrajectoryPrediction(
        trajectories_xyz=trajectories,
        reasoning_texts=texts,
        attention_arrays=None if attention is None else attention["arrays"],
        attention_layout=None if attention is None else attention["layout"],
    )

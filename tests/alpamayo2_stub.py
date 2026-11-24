# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A CPU stand-in for ``Alpamayo2Super`` shared by the sampling tests.

The VLM is stubbed (prefill + generation replaced by a fixed sentence and a cache built by a tiny
Qwen3-VL text decoder standing in for the expert); everything after the sentence, from the expert
mask to the diffusion sampler and ``extra``, runs the real code. ``stub_text_functions`` replaces
the trajectory-token fusion and the sentence extraction, which the stub's tokenizer cannot serve.
"""

from types import SimpleNamespace

import torch
from transformers import GenerationConfig
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel

from alpamayo2_super.diffusion.flow_matching import FlowMatching
from alpamayo2_super.models import alpamayo2_super as model_module
from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super

HIDDEN, N_TOKENS, ACTION_DIM, EOS, IMAGE_TOKEN, PAD = 32, 4, 2, 99, 7, 0
PROMPT = [1, 2] + [IMAGE_TOKEN] * 4 + [3] + [IMAGE_TOKEN] * 4 + [4, 5]
GENERATED = [11, 12, EOS]


class ActionSpace:
    def get_action_space_dims(self) -> tuple[int, int]:
        return (N_TOKENS, ACTION_DIM)

    def action_to_traj(self, action, hist_xyz, hist_rot):
        """``hist_xyz`` is the last history waypoint per sample, (b, T, 3) here (T = 16)."""
        anchor = hist_xyz.reshape(hist_xyz.shape[0], -1, 3)[:, -1]
        xyz = torch.cat([action, torch.zeros_like(action[..., :1])], dim=-1) + anchor[:, None]
        rot = hist_rot.reshape(hist_rot.shape[0], -1, 3, 3)[:, -1][:, None]
        return xyz, rot.expand(-1, N_TOKENS, -1, -1)


class ActionInProj(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(ACTION_DIM + 1, HIDDEN)

    def forward(self, x, t):
        t = t.reshape(x.shape[0], 1, 1).expand(-1, x.shape[1], 1)
        return self.proj(torch.cat([x, t], dim=-1))


class StubAlpamayo2Super:
    """Just enough of ``Alpamayo2Super`` for ``sample_trajectories_from_data`` on the CPU."""

    device = torch.device("cpu")
    sample_trajectories_from_data = Alpamayo2Super.sample_trajectories_from_data

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
            action_space=ActionSpace(),
            diffusion=FlowMatching(x_dims=(N_TOKENS, ACTION_DIM), num_inference_steps=3),
            action_in_proj=ActionInProj(),
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


def model_inputs() -> dict:
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


def stub_text_functions(monkeypatch) -> None:
    """Make token fusion a no-op and extract one fixed sentence per sampled sequence."""
    monkeypatch.setattr(model_module, "fuse_traj_tokens", lambda *args: args[2])
    monkeypatch.setattr(
        model_module,
        "extract_text_tokens",
        lambda tokenizer, sequences, task: {"cot": ["sentence"] * sequences.shape[0]},
    )

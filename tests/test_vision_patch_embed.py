# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests: the vision patch embedding as a GEMM equals the conv3d."""

from types import SimpleNamespace

import pytest
import torch

from alpamayo2_super.models.vision_patch_embed import (
    apply_patch_embed_as_linear,
    patch_embed_as_linear,
)

transformers = pytest.importorskip("transformers")
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig  # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionPatchEmbed  # noqa: E402

IN_CHANNELS, TEMPORAL, PATCH, EMBED = 3, 2, 4, 16
PATCH_NUMEL = IN_CHANNELS * TEMPORAL * PATCH**2


def _qwen3_patch_embed() -> Qwen3VLVisionPatchEmbed:
    torch.manual_seed(0)
    config = Qwen3VLVisionConfig(
        in_channels=IN_CHANNELS,
        temporal_patch_size=TEMPORAL,
        patch_size=PATCH,
        hidden_size=EMBED,
        depth=1,
        num_heads=2,
        intermediate_size=32,
        out_hidden_size=16,
    )
    return Qwen3VLVisionPatchEmbed(config).float().eval()


class _NoBiasPatchEmbed(torch.nn.Module):
    """Stand-in for Qwen2-VL's patch embed, whose conv has no bias."""

    def __init__(self) -> None:
        super().__init__()
        self.in_channels, self.temporal_patch_size, self.patch_size = IN_CHANNELS, TEMPORAL, PATCH
        self.embed_dim = EMBED
        kernel = [TEMPORAL, PATCH, PATCH]
        self.proj = torch.nn.Conv3d(
            IN_CHANNELS, EMBED, kernel_size=kernel, stride=kernel, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states.view(-1, IN_CHANNELS, TEMPORAL, PATCH, PATCH)
        return self.proj(hidden_states.to(self.proj.weight.dtype)).view(-1, EMBED)


def _patches(num: int = 37) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(num, PATCH_NUMEL)


def test_linear_equals_conv_and_restore_brings_conv_back():
    patch_embed = _qwen3_patch_embed()
    x = _patches()
    conv_out = patch_embed(x)
    conv_forward = patch_embed.forward

    restore = patch_embed_as_linear(patch_embed)
    assert patch_embed.forward is not conv_forward
    linear_out = patch_embed(x)
    assert linear_out.shape == (x.shape[0], EMBED)
    assert torch.allclose(linear_out, conv_out, atol=1e-5)
    # only the forward changed: same parameters, same state_dict keys
    assert set(patch_embed.state_dict()) == {"proj.weight", "proj.bias"}

    restore()
    assert "forward" not in patch_embed.__dict__
    assert torch.equal(patch_embed(x), conv_out)


def test_weights_are_read_at_call_time():
    """Applying before the weights are loaded (as in ``__init__``) is fine."""
    patch_embed = _qwen3_patch_embed()
    patch_embed_as_linear(patch_embed)
    reference = _qwen3_patch_embed()
    torch.manual_seed(42)
    new_state = {k: torch.randn_like(v) for k, v in reference.state_dict().items()}
    reference.load_state_dict(new_state)
    patch_embed.load_state_dict(new_state)
    x = _patches()
    assert torch.allclose(patch_embed(x), reference(x), atol=1e-5)


def test_no_bias_patch_embed():
    patch_embed = _NoBiasPatchEmbed().eval()
    x = _patches()
    conv_out = patch_embed(x)
    patch_embed_as_linear(patch_embed)
    assert torch.allclose(patch_embed(x), conv_out, atol=1e-5)


def test_input_dtype_is_cast_to_the_weight_dtype():
    patch_embed = _qwen3_patch_embed()
    x = _patches().to(torch.float64)
    conv_out = patch_embed(x)
    patch_embed_as_linear(patch_embed)
    out = patch_embed(x)
    assert out.dtype == torch.float32
    assert torch.allclose(out, conv_out, atol=1e-5)


def test_rejects_a_conv_that_is_not_one_patch_per_output():
    patch_embed = _NoBiasPatchEmbed()
    patch_embed.proj = torch.nn.Conv3d(IN_CHANNELS, EMBED, kernel_size=3, stride=1, bias=False)
    with pytest.raises(ValueError):
        patch_embed_as_linear(patch_embed)


def test_apply_on_vlm_layout_and_skip_otherwise():
    patch_embed = _qwen3_patch_embed()
    vlm = SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(patch_embed=patch_embed)))
    assert apply_patch_embed_as_linear(vlm) is True
    assert "forward" in patch_embed.__dict__

    other = SimpleNamespace(model=SimpleNamespace())
    assert apply_patch_embed_as_linear(other) is False


def test_config_flag_default_and_off(monkeypatch):
    """``Alpamayo2SuperConfig.patch_embed_as_linear`` defaults to True and can be turned off."""
    pytest.importorskip("hydra")
    from alpamayo2_super.config import Alpamayo2SuperConfig

    default = Alpamayo2SuperConfig()
    assert default.patch_embed_as_linear is True
    assert Alpamayo2SuperConfig(patch_embed_as_linear=False).patch_embed_as_linear is False

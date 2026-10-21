"""Camera occlusion helpers: frame replacement and per-record summary."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.ablation import ablated, summarize_record  # noqa: E402


def test_ablated_blank_and_real():
    frames = torch.randint(0, 256, (6, 4, 3, 8, 12), dtype=torch.uint8)
    data = {"image_frames": frames, "other": "kept"}
    b = ablated(data, 2)
    assert b["other"] == "kept" and b["image_frames"].shape == frames.shape and b["image_frames"].dtype == torch.uint8
    assert torch.equal(b["image_frames"][0], frames[0]) and not torch.equal(b["image_frames"][2], frames[2])
    blank = b["image_frames"][2]
    assert (blank.float().std(dim=(0, 2, 3)) == 0).all()  # flat per channel
    assert torch.allclose(blank[0, :, 0, 0].float(), frames[2].float().mean(dim=(0, 2, 3)).round())
    assert torch.equal(frames[2], data["image_frames"][2])  # input untouched
    real = torch.zeros_like(frames)
    r = ablated(data, 5, real)
    assert torch.equal(r["image_frames"][5], real[5]) and torch.equal(r["image_frames"][4], frames[4])


def test_summarize_record():
    own = np.zeros((64, 3)); own[:, 0] = np.linspace(0, 10, 64)
    real = own.copy(); real[:, 0] *= 2
    p1 = own.copy(); p2 = real.copy()
    s = summarize_record(own, real, [p1, p2], ["camA", "camB"], ["a", "b"])
    assert s["camA"]["ade_vs_own_m"] == 0 and s["camA"]["ade_vs_real_m"] > 0 and s["camA"]["cot"] == "a"
    assert s["camB"]["ade_vs_real_m"] == 0 and s["camB"]["dist_diff_vs_own_m"] == 10.0
    s2 = summarize_record(own, None, [p1], ["camA"], ["a"])
    assert "ade_vs_real_m" not in s2["camA"]

"""Camera occlusion: which camera's pictures drive the decision, measured causally.

For one instant and arm the model is run again once per camera with that camera's four frames
replaced, same seed as the main prediction (common random numbers), and the trajectory shift is
recorded. Two replacements:

- ``blank``: the camera's frames become a flat image of their own mean colour. The shift says how much
  the prediction depends on that camera at all. Works for every arm.
- ``real``: for a render arm, the camera's frames are swapped back to the real frames of the same
  instant. The remaining distance to the real arm's prediction says how much of the replica effect
  that camera carries; if swapping one camera removes the shift, the defect is in that camera's render.

Unlike attention maps this needs no model internals and is immune to attention sinks; it costs one
forward per camera (6 x 1.7 s per record).

Stored per record (``run_export.py --ablate blank real``): npz ``ablate_blank_xyz`` / ``ablate_real_xyz``
(cameras, 64, 3) and ``ablate_cameras``; json ``ablate_blank`` / ``ablate_real`` with, per camera, ADE and
end-distance difference vs the arm's own prediction and vs the real arm's prediction, and the sentence.
``python -m pd_alpamayo_eval.ablation <run>`` prints the scene summary.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

MODES = ("blank", "real")


def camera_names_of(data: dict) -> list[str]:
    """Camera names in the order of ``data['image_frames']`` (after task selection)."""
    from alpamayo2_super.common.constants import CAMERA_NAMES_TO_INDICES

    inv = {v: k for k, v in CAMERA_NAMES_TO_INDICES.items()}
    return [inv[int(i)] for i in data["camera_indices"].reshape(-1).tolist()]


def ablated(data: dict, k: int, replacement: torch.Tensor | None = None) -> dict:
    """Copy of ``data`` with camera ``k``'s frames replaced: by ``replacement[k]`` or by their mean colour."""
    frames = data["image_frames"].clone()
    if replacement is not None:
        frames[k] = replacement[k].to(frames.dtype)
    else:
        mean = frames[k].float().mean(dim=(0, 2, 3), keepdim=True)  # per channel over the 4 frames
        frames[k] = mean.round().clamp(0, 255).to(frames.dtype).expand_as(frames[k])
    out = dict(data)
    out["image_frames"] = frames
    return out


def _ade(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:, :2] - b[:, :2], axis=-1).mean())


def _dist(a: np.ndarray) -> float:
    return float(np.linalg.norm(a[-1, :2]))


def summarize_record(own: np.ndarray, real: np.ndarray | None, preds: list[np.ndarray], cameras: list[str], cots: list[str]) -> dict[str, Any]:
    """Per-camera effect of one ablation mode: (64, 3) predictions -> json-able dict."""
    out: dict[str, Any] = {}
    for cam, p, cot in zip(cameras, preds, cots):
        d = {"ade_vs_own_m": round(_ade(p, own), 3), "dist_diff_vs_own_m": round(_dist(p) - _dist(own), 2), "cot": cot}
        if real is not None:
            d["ade_vs_real_m"] = round(_ade(p, real), 3)
            d["dist_diff_vs_real_m"] = round(_dist(p) - _dist(real), 2)
        out[cam] = d
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import json

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pd_alpamayo_eval.make_video import _cot_text, _load_run, _stage, cot_differs

    parser = argparse.ArgumentParser(description="Scene summary of a camera-ablation run.")
    parser.add_argument("run")
    parser.add_argument("--cache", type=Path, default=Path("~/.cache/alpamayo-eval").expanduser())
    args = parser.parse_args(argv)
    _, records = _load_run(_stage(args.run, args.cache))
    arms = sorted({a for _, a in records})
    for mode in MODES:
        key = f"ablate_{mode}"
        rows = [(k, r) for k, r in records.items() if r.get(key)]
        if not rows:
            continue
        cameras = list(next(iter(rows))[1][key])
        print(f"\n== {mode}: {'camera frames replaced by their mean colour' if mode == 'blank' else 'camera frames swapped back to real'} ({len(rows)} records)")
        print(f"{'arm':15s} {'metric':34s}" + "".join(f"{c[7:-6]:>13s}" for c in cameras))
        for arm in arms:
            recs = [r for (t0, a), r in rows if a == arm]
            if not recs:
                continue
            for metric, label in (("ade_vs_own_m", "ADE shift vs own pred, median (m)"), ("dist_diff_vs_own_m", "distance diff vs own, median (m)"),
                                  ("ade_vs_real_m", "ADE vs real after swap, median (m)")):
                vals = {c: [r[key][c][metric] for r in recs if metric in r[key][c]] for c in cameras}
                if not any(vals.values()):
                    continue
                print(f"{arm:15s} {label:34s}" + "".join(f"{np.median(vals[c]):13.2f}" if vals[c] else f"{'-':>13s}" for c in cameras))
            flips = {c: np.mean([cot_differs(r[key][c]["cot"], _cot_text(r)) for r in recs]) for c in cameras}
            print(f"{arm:15s} {'sentence changes (share)':34s}" + "".join(f"{flips[c]:13.0%}" for c in cameras))
        if mode == "real":
            real_recs = {t0: r for (t0, a), r in records.items() if a == "real" and r.get("pred_xyz") is not None}
            for arm in arms:
                if arm == "real":
                    continue
                recs = [(t0, r) for (t0, a), r in rows if a == arm and t0 in real_recs]
                if not recs:
                    continue
                before = np.median([_ade(r["pred_xyz"][0], real_recs[t0]["pred_xyz"][0]) for t0, r in recs])
                print(f"{arm:15s} {'paired ADE vs real BEFORE any swap':34s}{before:13.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

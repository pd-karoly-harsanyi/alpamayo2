"""Compare arms across runs on their common instants: paired ADE and the predicted-distance bias.

    pixi run -e dev compare-runs \\
        s3://pd-nebius-ray/alpamayo-eval/full_variations/runs/alpamayo-run-27f93c3e \\
        s3://pd-nebius-ray/alpamayo-eval/full_variations/runs/alpamayo-run-af7ff7bc

Every run's arms become columns ``<arm>@<run id suffix>``; the first run's ``real`` is the
reference. For each other column it prints, over the instants present in both: paired ADE against
the reference (median / p90), the signed difference in distance covered by the predicted 6.4 s
trajectory (median, and the share of instants longer / shorter by more than 1 m), ADE against the
recorded future where available, and how often the Chain-of-Causation sentence differs materially
(``cot_differs``: punctuation, case and filler ignored; any decision word change counts; otherwise
less than half the content words in common).

The point: ``real@seed1`` vs ``real@seed42`` is the seed-to-seed noise floor; ``rendered@seed42``
vs ``real@seed42`` is the replica effect. A replica effect that is not larger than the floor, or
not one-sided in distance, is not distinguishable from sampling noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.make_video import _cot_text, _load_run, _stage, cot_differs  # noqa: E402


def _end_distance(pred: np.ndarray) -> float:
    return float(np.linalg.norm(pred[0, -1, :2]))


HORIZONS = {"1s": 10, "2s": 20, "3s": 30, "6.4s": 64}
WEIGHT_TAU_S = 2.0
_W = np.exp(-np.arange(1, 65) * 0.1 / WEIGHT_TAU_S)
_W /= _W.sum()


def _ade(a: np.ndarray, b: np.ndarray, steps: int = 64) -> float:
    """Mean xy distance over the first ``steps`` of the 6.4 s horizon (10 steps = 1 s)."""
    return float(np.linalg.norm(a[0, :steps, :2] - b[0, :steps, :2], axis=-1).mean())


def _wade(a: np.ndarray, b: np.ndarray) -> float:
    """Time-weighted ADE: per-step distance weighted by exp(-t / 2 s), so the near future counts most."""
    return float((np.linalg.norm(a[0, :, :2] - b[0, :, :2], axis=-1) * _W).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run locations (s3:// or local); the first run's `real` is the reference.")
    parser.add_argument("--reference", default="real", help="Arm of the first run used as reference (default: real).")
    parser.add_argument("--cache", type=Path, default=Path("~/.cache/alpamayo-eval").expanduser())
    parser.add_argument("--json", type=Path, default=None, help="Also write the table as JSON.")
    parser.add_argument("--labels", nargs="+", default=None, help="One tag per run instead of the automatic seed / export / run-id tags, e.g. --labels old new")
    args = parser.parse_args()

    columns: dict[str, dict[int, dict]] = {}
    labels: dict[str, str] = {}
    loaded = []
    for run in args.runs:
        run_dir = _stage(run, args.cache)
        summary, records = _load_run(run_dir)
        loaded.append((summary.get("run_id", run_dir.name), summary, records))
    if args.labels and len(args.labels) != len(loaded):
        raise SystemExit(f"--labels needs one label per run ({len(loaded)} runs, {len(args.labels)} labels)")
    # Tag = seed when that tells the runs apart; otherwise (same seed on another export, e.g. a newer
    # render of the same clip) the export name; otherwise the run id.
    seeds = [s.get("seed") for _, s, _ in loaded]
    exports = [(s.get("export") or "").rstrip("/").rsplit("/", 1)[-1] for _, s, _ in loaded]
    for i, (rid, summary, records) in enumerate(loaded):
        if args.labels:
            tag = args.labels[i]
        elif seeds[i] is not None and seeds.count(seeds[i]) == 1:
            tag = f"seed{seeds[i]}"
        elif exports.count(exports[i]) == 1:
            tag = exports[i]
        else:
            tag = rid[-8:]
        for (t0, arm), rec in records.items():
            if rec.get("pred_xyz") is None:
                continue
            col = f"{arm}@{tag}"
            columns.setdefault(col, {})[t0] = rec
            labels[col] = rid
    ref_col = next(c for c in columns if c.startswith(args.reference + "@"))  # first run's reference arm
    ref = columns[ref_col]
    print(f"reference: {ref_col} ({labels[ref_col]}), {len(ref)} instants\n")
    header = f"{'column':34s} {'n':>4s} {'pairedADE med':>14s} {'p90':>6s} {'dist diff med':>14s} {'longer':>7s} {'shorter':>8s} {'ADE vs GT med':>14s} {'text differs':>13s}"
    print(header)
    print("-" * len(header))
    table = []
    by_horizon: dict[str, dict[str, tuple[float, float]]] = {}
    for col, recs in columns.items():
        if col == ref_col:
            continue
        common = sorted(set(recs) & set(ref))
        if not common:
            continue
        paired = np.array([_ade(recs[t]["pred_xyz"], ref[t]["pred_xyz"]) for t in common])
        by_horizon[col] = {}
        for name, steps in HORIZONS.items():
            v = np.array([_ade(recs[t]["pred_xyz"], ref[t]["pred_xyz"], steps) for t in common])
            by_horizon[col][name] = (float(np.median(v)), float(np.percentile(v, 90)))
        wv = np.array([_wade(recs[t]["pred_xyz"], ref[t]["pred_xyz"]) for t in common])
        by_horizon[col]["weighted"] = (float(np.median(wv)), float(np.percentile(wv, 90)))
        dist = np.array([_end_distance(recs[t]["pred_xyz"]) - _end_distance(ref[t]["pred_xyz"]) for t in common])
        gt = [recs[t]["min_ade_vs_gt_m"] for t in common if recs[t].get("min_ade_vs_gt_m") is not None]
        text = np.mean([cot_differs(_cot_text(recs[t]), _cot_text(ref[t])) for t in common])
        row = {
            "column": col,
            "run_id": labels[col],
            "n": len(common),
            "paired_ade_median_m": round(float(np.median(paired)), 3),
            "paired_ade_p90_m": round(float(np.percentile(paired, 90)), 3),
            "distance_diff_median_m": round(float(np.median(dist)), 2),
            "longer_frac": round(float(np.mean(dist > 1.0)), 2),
            "shorter_frac": round(float(np.mean(dist < -1.0)), 2),
            "ade_vs_gt_median_m": round(float(np.median(gt)), 3) if gt else None,
            "text_differs_frac": round(float(text), 2),
        }
        table.append(row)
        print(
            f"{col:34s} {row['n']:4d} {row['paired_ade_median_m']:14.3f} {row['paired_ade_p90_m']:6.2f} "
            f"{row['distance_diff_median_m']:+14.2f} {row['longer_frac']:7.0%} {row['shorter_frac']:8.0%} "
            f"{(row['ade_vs_gt_median_m'] if row['ade_vs_gt_median_m'] is not None else float('nan')):14.3f} {row['text_differs_frac']:13.0%}"
        )
    ref_gt = [r["min_ade_vs_gt_m"] for r in ref.values() if r.get("min_ade_vs_gt_m") is not None]
    print(f"\n{ref_col}: ADE vs GT median {np.median(ref_gt):.3f} m (n={len(ref_gt)})")
    if by_horizon:
        names = list(HORIZONS) + ["weighted"]
        print(f"\npaired ADE vs {ref_col} by horizon, median / p90 (m); 'weighted' = exp(-t/{WEIGHT_TAU_S:g} s) time weighting:")
        print(f"{'column':34s}" + "".join(f"{n:>15s}" for n in names))
        for col, vals in by_horizon.items():
            print(f"{col:34s}" + "".join(f"{vals[n][0]:7.2f} /{vals[n][1]:6.2f}" for n in names))
        for row in table:
            row["by_horizon"] = by_horizon.get(row["column"])
    if args.json:
        args.json.write_text(json.dumps({"reference": ref_col, "rows": table}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

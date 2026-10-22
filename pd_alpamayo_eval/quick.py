"""Small experiments in minutes: run the model on a few chosen instants, then look at the pictures.

    pixi run -e dev quick madrid --times 0.0 4.2 8.0                      # attention maps, all arms
    pixi run -e dev quick madrid --times 4.2 --arms real rendered --ablate blank real   # + camera occlusion
    pixi run -e dev quick madrid --render-only s3://.../runs/alpamayo-run-xxxx          # pictures of an existing run

What happens: the job is submitted to the Ray cluster with your Nebius credentials forwarded (same as
``submit-nebius run --wait --forward-nebius-credentials``), only the chosen instants are run (about
2 s per instant and arm plus a ~5 min pod start, 14 s per camera-occluded instant), results are
staged, and for every instant a contact sheet is written to ``outputs/quick/<name>/``: the four front
and cross cameras x arms, with the trajectory paths and the attention overlay (value-weighted, middle
layers) and, when occlusion was run, the per-camera effect under each tile. A ``summary.txt`` holds the
numbers. Nothing is rerun that already exists: pass ``--render-only`` with any run uri to just draw.

Scenes come from ``registry.py`` (the reference export of the scene); ``--export`` overrides.
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.registry import BUCKET, SCENES  # noqa: E402

SHEET_CAMERAS = ["camera_front_wide_120fov", "camera_front_tele_30fov", "camera_cross_left_120fov", "camera_cross_right_120fov"]
TILE_W, TILE_H = 560, 315


def _pixi() -> str:
    return shutil.which("pixi") or str(Path("~/.pixi/bin/pixi").expanduser())


def submit(run_args: list[str], repo: Path) -> str:
    """Submit and wait; return the run id. Streams the job log."""
    cmd = [_pixi(), "run", "-e", "launch", "submit-nebius", "run", "--wait", "--forward-nebius-credentials", "--", *run_args]
    print("$", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    run_id = None
    for line in proc.stdout:
        print(line, end="", flush=True)
        m = re.search(r"submitted\s*:\s*(alpamayo-\S+)", line)
        if m:
            run_id = m.group(1)
    proc.wait()
    if proc.returncode != 0 or not run_id:
        raise SystemExit(f"submit failed (exit {proc.returncode}, run id {run_id})")
    return run_id


def stage_needed(run_uri: str, cache: Path) -> None:
    """Stage the run and ONLY the t0 frames its records need (not the whole export: 2600 images for 10 needed)."""
    import json

    from pd_alpamayo_eval.export_common import MANIFEST_NAME
    from pd_alpamayo_eval.make_video import _load_run, _stage
    from pd_alpamayo_eval.s3util import NEBIUS_BUCKET, client, parse_s3

    run_dir = _stage(run_uri, cache)  # small: json/npz/results
    summary, records = _load_run(run_dir)
    export_uri = summary.get("export")
    if not export_uri or not export_uri.startswith("s3://"):
        return
    bucket, prefix = parse_s3(export_uri)
    s3 = client("nebius" if bucket == NEBIUS_BUCKET else "default")
    local = cache / bucket / prefix
    local.mkdir(parents=True, exist_ok=True)
    if not (local / MANIFEST_NAME).is_file():
        s3.download_file(bucket, f"{prefix}/{MANIFEST_NAME}", str(local / MANIFEST_NAME))
    manifest = json.loads((local / MANIFEST_NAME).read_text(encoding="utf-8"))
    frames_of = {int(inst["t0_ns"]): inst["frames"] for inst in manifest["instants"]}
    wanted = []
    for (t0, arm) in records:
        for camera, ts in frames_of.get(int(t0), {}).items():
            wanted.append(f"{arm}/{camera}/{int(ts[-1])}.jpg")
    n = 0
    for rel in wanted:
        dst = local / rel
        if dst.is_file():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, f"{prefix}/{rel}", str(dst))
        n += 1
    print(f"staged {n} frames of {len(wanted)} needed for {len(records)} records", flush=True)


def render(run_uri: str, out_dir: Path, cache: Path) -> list[Path]:
    from PIL import Image, ImageDraw

    from pd_alpamayo_eval.dashboard import CAMERA_LABELS, Session, _ablation_effects
    from pd_alpamayo_eval.make_video import _cot_text

    stage_needed(run_uri, cache)
    s = Session([run_uri], cache, stage_exports=False)
    arms = list(s.columns)
    weighting = "attention x value norm" if s.has_vw else "attention"
    band = "mid layers (16-47)" if s.has_layers else None
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"run {run_uri}", f"scene {s.manifest['scene']}  arms {arms}  attention {'yes' if s.has_attention else 'no'} "
             f"({weighting}{', ' + band if band else ''})  occlusion {sorted(s.has_ablation) or 'no'}", ""]
    written = []
    for idx, t0 in enumerate(s.t0s):
        if not any(s.columns[a].get(t0) for a in arms):
            continue
        ref = s.columns[s.reference].get(t0)
        sheet = Image.new("RGB", (len(arms) * TILE_W, len(SHEET_CAMERAS) * TILE_H + 22 * len(arms)))
        lines.append(f"t={s.times[idx]:.1f}s (instant {idx + 1}/{len(s.t0s)})")
        for c, arm in enumerate(arms):
            rec = s.columns[arm].get(t0)
            if rec is None:
                continue
            paired = (float(np.linalg.norm(rec["pred_xyz"][0, :, :2] - ref["pred_xyz"][0, :, :2], axis=-1).mean())
                      if (ref is not None and arm != s.reference and rec.get("pred_xyz") is not None) else None)
            head = f"{arm}: {_cot_text(rec)}" + (f"   [paired ADE vs {s.reference} {paired:.2f} m]" if paired is not None else "")
            lines.append("  " + head)
            for r, cam in enumerate(SHEET_CAMERAS):
                mode = "raw" if s.has_attention else "off"
                im = Image.open(io.BytesIO(s.thumbnail(arm, cam, idx, overlay=True, attention=mode, band=band, weighting=weighting))).convert("RGB").resize((TILE_W, TILE_H))
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, TILE_W, 14], fill=(0, 0, 0))
                d.text((4, 1), f"{arm} · {CAMERA_LABELS.get(cam, cam)} · t={s.times[idx]:.1f}s", fill="white")
                for mode_name, key in (("blank", "blank"), ("real", "real")):
                    eff = _ablation_effects(s, arm, t0, rec, key) if key in s.has_ablation else None
                    if eff and cam in eff:
                        y = TILE_H - 16 - (14 if mode_name == "blank" and "real" in s.has_ablation and arm != s.reference else 0)
                        label = "blanked: moves" if mode_name == "blank" else "swap to real removes"
                        d.rectangle([0, y, TILE_W, y + 14], fill=(0, 0, 0))
                        d.text((4, y + 1), f"{label} {eff[cam]:.2f} m", fill=(255, 200, 80))
                sheet.paste(im, (c * TILE_W, r * TILE_H))
            # sentence strip under the tiles
            d = ImageDraw.Draw(sheet)
            y = len(SHEET_CAMERAS) * TILE_H + 22 * c
            d.rectangle([0, y, sheet.width, y + 22], fill=(30, 30, 30))
            d.text((6, y + 4), head[:230], fill="white")
            for key in ("blank", "real"):
                eff = _ablation_effects(s, arm, t0, rec, key) if key in s.has_ablation else None
                if eff:
                    top = sorted(eff.items(), key=lambda kv: -kv[1])
                    lines.append(f"     occlusion {key:5s}: " + ", ".join(f"{CAMERA_LABELS.get(k, k)} {v:.2f} m" for k, v in top))
        p = out_dir / f"t{s.times[idx]:05.1f}s.jpg"
        sheet.save(p, quality=88)
        written.append(p)
        lines.append("")
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", nargs="?", help=f"Scene name ({', '.join(SCENES)}) or omit with --export / --render-only.")
    parser.add_argument("--export", default=None, help="Export uri instead of the scene's reference export.")
    parser.add_argument("--times", type=float, nargs="+", default=None, help="Clip times in seconds to run (dashboard 't=').")
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="Instant indices to run (dashboard number minus 1).")
    parser.add_argument("--arms", nargs="+", default=None, help="Arms (default: all in the export).")
    parser.add_argument("--ablate", nargs="+", choices=["blank", "real"], default=None, help="Also run camera occlusion.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default=None, help="Output folder name under outputs/quick/ (default: scene + time stamp).")
    parser.add_argument("--render-only", default=None, metavar="RUN_URI", help="Skip the job; draw the sheets of this run.")
    parser.add_argument("--cache", type=Path, default=Path("~/.cache/alpamayo-eval").expanduser())
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]

    if args.render_only:
        run_uri = args.render_only
        name = args.name or run_uri.rstrip("/").rsplit("/", 1)[-1]
    else:
        if not (args.times or args.indices):
            raise SystemExit("give --times or --indices (a handful of instants), or --render-only <run uri>")
        if args.export:
            export = args.export
        elif args.scene in SCENES:
            export = f"{BUCKET}/{SCENES[args.scene][0][0]}"
        else:
            raise SystemExit(f"unknown scene {args.scene!r}; known: {', '.join(SCENES)} (or pass --export)")
        run_args = ["--export", export, "--seed", str(args.seed)]
        if args.times:
            run_args += ["--instant-times", *map(str, args.times)]
        if args.indices:
            run_args += ["--instant-indices", *map(str, args.indices)]
        if args.arms:
            run_args += ["--arms", *args.arms]
        if args.ablate:
            run_args += ["--ablate", *args.ablate]
        run_id = submit(run_args, repo)
        run_uri = f"{export.rstrip('/')}/runs/{run_id}"
        name = args.name or f"{args.scene or 'export'}_{time.strftime('%m%d_%H%M')}_{run_id[-8:]}"
    out_dir = repo / "outputs" / "quick" / name
    written = render(run_uri, out_dir, args.cache)
    print(f"\nwrote {len(written)} sheets -> {out_dir}/  (summary.txt alongside)")
    for p in written:
        print("  ", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

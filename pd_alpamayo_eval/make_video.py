"""Side-by-side video of one run: per instant, one tile per arm (real, rendered, variations).

    pixi run -e dev make-video s3://pd-nebius-ray/alpamayo-eval/w5s_3s/runs/alpamayo-run-0d74dd90 \\
        [--out w5s_3s_0d74dd90.mp4] [--fps 2] [--columns 2] [--cameras camera_front_wide_120fov]

The run location may be an ``s3://`` prefix (staged into ``--cache``, default ``~/.cache/alpamayo-eval``,
with the ``nebius`` profile for the Nebius bucket) or a local directory. The export the run was made
from is read from the run's ``summary.json`` and staged the same way; its frames are the images.

Each video frame is one instant. Each tile shows, top to bottom: the camera frame(s) at t0, a
bird's-eye plot with ego history, recorded future (GT), that arm's predicted trajectory and, on
every non-real tile, the real arm's prediction as a faint reference, and the Chain-of-Causation
text, boxed red when it differs from the real arm's. The header carries the instant index, the
time in the window and the ADE numbers. Arms are laid out ``--columns`` per row, so weather
variations (``rendered_<type>``) extend the mosaic automatically; an arm missing for an instant
shows a grey tile. Playback is slow on purpose (``--fps 2``) so the text can be read.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.export_common import MANIFEST_NAME, read_manifest  # noqa: E402
from pd_alpamayo_eval.s3util import NEBIUS_BUCKET, client, download_prefix, is_s3, parse_s3  # noqa: E402

ARM_COLORS = {"real": "#2ca02c", "rendered": "#ff7f0e"}
EXTRA_COLORS = ["#9467bd", "#17becf", "#d62728", "#8c564b", "#e377c2"]
CAMERA_LABELS = {
    "camera_cross_left_120fov": "cross left",
    "camera_front_wide_120fov": "front wide",
    "camera_cross_right_120fov": "cross right",
    "camera_rear_left_70fov": "rear left",
    "camera_rear_tele_30fov": "rear tele",
    "camera_rear_right_70fov": "rear right",
    "camera_front_tele_30fov": "front tele",
}
TILE_W = 640
IMG_H = 360
BEV_H = 400
TEXT_H = 110
HEADER_H = 70
PAD = 8


def _stage(uri_or_dir: str, cache: Path, skip_suffixes=(".png",)) -> Path:
    """Local directory for a run or export; s3:// locations are mirrored into the cache once."""
    if not is_s3(uri_or_dir):
        path = Path(uri_or_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    bucket, prefix = parse_s3(uri_or_dir)
    local = cache / bucket / prefix
    profile = "nebius" if bucket == NEBIUS_BUCKET else "default"
    s3 = client(profile)
    # NVIDIA's per-instant PNGs are large and not needed for the video.
    n = _download_filtered(s3, uri_or_dir, local, skip_suffixes)
    print(f"staged {uri_or_dir} -> {local} ({n} new files)")
    return local


def _download_filtered(s3, uri: str, local_dir: Path, skip_suffixes, skip_dirs=("runs/",)) -> int:
    from pd_alpamayo_eval.s3util import list_objects

    bucket, prefix = parse_s3(uri)
    n = 0
    for key, size in list_objects(s3, bucket, prefix):
        if key.endswith(tuple(skip_suffixes)):
            continue
        rel = key[len(prefix) :].lstrip("/")
        if rel.startswith(tuple(skip_dirs)):  # an export's runs/ live under it; staged separately
            continue
        dst = local_dir / rel
        if dst.is_file() and dst.stat().st_size == size:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        s3.download_file(bucket, key, str(tmp))
        os.replace(tmp, dst)
        n += 1
    return n


def _load_run(run_dir: Path) -> tuple[dict, dict[tuple[int, str], dict]]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    records = {}
    for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("error"):
            continue
        key = (int(rec["t0_ns"]), rec["arm"])
        npz_path = run_dir / f"{rec['t0_ns']}_{rec['arm']}.npz"
        if npz_path.is_file():
            with np.load(npz_path) as npz:
                rec["pred_xyz"] = npz["pred_xyz"]
                rec["gt_xyz"] = npz["gt_xyz"] if "gt_xyz" in npz else None
        records[key] = rec
    return summary, records


def _bev_limits(records: dict, manifest: dict) -> tuple[float, float, float]:
    """Fixed axes for the whole video: forward range and symmetric lateral range."""
    xs, ys = [0.0, 15.0], [3.0]
    for rec in records.values():
        for arr in (rec.get("pred_xyz"), rec.get("gt_xyz")):
            if arr is not None:
                a = np.asarray(arr).reshape(-1, 3)
                xs.append(float(a[:, 0].max()))
                xs.append(float(a[:, 0].min()))
                ys.append(float(np.abs(a[:, 1]).max()))
    return min(xs) - 2, max(xs) + 2, max(ys) + 2


def _render_bev(rec: dict | None, ref: dict | None, history_xyz: np.ndarray, color: str, limits, arm: str) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_min, x_max, y_abs = limits
    # Equal scale, but never narrower than a third of the forward range, so the strip stays readable.
    y_abs = max(y_abs, (x_max - x_min) / 3)
    fig, ax = plt.subplots(figsize=(TILE_W / 100, BEV_H / 100), dpi=100)
    # Driving direction up. The horizontal axis is y (positive = left in FLU) with the axis
    # inverted, so the left of the car is on the left of the plot.
    ax.plot(history_xyz[:, 1], history_xyz[:, 0], color="#1f77b4", lw=3, label="ego history (1.5 s)")
    if ref is not None and arm != "real" and ref.get("pred_xyz") is not None:
        for traj in ref["pred_xyz"]:
            ax.plot(traj[:, 1], traj[:, 0], color=ARM_COLORS["real"], lw=2, alpha=0.4, label="real prediction")
    if rec is not None and rec.get("pred_xyz") is not None:
        for traj in rec["pred_xyz"]:
            ax.plot(traj[:, 1], traj[:, 0], color=color, lw=3, label=f"{arm} prediction (6.4 s)")
    if rec is not None and rec.get("gt_xyz") is not None:  # drawn last so it stays visible on top
        gt = rec["gt_xyz"]
        ax.plot(gt[:, 1], gt[:, 0], color="black", lw=1.8, ls=(0, (4, 3)), label="recorded future")
        ax.plot(gt[-1, 1], gt[-1, 0], "x", color="black", ms=7)
    ax.plot(0, 0, "o", color="#ff7f0e", ms=8, label="ego at t0")
    ax.set_xlim(y_abs, -y_abs)
    ax.set_ylim(x_min, x_max)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("left <-   lateral y (m)   -> right")
    ax.set_ylabel("forward (m)")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=7, framealpha=0.85)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


def _cot_text(rec: dict | None) -> str:
    """The Chain-of-Causation sentence; tolerates the first run's stored array repr ``[['...']]``."""
    if not rec or not rec.get("cot"):
        return ""
    text = str(rec["cot"][0]).strip()
    while text[:1] in "[(" and text[-1:] in "])":
        text = text[1:-1].strip()
    return text.strip("'\" ")


def _font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _history_from_export(export_dir: Path, t0_ns: int) -> np.ndarray:
    from pd_alpamayo_eval.load_export import load_export

    arms = read_manifest(export_dir)["arms"]
    data = load_export(export_dir, t0_ns, arm=arms[0], include_future=False)
    return data["ego_history_xyz"][0, 0].numpy()


def _tile(
    export_dir: Path,
    instant: dict,
    arm: str,
    rec: dict | None,
    ref: dict | None,
    history_xyz: np.ndarray,
    color: str,
    limits,
    cameras: list[str],
    font,
    font_small,
):
    from PIL import Image, ImageDraw

    h = IMG_H * len(cameras) + BEV_H + TEXT_H
    tile = Image.new("RGB", (TILE_W, h), "#f4f4f4")
    draw = ImageDraw.Draw(tile)
    y = 0
    available = arm in instant.get("arms", [])
    for camera in cameras:
        if available:
            t_ns = instant["frames"][camera][-1]
            path = export_dir / arm / camera / f"{t_ns}.jpg"
            with Image.open(path) as im:
                frame = im.convert("RGB").resize((TILE_W, IMG_H), Image.BILINEAR)
            tile.paste(frame, (0, y))
        else:
            draw.rectangle([0, y, TILE_W, y + IMG_H], fill="#bbbbbb")
            draw.text((TILE_W // 2 - 90, y + IMG_H // 2 - 10), f"{arm}: not available", fill="black", font=font)
        draw.rectangle([0, y, 150, y + 22], fill=(0, 0, 0))
        draw.text((6, y + 3), f"{arm} | {CAMERA_LABELS.get(camera, camera)}", fill="white", font=font_small)
        y += IMG_H
    bev = _render_bev(rec if available else None, ref, history_xyz, color, limits, arm)
    tile.paste(Image.fromarray(bev).resize((TILE_W, BEV_H)), (0, y))
    y += BEV_H
    text = _cot_text(rec) if rec else ("(no output)" if available else "")
    differs = bool(ref and rec and ref is not rec and _cot_text(ref) != _cot_text(rec))
    draw.rectangle([PAD, y + 4, TILE_W - PAD, y + TEXT_H - 4], fill="white", outline="#d62728" if differs else "#999999", width=3 if differs else 1)
    lines = textwrap.wrap(str(text), width=68)[:4]
    for i, line in enumerate(lines):
        draw.text((PAD + 8, y + 10 + i * 20), line, fill="black", font=font_small)
    metrics = []
    if rec and rec.get("min_ade_vs_gt_m") is not None:
        metrics.append(f"ADE vs GT {rec['min_ade_vs_gt_m']:.2f} m")
    if rec and ref and arm != "real" and rec.get("pred_xyz") is not None and ref.get("pred_xyz") is not None:
        d = np.linalg.norm(rec["pred_xyz"][0, :, :2] - ref["pred_xyz"][0, :, :2], axis=-1).mean()
        metrics.append(f"vs real {d:.2f} m")
    if metrics:
        draw.text((PAD + 8, y + TEXT_H - 24), "   ".join(metrics), fill="#333333", font=font_small)
    return tile


def make_frames(run_dir: Path, export_dir: Path, out_frames: Path, columns: int, cameras: list[str], arms_filter: list[str] | None):
    from PIL import Image, ImageDraw

    summary, records = _load_run(run_dir)
    manifest = read_manifest(export_dir)
    instants = manifest["instants"]
    arms = arms_filter or [a for a in summary["arms"] if any(k[1] == a for k in records)]
    if "real" in arms:
        arms = ["real"] + [a for a in arms if a != "real"]
    colors = dict(ARM_COLORS)
    extra = iter(EXTRA_COLORS)
    for arm in arms:
        colors.setdefault(arm, next(extra, "#7f7f7f"))
    limits = _bev_limits(records, manifest)
    font, font_small = _font(22), _font(15)
    rows = (len(arms) + columns - 1) // columns
    tile_h = IMG_H * len(cameras) + BEV_H + TEXT_H
    W = columns * TILE_W + (columns + 1) * PAD
    H = HEADER_H + rows * tile_h + (rows + 1) * PAD
    t_first = int(manifest["instants"][0]["t0_ns"])
    out_frames.mkdir(parents=True, exist_ok=True)
    for i, instant in enumerate(instants):
        t0 = int(instant["t0_ns"])
        history = _history_from_export(export_dir, t0)
        ref = records.get((t0, "real"))
        canvas = Image.new("RGB", (W, H), "#202020")
        draw = ImageDraw.Draw(canvas)
        head = f"{manifest['scene']}   instant {i + 1}/{len(instants)}   t = +{(t0 - t_first) / 1e9:.1f} s   run {summary.get('run_id', '')}"
        draw.text((PAD + 4, 10), head, fill="white", font=font)
        ades = [f"{a}: {records[(t0, a)]['min_ade_vs_gt_m']:.2f} m" for a in arms if (t0, a) in records and records[(t0, a)].get("min_ade_vs_gt_m") is not None]
        draw.text((PAD + 4, 40), "minADE vs GT  " + "   ".join(ades), fill="#cccccc", font=font_small)
        for j, arm in enumerate(arms):
            r, c = divmod(j, columns)
            tile = _tile(export_dir, instant, arm, records.get((t0, arm)), ref, history, colors[arm], limits, cameras, font, font_small)
            canvas.paste(tile, (PAD + c * (TILE_W + PAD), HEADER_H + PAD + r * (tile_h + PAD)))
        canvas.save(out_frames / f"frame_{i:04d}.png")
        if (i + 1) % 10 == 0 or i + 1 == len(instants):
            print(f"  frames {i + 1}/{len(instants)}")
    return len(instants), (W, H)


def encode(frames_dir: Path, out: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg not found (it is in the pixi dev env: pixi run -e dev make-video ...)")
    base = [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "frame_%04d.png")]
    # Even dimensions and a 10 fps container (frames duplicated) for broad player support.
    filters = ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-r", "10", "-pix_fmt", "yuv420p"]
    for codec in (["-c:v", "libx264", "-crf", "20"], ["-c:v", "mpeg4", "-q:v", "3"]):
        result = subprocess.run(base + codec + filters + [str(out)], capture_output=True, text=True)
        if result.returncode == 0:
            return
    raise SystemExit(f"ffmpeg failed: {result.stderr.strip()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", help="Run location: s3://.../runs/<id> or a local directory with results.jsonl.")
    parser.add_argument("--export", default=None, help="Override the export location (default: from the run's summary.json).")
    parser.add_argument("--out", type=Path, default=None, help="Output .mp4 (default: <run id>.mp4 in the cwd).")
    parser.add_argument("--fps", type=float, default=2.0, help="Instants per second (default 2 = 0.5 s each).")
    parser.add_argument("--columns", type=int, default=2, help="Tiles per row.")
    parser.add_argument("--arms", nargs="+", default=None, help="Arms to show (default: all in the run; real first).")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["camera_front_wide_120fov"],
        help="Camera frames to stack in each tile, top to bottom.",
    )
    parser.add_argument("--cache", type=Path, default=Path("~/.cache/alpamayo-eval").expanduser())
    parser.add_argument("--keep-frames", type=Path, default=None, help="Also keep the PNG frames in this directory.")
    args = parser.parse_args()

    run_dir = _stage(args.run, args.cache)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    export_uri = args.export or summary.get("export")
    if not export_uri:
        raise SystemExit("summary.json has no export location; pass --export")
    export_dir = _stage(export_uri, args.cache, skip_suffixes=())
    if not (export_dir / MANIFEST_NAME).is_file():
        raise SystemExit(f"{export_dir} has no {MANIFEST_NAME}")
    out = args.out or Path(f"{summary.get('run_id', run_dir.name)}.mp4")

    frames_dir = args.keep_frames or Path(tempfile.mkdtemp(prefix="alpamayo-video-"))
    n, size = make_frames(run_dir, export_dir, frames_dir, args.columns, args.cameras, args.arms)
    encode(frames_dir, out, args.fps)
    if args.keep_frames is None:
        shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"wrote {out} : {n} instants at {args.fps} fps ({n / args.fps:.0f} s), {size[0]}x{size[1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

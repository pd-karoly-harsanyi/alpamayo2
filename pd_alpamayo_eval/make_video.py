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
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.export_common import MANIFEST_NAME, NUM_FUTURE_STEPS, STEP_NS, read_manifest  # noqa: E402
from pd_alpamayo_eval.load_export import EgoTrajectory  # noqa: E402
from pd_alpamayo_eval.s3util import NEBIUS_BUCKET, client, download_prefix, is_s3, parse_s3  # noqa: E402

ARM_COLORS = {"real": "#2ca02c", "rendered": "#ff7f0e", "rendered_rainy": "#1f77b4", "rendered_snowy": "#9467bd"}
EXTRA_COLORS = ["#17becf", "#bcbd22", "#8c564b", "#e377c2"]  # no red: red is the time cursor


def _shade(hex_color: str, factor: float) -> str:
    """Darken (factor < 1) or lighten (factor > 1) an ``#rrggbb`` colour, staying in gamut."""
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (1, 3, 5))
    if factor <= 1:
        r, g, b = (int(c * factor) for c in (r, g, b))
    else:
        r, g, b = (int(c + (255 - c) * (1 - 1 / factor)) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


def column_color(arm: str, run_index: int, extra: dict[str, str] | None = None) -> str:
    """Colour of a dashboard column: one hue per arm (real green, rendered orange, rainy blue, snowy
    purple), and a darker shade of that hue for every further run of the same arm (run 1: 0.65 x,
    run 2: 0.42 x, ...), so ``rendered`` and ``rendered@v2`` are the same family but never the same line.
    ``extra`` maps arms outside ARM_COLORS to colours (filled on first use by the caller)."""
    base = ARM_COLORS.get(arm) or (extra or {}).get(arm) or "#7f7f7f"
    return base if run_index == 0 else _shade(base, max(0.3, 0.65**run_index))
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
TIMELINE_H = 420  # error-over-time strip: one row per arm
PAD = 8
FRUSTUM_LENGTH_M = {30: 9.0, 70: 6.0, 120: 5.0}  # by horizontal FOV; drawn at the ego on the BEV
_FOV_RE = re.compile(r"(\d+)fov")


def _stage(uri_or_dir: str, cache: Path, skip_suffixes=(".png",), download: bool = True) -> Path:
    """Local directory for a run or export; s3:// locations are mirrored into the cache once.
    ``download=False`` only resolves the cache path (for callers that staged the needed files themselves)."""
    if not is_s3(uri_or_dir):
        path = Path(uri_or_dir).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path
    bucket, prefix = parse_s3(uri_or_dir)
    local = cache / bucket / prefix
    if not download:
        return local
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
        tmp = dst.with_name(f"{dst.name}.part-{os.getpid()}")  # per-process: concurrent stagers must not collide
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
                for name in ("attn_img", "attn_layout", "attn_mass", "attn_image_mass", "attn_img_layers", "attn_vw_img_layers", "attn_vw_mass"):  # expert attention, if recorded
                    rec[name] = npz[name] if name in npz else None
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


def _camera_frustums(manifest: dict) -> list[dict]:
    """Per camera: position and yaw in the ego frame, horizontal FOV, and whether the trajectory task uses it."""
    from scipy.spatial.transform import Rotation

    try:
        from alpamayo2_super.input_profiles import TASK_INPUT_PROFILES

        used = set(TASK_INPUT_PROFILES["trajectory"].camera_names)
    except Exception:  # pragma: no cover - src not on the path
        used = {c for c in CAMERA_LABELS if c != "camera_rear_tele_30fov"}
    frustums = []
    for name, cal in (manifest.get("cameras") or {}).items():
        pose = cal.get("cam_to_ego")
        if not pose:
            continue
        rot = Rotation.from_quat([pose["qx"], pose["qy"], pose["qz"], pose["qw"]]).as_matrix()
        optical_axis = rot[:, 2]  # camera z (RDF forward) expressed in the ego frame (FLU)
        m = _FOV_RE.search(name)
        fov = int(m.group(1)) if m else 90
        frustums.append(
            {
                "name": name,
                "used": name in used,
                "x": float(pose["x"]),
                "y": float(pose["y"]),
                "yaw": math.atan2(optical_axis[1], optical_axis[0]),
                "fov": math.radians(fov),
                "length": FRUSTUM_LENGTH_M.get(fov, 6.0),
            }
        )
    return frustums


def _partial_gt(trajectory, t0_ns: int) -> tuple[np.ndarray | None, bool]:
    """Recorded future in the ego frame at t0, as far as the pose log reaches; (None, False) if nothing."""
    future_ts = t0_ns + np.arange(1, NUM_FUTURE_STEPS + 1, dtype=np.int64) * STEP_NS
    future_ts = future_ts[future_ts <= trajectory.ts[-1]]
    if future_ts.size == 0:
        return None, False
    (t0_xyz,), t0_rot = trajectory.sample(np.array([t0_ns], dtype=np.int64))
    xyz, _ = trajectory.sample(future_ts)
    local = t0_rot[0].inv().apply(xyz - t0_xyz)
    return local, future_ts.size < NUM_FUTURE_STEPS


def _draw_ego(ax) -> None:
    from matplotlib.patches import Polygon

    # Ego footprint, ~4.5 x 2 m, drawn in plot coords (horizontal = y, vertical = x).
    ax.add_patch(Polygon([(1.0, -1.0), (-1.0, -1.0), (-1.0, 3.5), (1.0, 3.5)], closed=True, facecolor="#ff7f0e", alpha=0.45, edgecolor="#ff7f0e"))


def _draw_frustum_inset(ax, frustums: list[dict]) -> None:
    """Equal-aspect inset with the ego and the camera FOV wedges; the main plot is lateral-stretched."""
    from matplotlib.patches import Polygon

    ins = ax.inset_axes([0.70, 0.03, 0.28, 0.36])
    _draw_ego(ins)
    for f in frustums:
        angles = np.linspace(f["yaw"] - f["fov"] / 2, f["yaw"] + f["fov"] / 2, 12)
        pts = [(f["y"], f["x"])] + [(f["y"] + f["length"] * math.sin(a), f["x"] + f["length"] * math.cos(a)) for a in angles]
        if f["used"]:
            ins.add_patch(Polygon(pts, closed=True, facecolor="#1f77b4", alpha=0.14, edgecolor="#1f77b4", lw=0.8))
        else:
            ins.add_patch(Polygon(pts, closed=True, facecolor="none", edgecolor="#999999", lw=0.8, ls=(0, (2, 2))))
    r = 11.0
    ins.set_xlim(r, -r)
    ins.set_ylim(-r, r)
    ins.set_aspect("equal")
    ins.set_xticks([])
    ins.set_yticks([])
    n_used = sum(f["used"] for f in frustums)
    ins.set_title(f"model input: {n_used} camera FOVs", fontsize=7, pad=2)
    ins.patch.set_alpha(0.85)


def _render_bev(
    rec: dict | None,
    ref: dict | None,
    history_xyz: np.ndarray,
    color: str,
    limits,
    arm: str,
    gt_partial: tuple[np.ndarray | None, bool] = (None, False),
    frustums: list[dict] | None = None,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    x_min, x_max, y_abs = limits
    # Lateral range: the data's, but at least +-10 m so the frustums and the plot have width. Not
    # equal-aspect: with a 60 m forward range an equal-aspect strip would be unreadable.
    y_abs = max(y_abs, 10.0)
    fig, ax = plt.subplots(figsize=(TILE_W / 100, BEV_H / 100), dpi=100)
    # Driving direction up. The horizontal axis is y (positive = left in FLU) with the axis
    # inverted, so the left of the car is on the left of the plot.
    _draw_ego(ax)
    ax.plot(history_xyz[:, 1], history_xyz[:, 0], color="#1f77b4", lw=3, label="ego history (1.5 s)")
    if ref is not None and arm != "real" and ref.get("pred_xyz") is not None:
        for traj in ref["pred_xyz"]:
            ax.plot(traj[:, 1], traj[:, 0], color=ARM_COLORS["real"], lw=2, alpha=0.4, label="real prediction")
    if rec is not None and rec.get("pred_xyz") is not None:
        for traj in rec["pred_xyz"]:
            ax.plot(traj[:, 1], traj[:, 0], color=color, lw=3, label=f"{arm} prediction (6.4 s)")
    gt, truncated = gt_partial
    if gt is None and rec is not None and rec.get("gt_xyz") is not None:
        gt, truncated = rec["gt_xyz"], False
    if gt is not None:  # drawn last so it stays visible on top
        label = f"recorded future (ends at +{len(gt) / 10:.1f} s)" if truncated else "recorded future (6.4 s)"
        ax.plot(gt[:, 1], gt[:, 0], color="black", lw=1.8, ls=(0, (4, 3)), label=label)
        ax.plot(gt[-1, 1], gt[-1, 0], "x", color="black", ms=7)
    ax.plot(0, 0, "o", color="#ff7f0e", ms=6, label="ego at t0")
    ax.set_xlim(y_abs, -y_abs)
    ax.set_ylim(x_min, x_max)
    ax.set_xlabel("left <-   lateral y (m), stretched   -> right")
    ax.set_ylabel("forward (m)")
    ax.grid(alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, l in zip(handles, labels):
        seen.setdefault(l, h)
    if frustums:
        seen["camera FOV used (inset)"] = Patch(facecolor="#1f77b4", alpha=0.25, edgecolor="#1f77b4")
        if any(not f["used"] for f in frustums):
            seen["camera not used: rear tele"] = Patch(facecolor="none", edgecolor="#999999", ls=(0, (2, 2)))
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=7, framealpha=0.85)
    if frustums:
        _draw_frustum_inset(ax, frustums)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy()


def _paired_ade(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[0, :, :2] - b[0, :, :2], axis=-1).mean())


def _error_series(records: dict, instants: list[dict], arms: list[str]) -> dict[str, np.ndarray]:
    """Per instant: paired ADE vs real for each non-real arm, and the real arm's ADE vs GT (nan when missing)."""
    series: dict[str, list[float]] = {a: [] for a in arms if a != "real"}
    series["real vs recorded future"] = []
    for inst in instants:
        t0 = int(inst["t0_ns"])
        ref = records.get((t0, "real"))
        for a in arms:
            if a == "real":
                continue
            rec = records.get((t0, a))
            ok = ref is not None and rec is not None and ref.get("pred_xyz") is not None and rec.get("pred_xyz") is not None
            series[a].append(_paired_ade(rec["pred_xyz"], ref["pred_xyz"]) if ok else np.nan)
        gt = ref.get("min_ade_vs_gt_m") if ref else None
        series["real vs recorded future"].append(float(gt) if gt is not None else np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in series.items()}


def _render_timeline(times_s: np.ndarray, series: dict[str, np.ndarray], colors: dict[str, str], width: int) -> tuple[np.ndarray, np.ndarray]:
    """Base image of the error-over-time strip and the pixel x of every instant (for the cursor)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # One row per arm (same y scale); the real arm's error vs the recorded future is dotted in every row.
    arms = [k for k in series if k != "real vs recorded future"] or ["real vs recorded future"]
    gt_series = series.get("real vs recorded future")
    fig, axes = plt.subplots(len(arms), 1, figsize=(width / 100, TIMELINE_H / 100), dpi=100, sharex=True, squeeze=False)
    y_max = 0.5
    for values in series.values():
        if np.isfinite(values).any():
            y_max = max(y_max, float(np.nanpercentile(values, 98)))
    for ax, name in zip(axes[:, 0], arms):
        if gt_series is not None and name != "real vs recorded future":
            ax.plot(times_s, gt_series, color="black", lw=0.9, ls=":", label="real vs recorded future")
        values = series[name]
        label = f"{name} vs real (median {np.nanmedian(values):.2f} m)" if name != "real vs recorded future" else f"{name} (median {np.nanmedian(values):.2f} m)"
        ax.plot(times_s, values, color=colors.get(name, "#7f7f7f"), lw=1.6, label=label)
        ax.set_ylim(0, y_max * 1.1)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=6, ncol=2, framealpha=0.85)
        ax.tick_params(labelsize=7)
    axes[0, 0].set_xlim(float(times_s[0]), float(times_s[-1]) if times_s[-1] > times_s[0] else float(times_s[0]) + 1)
    axes[-1, 0].set_xlabel("time in scene (s)", fontsize=8)
    axes[0, 0].set_title("paired ADE of each arm against the real arm per instant (dotted: real arm against the recorded future); same scale", fontsize=8)
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    xs = np.array([axes[0, 0].transData.transform((t, 0.0))[0] for t in times_s])
    rgba = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    return rgba[..., :3].copy(), xs


def _cot_text(rec: dict | None) -> str:
    """The Chain-of-Causation sentence; tolerates the first run's stored array repr ``[['...']]``."""
    if not rec or not rec.get("cot"):
        return ""
    text = str(rec["cot"][0]).strip()
    while text[:1] in "[(" and text[-1:] in "])":
        text = text[1:-1].strip()
    return text.strip("'\" ")


# Words that carry the driving decision. A change in the action words, or in the direction attached to
# the action, counts as a different reasoning even when the rest of the sentence is the same.
ACTION_WORDS = frozenset(
    "stop stopping stopped slow slowing slower decelerate brake braking yield wait waiting accelerate "
    "accelerating speed faster proceed continue maintain keep nudge merge merging overtake overtaking "
    "pass passing turn turning change changing reverse pull park".split()
)
DIRECTION_WORDS = ("left", "right", "straight")
# Filler that only jitters between samples of the same sentence.
_FILLER = frozenset("a an the to of on in at for and or due because with by from is are be as it its this that there".split())


def _tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[a-z0-9]+", text.lower())


def cot_words(text: str) -> set[str]:
    """Lower-cased content words of a sentence: no punctuation, no articles/prepositions."""
    return {w for w in _tokens(text) if w not in _FILLER}


def cot_decision(text: str) -> tuple[tuple[str, ...], str | None]:
    """(action words in order, first direction word): 'Nudge left due to a car on the right' -> (('nudge',), 'left')."""
    toks = _tokens(text)
    actions = tuple(w for w in toks if w in ACTION_WORDS)
    direction = next((w for w in toks if w in DIRECTION_WORDS), None)
    return actions, direction


def cot_differs(a: str, b: str, min_overlap: float = 0.5) -> bool:
    """True when two Chain-of-Causation sentences say something materially different.

    Punctuation, case and filler words are ignored ('an open car door.' == 'the open car door'). A change in
    the action words (stop/slow/accelerate/nudge/yield/...) or in the direction attached to them
    (left/right/straight) always counts. Otherwise the sentences differ only when their content words
    overlap by less than ``min_overlap`` (Jaccard), so a reworded reason for the same action
    ('parked car' vs 'open car door') does not.
    """
    if cot_decision(a) != cot_decision(b):
        return True
    wa, wb = cot_words(a), cot_words(b)
    if wa == wb:
        return False
    if not wa or not wb:
        return True
    return len(wa & wb) / len(wa | wb) < min_overlap


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
    gt_partial: tuple[np.ndarray | None, bool] = (None, False),
    frustums: list[dict] | None = None,
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
    bev = _render_bev(rec if available else None, ref, history_xyz, color, limits, arm, gt_partial, frustums)
    tile.paste(Image.fromarray(bev).resize((TILE_W, BEV_H)), (0, y))
    y += BEV_H
    text = _cot_text(rec) if rec else ("(no output)" if available else "")
    differs = bool(ref and rec and ref is not rec and cot_differs(_cot_text(ref), _cot_text(rec)))
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
    H = HEADER_H + rows * tile_h + (rows + 1) * PAD + TIMELINE_H + PAD
    t_first = int(manifest["instants"][0]["t0_ns"])
    times_s = np.array([(int(inst["t0_ns"]) - t_first) / 1e9 for inst in instants])
    frustums = _camera_frustums(manifest)
    trajectory = EgoTrajectory(manifest)
    series = _error_series(records, instants, arms)
    timeline, cursor_xs = _render_timeline(times_s, series, colors, W - 2 * PAD)
    timeline_img = Image.fromarray(timeline)
    timeline_y = HEADER_H + rows * tile_h + (rows + 1) * PAD
    out_frames.mkdir(parents=True, exist_ok=True)
    for i, instant in enumerate(instants):
        t0 = int(instant["t0_ns"])
        history = _history_from_export(export_dir, t0)
        gt_partial = _partial_gt(trajectory, t0)
        ref = records.get((t0, "real"))
        canvas = Image.new("RGB", (W, H), "#202020")
        draw = ImageDraw.Draw(canvas)
        head = f"{manifest['scene']}   instant {i + 1}/{len(instants)}   t = +{times_s[i]:.1f} s   run {summary.get('run_id', '')}"
        draw.text((PAD + 4, 10), head, fill="white", font=font)
        parts = []
        for a in arms:
            rec = records.get((t0, a))
            if a == "real":
                gt = rec.get("min_ade_vs_gt_m") if rec else None
                parts.append(f"real vs GT {gt:.2f} m" if gt is not None else "real vs GT n/a")
            elif not np.isnan(series[a][i]):
                parts.append(f"{a} vs real {series[a][i]:.2f} m")
        draw.text((PAD + 4, 40), "this instant:  " + "   ".join(parts), fill="#cccccc", font=font_small)
        for j, arm in enumerate(arms):
            r, c = divmod(j, columns)
            tile = _tile(
                export_dir, instant, arm, records.get((t0, arm)), ref, history, colors[arm], limits, cameras, font, font_small,
                gt_partial=gt_partial, frustums=frustums,
            )
            canvas.paste(tile, (PAD + c * (TILE_W + PAD), HEADER_H + PAD + r * (tile_h + PAD)))
        # Error-over-time strip with a cursor at this instant and the running mean so far.
        canvas.paste(timeline_img, (PAD, timeline_y))
        cx = int(PAD + cursor_xs[i])
        draw.line([(cx, timeline_y + 22), (cx, timeline_y + TIMELINE_H - 30)], fill="#d62728", width=2)
        means = [f"{a} {np.nanmean(series[a][: i + 1]):.2f} m" for a in arms if a != "real" and np.isfinite(series[a][: i + 1]).any()]
        mean_text = "mean so far vs real:  " + "   ".join(means)
        draw.text((W - PAD - 14 - draw.textlength(mean_text, font=font_small), timeline_y + 24), mean_text, fill="#333333", font=font_small)
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

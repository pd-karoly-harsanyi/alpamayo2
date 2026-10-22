"""Interactive dashboard for inspecting runs: instant slider, arms side by side, plotly plots.

    pixi run -e dev dashboard --scene italy            # runs listed in registry.py: reference, v2, seed floor
    pixi run -e dev dashboard s3://pd-nebius-ray/alpamayo-eval/full_variations/runs/alpamayo-run-27f93c3e \\
        [s3://pd-nebius-ray/alpamayo-eval/full_variations/runs/alpamayo-run-af7ff7bc ...] [--port 8050]

then open http://localhost:8050. Runs are staged into ``~/.cache/alpamayo-eval`` (same cache as
make-video); images are read lazily per instant and thumbnailed on first use. The first run's
``real`` arm is the reference; arms of further runs appear as ``<arm>@seed<N>`` columns (e.g.
``real@seed1`` is the seed-to-seed noise floor). A run on ANOTHER export of the same scene (a newer
render of the same clip) is allowed as long as its instants are the same t0 grid: its columns are
tagged ``<arm>@<export name>`` and show that export's images. ``--labels`` overrides the tags.

What you get: a slider over instants with play / pause / step and speed (click the slider handle,
then the arrow keys step too); a card per arm with the camera frame(s) at t0, a bird's-eye plot
(history, recorded future as far as the pose log goes, the arm's prediction, the reference
prediction, the ego footprint and a camera-FOV inset), the Chain-of-Causation text flagged when it
differs from the reference, and the metrics of that instant; an error-over-time chart (paired ADE
of every column against the reference, dotted: reference against the recorded future) that you
can click to jump. Scene-level numbers: `compare_runs.py`.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.export_common import MANIFEST_NAME, NUM_FUTURE_STEPS, STEP_NS, read_manifest  # noqa: E402
from pd_alpamayo_eval.load_export import EgoTrajectory  # noqa: E402
from pd_alpamayo_eval.attention import camera_shares, text_share  # noqa: E402
from pd_alpamayo_eval.make_video import (  # noqa: E402
    ARM_COLORS,
    CAMERA_LABELS,
    EXTRA_COLORS,
    FRUSTUM_LENGTH_M,
    _camera_frustums,
    _cot_text,
    column_color,
    cot_differs,
    _load_run,
    _stage,
)

THUMB_W = 960
TXT_LINES = 3  # height of the model-reasoning box, in lines; longer sentences are clipped with an ellipsis
# Paired-ADE horizons selectable in the dashboard: steps of the 6.4 s prediction, None = time-weighted.
HORIZONS = {"1 s": 10, "2 s": 20, "3 s": 30, "6.4 s": 64, "weighted (exp, 2 s)": None}
HORIZON_W = np.exp(-np.arange(1, 65) * 0.1 / 2.0)
HORIZON_W /= HORIZON_W.sum()
OVERLAY_COLORS = {"history": (31, 119, 180), "gt": (0, 0, 0)}


def _hex_rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


class CameraProjector:
    """Project ego-frame (FLU, metres) points into a camera image with the manifest's calibration.

    pdncc stores the PAI-AV f-theta lenses as a Kannala-Brandt fit (OpenCV fisheye convention:
    theta_d = theta * (1 + k1 theta^2 + k2 theta^4 + k3 theta^6 + k4 theta^8)), a pinhole K and the
    camera pose in the ego frame (camera axes RDF -> ego FLU).
    """

    def __init__(self, cal: dict):
        from scipy.spatial.transform import Rotation

        k = np.asarray(cal["K"], dtype=float).reshape(3, 3)
        self.fx, self.fy, self.cx, self.cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        d = np.asarray(cal.get("D", []), dtype=float).reshape(-1)
        self.k = np.zeros(4)
        self.k[: min(4, d.size)] = d[:4]
        self.width, self.height = int(cal["width"]), int(cal["height"])
        pose = cal["cam_to_ego"]
        self.rot_ego_from_cam = Rotation.from_quat([pose["qx"], pose["qy"], pose["qz"], pose["qw"]]).as_matrix()
        self.t_ego = np.array([pose["x"], pose["y"], pose["z"]], dtype=float)

    def project(self, points_ego: np.ndarray, max_angle_deg: float = 88.0) -> tuple[np.ndarray, np.ndarray]:
        """Pixel coordinates (N,2) and a validity mask (in front of the camera, inside the image)."""
        p = (np.asarray(points_ego, dtype=float).reshape(-1, 3) - self.t_ego) @ self.rot_ego_from_cam  # = R^T (p - t)
        x, y, z = p[:, 0], p[:, 1], p[:, 2]
        r = np.hypot(x, y)
        theta = np.arctan2(r, z)
        theta_d = theta * (1 + self.k[0] * theta**2 + self.k[1] * theta**4 + self.k[2] * theta**6 + self.k[3] * theta**8)
        scale = np.where(r > 1e-9, theta_d / np.maximum(r, 1e-9), 0.0)
        u = self.fx * scale * x + self.cx
        v = self.fy * scale * y + self.cy
        valid = (z > 0.05) & (np.degrees(theta) < max_angle_deg) & (u >= 0) & (u < self.width) & (v >= 0) & (v < self.height)
        return np.stack([u, v], axis=-1), valid


def draw_paths(image, projector: CameraProjector, paths: list[tuple[np.ndarray, tuple[int, int, int], int, bool]], scale: float):
    """Draw projected polylines on a PIL image: (points_ego, rgb, width_px, dashed) per path."""
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image, "RGBA")
    for points, rgb, width, dashed in paths:
        uv, valid = projector.project(points)
        if valid.sum() < 3:  # e.g. a side camera seeing only the first point, under the car body
            continue
        uv = uv * scale
        pts = [(float(u), float(v)) if ok else None for (u, v), ok in zip(uv, valid)]
        for i in range(len(pts) - 1):
            a, b = pts[i], pts[i + 1]
            if a is None or b is None or (dashed and i % 2):
                continue
            draw.line([a, b], fill=(*rgb, 230), width=width)
        last = next((q for q in reversed(pts) if q is not None), None)
        if last is not None:
            draw.ellipse([last[0] - width, last[1] - width, last[0] + width, last[1] + width], fill=(*rgb, 230))
    return image
CAMERA_SETS = {
    "front wide": ["camera_front_wide_120fov"],
    "front three": ["camera_cross_left_120fov", "camera_front_wide_120fov", "camera_cross_right_120fov"],
    "all six used": [
        "camera_cross_left_120fov",
        "camera_front_wide_120fov",
        "camera_cross_right_120fov",
        "camera_rear_left_70fov",
        "camera_front_tele_30fov",
        "camera_rear_right_70fov",
    ],
    **{CAMERA_LABELS[c]: [c] for c in CAMERA_LABELS if c != "camera_front_wide_120fov"},
}


class Session:
    """Everything the callbacks need, loaded once."""

    def __init__(self, run_uris: list[str], cache: Path, labels: list[str] | None = None):
        self.cache = cache
        self.columns: dict[str, dict[int, dict]] = {}
        self.arm_of: dict[str, str] = {}
        self.run_of: dict[str, str] = {}
        self.export_dir_of: dict[str, Path] = {}  # per column: where its images live
        self.run_index_of: dict[str, int] = {}  # per column: 0 = reference run, 1, 2, ... = further runs (colour shade)
        self.export_of: dict[str, str] = {}  # per column: export name
        export_uri = None
        export_dirs: dict[str, Path] = {}
        if labels and len(labels) != len(run_uris):
            raise SystemExit(f"--labels needs one label per run ({len(run_uris)} runs, {len(labels)} labels)")
        for i, uri in enumerate(run_uris):
            run_dir = _stage(uri, cache)
            summary, records = _load_run(run_dir)
            run_export = summary.get("export")
            if not run_export:
                raise SystemExit(f"{uri}: no export recorded in summary.json")
            if export_uri is None:
                export_uri = run_export
            name = run_export.rstrip("/").rsplit("/", 1)[-1]
            if run_export not in export_dirs:
                export_dirs[run_export] = _stage(run_export, cache, skip_suffixes=())
                if not (export_dirs[run_export] / MANIFEST_NAME).is_file():
                    raise SystemExit(f"{export_dirs[run_export]} has no {MANIFEST_NAME}")
            if labels:
                tag = f"@{labels[i]}" if labels[i] else ""
            elif i == 0:
                tag = ""
            elif run_export != export_uri:
                tag = f"@{name}"
            else:
                tag = f"@seed{summary.get('seed')}"
            if tag == "@+":
                # Merge run: same export, seed and arms as an earlier run (predictions identical), carrying extra
                # fields (occlusion, attention). Its fields are added to the existing records instead of new columns.
                merged = 0
                for (t0, arm), rec in records.items():
                    base = self.columns.get(arm, {}).get(int(t0))
                    if base is None or base.get("pred_xyz") is None or not np.array_equal(base["pred_xyz"], rec["pred_xyz"]):
                        raise SystemExit(f"{uri}: cannot merge {arm} @ {t0}: no identical prediction in the earlier runs")
                    for k, v in rec.items():
                        if k not in base or base[k] is None:
                            base[k] = v
                    merged += 1
                print(f"merged {merged} records of {summary.get('run_id', run_dir.name)} into existing columns", flush=True)
                continue
            for (t0, arm), rec in records.items():
                col = f"{arm}{tag}"
                self.columns.setdefault(col, {})[int(t0)] = rec
                self.arm_of[col] = arm
                self.run_of[col] = summary.get("run_id", run_dir.name)
                self.export_dir_of[col] = export_dirs[run_export]
                self.export_of[col] = name
                self.run_index_of[col] = i
        self.export_dir = export_dirs[export_uri]
        self.manifest = read_manifest(self.export_dir)
        for other_uri, other_dir in export_dirs.items():  # a second export must sit on the same t0 grid (same clip, same pose log)
            if other_uri == export_uri:
                continue
            other = read_manifest(other_dir)
            if other.get("scene") != self.manifest.get("scene"):
                raise SystemExit(f"{other_uri} is scene {other.get('scene')!r}, the reference export is {self.manifest.get('scene')!r}")
            if [int(x["t0_ns"]) for x in other["instants"]] != [int(x["t0_ns"]) for x in self.manifest["instants"]]:
                raise SystemExit(f"{other_uri} has a different instant grid than {export_uri}; the dashboard joins columns by t0")
        self.instants = self.manifest["instants"]
        self.t0s = [int(inst["t0_ns"]) for inst in self.instants]
        self.times = np.array([(t - self.t0s[0]) / 1e9 for t in self.t0s])
        self.trajectory = EgoTrajectory(self.manifest)
        self.frustums = _camera_frustums(self.manifest)
        self.reference = "real" if "real" in self.columns else next(iter(self.columns))
        order = [self.reference] + [c for c in self.columns if c != self.reference]
        self.columns = {c: self.columns[c] for c in order}
        self.colors = {}
        extra_arms: dict[str, str] = {}
        extra = iter(EXTRA_COLORS)
        for col in self.columns:
            arm = self.arm_of[col]
            if arm not in ARM_COLORS and arm not in extra_arms:
                extra_arms[arm] = next(extra, "#7f7f7f")
            self.colors[col] = column_color(arm, self.run_index_of[col], extra_arms)
        self._series()
        self.has_attention = any(rec.get("attn_img") is not None for recs in self.columns.values() for rec in recs.values())
        self.has_ablation = {m for m in ("blank", "real") if any(rec.get(f"ablate_{m}") for recs in self.columns.values() for rec in recs.values())}
        self.has_layers = any(rec.get("attn_img_layers") is not None for recs in self.columns.values() for rec in recs.values())
        self.has_vw = any(rec.get("attn_vw_img_layers") is not None for recs in self.columns.values() for rec in recs.values())
        self.export_name = export_uri.rstrip("/").rsplit("/", 1)[-1]
        self.projectors = {}
        for name, cal in (self.manifest.get("cameras") or {}).items():
            try:
                self.projectors[name] = CameraProjector(cal)
            except (KeyError, ValueError):
                pass

    def _series(self) -> None:
        ref = self.columns[self.reference]
        self.paired: dict[str, np.ndarray] = {}
        self.gt_ade: dict[str, np.ndarray] = {}
        self.end_dist: dict[str, np.ndarray] = {}
        self.text_differs: dict[str, np.ndarray] = {}
        self.gt_ade_partial: dict[str, np.ndarray] = {}  # ADE over the available horizon where the full 6.4 s is missing
        self.gt_horizon_s = np.array([0.0] * len(self.t0s))
        partial_gts = {}
        for i, t0 in enumerate(self.t0s):
            g, truncated = self.partial_gt(t0)
            partial_gts[t0] = g if (g is not None and truncated and len(g) >= 10) else None  # at least 1 s
            self.gt_horizon_s[i] = (len(g) / 10.0) if g is not None else 0.0
        self.paired_h: dict[str, dict[str, np.ndarray]] = {}  # horizon label -> column -> per-instant paired ADE
        self.gt_ade_h: dict[str, dict[str, np.ndarray]] = {"3 s": {}, "6.4 s": {}}
        gts = {t0: self.partial_gt(t0)[0] for t0 in self.t0s}  # full or partial recorded future (local frame)
        for col, recs in self.columns.items():
            paired, gt, gtp, end, diff = [], [], [], [], []
            per_h = {h: [] for h in HORIZONS}
            gt3, gt64 = [], []
            for t0 in self.t0s:
                rec = recs.get(t0); g = gts[t0]
                ok = rec is not None and rec.get("pred_xyz") is not None and g is not None
                gt3.append(float(np.linalg.norm(rec["pred_xyz"][0, :30, :2] - g[:30, :2], axis=-1).mean()) if ok and len(g) >= 30 else np.nan)
                gt64.append(float(np.linalg.norm(rec["pred_xyz"][0, :64, :2] - g[:64, :2], axis=-1).mean()) if ok and len(g) >= 64 else np.nan)
            self.gt_ade_h["3 s"][col] = np.array(gt3)
            self.gt_ade_h["6.4 s"][col] = np.array(gt64)
            for t0 in self.t0s:
                rec, r = recs.get(t0), ref.get(t0)
                ok = rec is not None and rec.get("pred_xyz") is not None
                okr = ok and r is not None and r.get("pred_xyz") is not None
                dist = np.linalg.norm(rec["pred_xyz"][0, :, :2] - r["pred_xyz"][0, :, :2], axis=-1) if okr else None
                paired.append(float(dist.mean()) if okr else np.nan)
                for h, steps in HORIZONS.items():
                    per_h[h].append((float((dist * HORIZON_W).sum()) if steps is None else float(dist[:steps].mean())) if okr else np.nan)
                gt.append(float(rec["min_ade_vs_gt_m"]) if ok and rec.get("min_ade_vs_gt_m") is not None else np.nan)
                g = partial_gts[t0]
                gtp.append(float(np.linalg.norm(rec["pred_xyz"][0, : len(g), :2] - g[:, :2], axis=-1).mean()) if (ok and g is not None and rec.get("min_ade_vs_gt_m") is None) else np.nan)
                end.append(float(np.linalg.norm(rec["pred_xyz"][0, -1, :2])) if ok else np.nan)
                diff.append(float(cot_differs(_cot_text(rec), _cot_text(r))) if (rec and r) else np.nan)
            self.paired[col] = np.array(paired)
            for h in HORIZONS:
                self.paired_h.setdefault(h, {})[col] = np.array(per_h[h])
            self.gt_ade[col] = np.array(gt)
            self.gt_ade_partial[col] = np.array(gtp)
            self.end_dist[col] = np.array(end)
            self.text_differs[col] = np.array(diff)
        x_all = [0.0, 15.0]
        for recs in self.columns.values():
            for rec in recs.values():
                for arr in (rec.get("pred_xyz"), rec.get("gt_xyz")):
                    if arr is not None:
                        x_all += [float(np.asarray(arr).reshape(-1, 3)[:, 0].max()), float(np.asarray(arr).reshape(-1, 3)[:, 0].min())]
        self.forward_range = (min(x_all) - 3, max(x_all) + 3)
        finite = [v for arr in list(self.paired.values()) + [self.gt_ade[self.reference]] for v in arr if np.isfinite(v)]
        self.error_y_max = max(0.5, float(np.percentile(finite, 99))) * 1.05 if finite else 1.0
        self.error_y_max_h = {}
        for h, cols in self.paired_h.items():
            vals = [v for arr in cols.values() for v in arr if np.isfinite(v)]
            self.error_y_max_h[h] = max(0.2, float(np.percentile(vals, 99))) * 1.05 if vals else 1.0

    @lru_cache(maxsize=4096)
    def history(self, t0: int) -> np.ndarray:
        ts = t0 + np.arange(-15, 1, dtype=np.int64) * STEP_NS
        xyz, rot = self.trajectory.sample(ts)
        return rot[-1].inv().apply(xyz - xyz[-1])

    @lru_cache(maxsize=4096)
    def partial_gt(self, t0: int):
        future = t0 + np.arange(1, NUM_FUTURE_STEPS + 1, dtype=np.int64) * STEP_NS
        future = future[future <= self.trajectory.ts[-1]]
        if future.size == 0:
            return None, False
        (t0_xyz,), t0_rot = self.trajectory.sample(np.array([t0], dtype=np.int64))
        xyz, _ = self.trajectory.sample(future)
        return t0_rot[0].inv().apply(xyz - t0_xyz), future.size < NUM_FUTURE_STEPS

    def attention_grid(self, col: str, index: int, camera: str, mode: str, band: str | None = None, weighting: str | None = None,
                       scale: str = "frame"):
        return _attention_grid_impl(self, col, index, camera, mode, band, weighting, scale)

    def thumbnail(self, col: str, camera: str, index: int, overlay: bool = False, attention: str = "off", band: str | None = None,
                  weighting: str | None = None, scale: str = "frame") -> bytes:
        from PIL import Image

        arm = self.arm_of[col]
        inst = self.instants[index]
        t_ns = int(inst["frames"][camera][-1])
        src = self.export_dir_of[col] / arm / camera / f"{t_ns}.jpg"
        thumb = self.cache / "thumbs" / self.export_of[col] / arm / camera / f"{t_ns}.jpg"
        if not thumb.is_file():
            thumb.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_W, THUMB_W))
                im.save(thumb, quality=88)
        projector = self.projectors.get(camera)
        t0 = self.t0s[index]
        rec = self.columns[col].get(t0)
        heat = self.attention_grid(col, index, camera, attention, band, weighting, scale) if (attention in ATTENTION_MODES and rec is not None) else None
        draw = overlay and projector is not None
        if heat is None and not draw:
            return thumb.read_bytes()
        with Image.open(thumb) as im:
            im = im.convert("RGB")
            if heat is not None:  # first, so the paths stay on top
                grid, vmax, diverging = heat
                im = _blend_heatmap(im, grid, vmax, diverging)
            if draw:
                # Paths at t0: the reference prediction (on non-reference arms), this arm's prediction, the recorded future.
                ref = self.columns[self.reference].get(t0)
                paths = []
                if col != self.reference and ref is not None and ref.get("pred_xyz") is not None:
                    paths.append((ref["pred_xyz"][0], _hex_rgb(self.colors[self.reference]), 3, False))
                if rec is not None and rec.get("pred_xyz") is not None:
                    paths.append((rec["pred_xyz"][0], _hex_rgb(self.colors[col]), 5, False))
                gt, _ = self.partial_gt(t0)
                if gt is not None:  # last, thin and dashed, so it stays visible on top of the predictions
                    paths.append((gt, OVERLAY_COLORS["gt"], 2, True))
                scale = im.width / projector.width
                draw_paths(im, projector, paths, scale)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
        return buf.getvalue()


HEAT_ALPHA = 0.8  # peak opacity of the attention heatmap
HEAT_GAMMA = 0.7  # v ** gamma before colouring: lifts the mid-range so a map is not one hot cell and nothing else
HEAT_FLOOR_PCT = 50  # the frame's median is subtracted first: only what stands out from the floor gets colour
# Colour scale of the heatmap: "instant" = one scale for all 24 frames of the instant (cameras comparable, a
# camera with a lower peak looks fainter); "frame" = each frame scaled to its own peak (maximum contrast).
HEAT_SCALES = ("frame", "instant")

# What the heatmap shows. Raw maps are dominated by "attention sinks": a few background patches (sky)
# that the model uses as global stores, the same on every instant (maps correlate ~0.75 between
# instants; late layers put half the attention into 5 % of the cells). Subtracting the scene average
# leaves what THIS instant attends to; subtracting the real arm shows what the render changed.
ATTENTION_MODES = {
    "minus scene average": ("all", "baseline"),
    "minus real (paired)": ("all", "paired"),
    "raw": ("all", "raw"),  # with 'attention x value norm' this is a usable map; with plain attention it is sink-dominated
    # camera occlusion (causal, needs a run made with --ablate): no image heatmap, the bird's-eye wedges carry the numbers
    "occlusion: blank camera": (None, "ablate_blank"),
    "occlusion: real swap": (None, "ablate_real"),
}
ABLATION_MODES = {m for m, (_, k) in ATTENTION_MODES.items() if k.startswith("ablate_")}
# Layer bands of the expert (64 layers), selectable when the run stored every layer (attn_img_layers).
# Diagnostic run bd4af965 (Madrid, 2026-10-21): early layers diffuse, deep layers sink-dominated, the
# middle band is where attention sits on content (lead cars, the traffic light).
LAYER_BANDS = {"mid layers (16-47)": (16, 48), "early layers (0-15)": (0, 16), "late layers (48-63)": (48, 64), "all layers": (0, 64)}
# Weighting of the map: plain attention weights, or attention x value norm (the token's actual contribution
# to the expert's output; attention sinks have large weights but near-zero values, so they drop out).
WEIGHTINGS = {"attention x value norm": "vw", "attention": "attn"}
ATTENTION_HELP = ("model attention: where the trajectory expert's 64 tokens read from the image (attention x value norm, so attention sinks drop out; "
                  "middle layers of the expert, where attention sits on content; averaged over heads and diffusion steps; each frame scaled to its own peak "
                  "above its floor, so a bright spot means 'this stands out in this camera', not 'this camera matters most'). "
                  "occlusion: causal, from a run made with --ablate: the bird's-eye wedges show how much the prediction moves when a camera is blanked, "
                  "or how much of the gap to the real prediction closes when a camera is swapped back to real.")


def _attention_grids(rec: dict, depth, weighting: str = "attn") -> dict[tuple[str, int], np.ndarray] | None:
    """{(camera, frame): (gh, gw)} grids of one record. ``depth``: 'all' | 'late' (the stored averages) or a
    (first, last+1) layer band, served from the per-layer arrays when the run stored every layer.
    ``weighting`` 'vw' = attention x value norm (needs ``attn_vw_img_layers``), 'attn' = plain weights."""
    if rec is None or rec.get("attn_img") is None:
        return None
    layout = json.loads(str(rec["attn_layout"]))
    if weighting == "vw":
        layers = rec.get("attn_vw_img_layers")
        if layers is None:
            return None
        band = depth if isinstance(depth, tuple) else ((48, 64) if depth == "late" else (0, layers.shape[0]))
        flat = np.asarray(layers[band[0] : band[1]], dtype=np.float32).mean(0)
    elif isinstance(depth, tuple):
        layers = rec.get("attn_img_layers")
        if layers is None:
            return None
        flat = np.asarray(layers[depth[0] : depth[1]], dtype=np.float32).mean(0)
    else:
        flat = np.asarray(rec["attn_img"], dtype=np.float32)[1 if depth == "late" else 0]
    return {(it["camera"], it["frame"]): flat[it["offset"] : it["offset"] + it["gh"] * it["gw"]].reshape(it["gh"], it["gw"]) for it in layout["images"]}


def _session_attention_baseline(s: "Session", depth, weighting: str = "attn") -> dict[tuple[str, int], np.ndarray]:
    """Mean attention grid per (camera, frame) over every record of every column; cached on the session."""
    cache = s.__dict__.setdefault("_attn_baseline", {})
    ck = (depth, weighting)
    if ck not in cache:
        sums: dict[tuple[str, int], np.ndarray] = {}
        counts: dict[tuple[str, int], int] = {}
        for recs in s.columns.values():
            for rec in recs.values():
                g = _attention_grids(rec, depth, weighting)
                if not g:
                    continue
                for k, v in g.items():
                    if k in sums and sums[k].shape == v.shape:
                        sums[k] += v
                        counts[k] += 1
                    elif k not in sums:
                        sums[k] = v.copy()
                        counts[k] = 1
        cache[ck] = {k: sums[k] / counts[k] for k in sums}
    return cache[ck]


def _attention_grid_impl(s: "Session", col: str, index: int, camera: str, mode: str, band: str | None = None, weighting: str | None = None,
                         scale: str = "frame"):
    """(grid, vmax, diverging) for one camera's t0 frame, or None. vmax is shared by the instant's 24 frames.
    ``band``: a LAYER_BANDS key; used instead of the mode's stored depth when the run has every layer.
    ``weighting``: a WEIGHTINGS key; 'attention x value norm' needs the run to have recorded it."""
    depth, kind = ATTENTION_MODES[mode]
    if depth is None:  # occlusion modes have no per-pixel map
        return None
    if s.has_layers and band in LAYER_BANDS:  # the band overrides the mode's stored depth whenever every layer is available
        depth = LAYER_BANDS[band]
    w = WEIGHTINGS.get(weighting or "", "attn")
    if w == "vw" and not s.has_vw:
        w = "attn"
    t0 = s.t0s[index]
    rec = s.columns[col].get(t0)
    grids = _attention_grids(rec, depth, w)
    if not grids:
        return None
    frame = max(f for _, f in grids)
    if kind == "raw":
        diff, diverging = grids, False
    elif kind == "baseline":
        base = _session_attention_baseline(s, depth, w)
        diff = {k: np.clip(v - base[k], 0.0, None) if k in base and base[k].shape == v.shape else v for k, v in grids.items()}
        diverging = False
    else:  # paired: this arm minus the real arm at the same instant
        ref = _attention_grids(s.columns[s.reference].get(t0), depth, w)
        if not ref or col == s.reference:
            return None
        diff = {k: v - ref[k] for k, v in grids.items() if k in ref and ref[k].shape == v.shape}
        diverging = True
    if (camera, frame) not in diff:
        return None
    if scale == "frame":
        vmax = float(np.abs(diff[(camera, frame)]).max()) or 1.0
    else:
        vmax = max(float(np.abs(v).max()) for v in diff.values()) or 1.0
    return diff[(camera, frame)], vmax, diverging


def _blend_heatmap(im, heat: np.ndarray, vmax: float, diverging: bool = False):
    """Blend an attention grid over a PIL image; opacity proportional to |value|.

    Sequential maps use inferno (dark red to yellow); diverging maps (this arm minus real) use coolwarm,
    red = more attention than real, blue = less. The grid is the vision tower's token grid of the
    (aspect-preserving) resized frame, so stretching it to the thumbnail is exact to within one 28 px token."""
    from matplotlib import colormaps
    from PIL import Image

    g = np.asarray(heat, dtype=np.float32)
    if diverging:
        v = np.clip(g / vmax, -1.0, 1.0)
        v = np.sign(v) * np.abs(v) ** HEAT_GAMMA
        rgba = (colormaps["coolwarm"]((v + 1) / 2) * 255).astype(np.uint8)
        alpha = np.abs(v)
    else:
        lo = float(np.percentile(g, HEAT_FLOOR_PCT))
        v = np.clip((g - lo) / max(vmax - lo, 1e-9), 0.0, 1.0) ** HEAT_GAMMA
        rgba = (colormaps["inferno"](v) * 255).astype(np.uint8)  # (gh, gw, 4)
        alpha = v
    rgba[..., 3] = (alpha * HEAT_ALPHA * 255).astype(np.uint8)
    layer = Image.fromarray(rgba, "RGBA").resize(im.size, Image.BILINEAR)
    return Image.alpha_composite(im.convert("RGBA"), layer).convert("RGB")


def _ablation_effects(s: Session, col: str, t0: int, rec: dict, mode: str) -> dict[str, float] | None:
    """Per camera, the effect of occluding it, in metres of ADE.

    blank: ADE between the prediction with the camera blanked and the arm's own prediction (dependence on
    the camera). real swap (render arms): how much of this arm's gap to the real prediction the swap
    removes, i.e. paired ADE before the swap minus ADE vs real after it (clipped at 0)."""
    abl = rec.get(f"ablate_{mode}") if rec else None
    if not abl:
        return None
    if mode == "blank":
        return {cam: float(v["ade_vs_own_m"]) for cam, v in abl.items()}
    ref = s.columns[s.reference].get(t0)
    if ref is None or ref.get("pred_xyz") is None or rec.get("pred_xyz") is None:
        return None
    before = float(np.linalg.norm(rec["pred_xyz"][0, :, :2] - ref["pred_xyz"][0, :, :2], axis=-1).mean())
    return {cam: max(0.0, before - float(v.get("ade_vs_real_m", before))) for cam, v in abl.items()}


def _ablation_wedges(s: Session, col: str, t0: int, rec: dict, mode: str):
    eff = _ablation_effects(s, col, t0, rec, mode)
    if not eff:
        return None, None
    abl = rec[f"ablate_{mode}"]
    if mode == "blank":
        texts = {cam: f"blanked: prediction moves {eff[cam]:.2f} m (ADE), {abl[cam]['dist_diff_vs_own_m']:+.1f} m distance" for cam in eff}
    else:
        texts = {cam: f"swapped to real: gap to real {abl[cam].get('ade_vs_real_m', float('nan')):.2f} m (removes {eff[cam]:.2f} m)" for cam in eff}
    return eff, texts


def _ablation_row(s: Session, col: str, t0: int, rec: dict, mode: str):
    eff = _ablation_effects(s, col, t0, rec, mode) if rec else None
    label = "occlusion: ADE shift when camera blanked (m)" if mode == "blank" else "occlusion: gap to real removed by swapping camera (m)"
    if not eff:
        return (label, "not recorded for this record" if (rec and rec.get("pred_xyz") is not None) else "-")
    top = sorted(eff.items(), key=lambda kv: -kv[1])
    return (label, " · ".join(f"{CAMERA_LABELS.get(c, c)} {v:.2f}" for c, v in top))


def bev_figure(s: Session, col: str, index: int, stretch_lateral: bool, shown_cameras: list[str], shares: dict[str, float] | None = None,
               share_text: dict[str, str] | None = None):
    """Bird's-eye plot with a FIXED trace structure (same traces, order and names on every instant), so
    Plotly.react updates the data in place rather than clearing and rebuilding the plot."""
    import plotly.graph_objects as go

    t0 = s.t0s[index]
    rec = s.columns[col].get(t0)
    ref = s.columns[s.reference].get(t0)
    fig = go.Figure()
    shown = set(shown_cameras)
    legend_done: set[str] = set()
    for f in s.frustums:  # constant geometry; only the colour of the shown camera changes
        a = np.linspace(f["yaw"] - f["fov"] / 2, f["yaw"] + f["fov"] / 2, 14)
        xs = [f["y"]] + [f["y"] + f["length"] * math.sin(v) for v in a] + [f["y"]]
        ys = [f["x"]] + [f["x"] + f["length"] * math.cos(v) for v in a] + [f["x"]]
        label = CAMERA_LABELS.get(f["name"], f["name"])
        hover = f"{label}{'' if f['used'] else ' (not used)'}"
        if f["name"] in shown:
            kind, line, fill = "camera shown above", dict(color="#d4a017", width=1.5), "rgba(241,196,15,0.45)"
        elif f["used"]:
            kind, line, fill = "other model-input cameras", dict(color="#1f77b4", width=1), "rgba(31,119,180,0.15)"
        else:
            kind, line, fill = "camera not used (rear tele)", dict(color="#999999", width=1, dash="dot"), "rgba(0,0,0,0)"
        if shares is not None and f["used"]:  # attention / occlusion mode: wedge opacity = this camera's value
            share = shares.get(f["name"], 0.0)
            top = max(shares.values()) or 1.0
            fill = f"rgba(255,127,14,{0.08 + 0.8 * max(share, 0.0) / top:.2f})"
            hover += ": " + (share_text[f["name"]] if share_text and f["name"] in share_text else f"{share:.0%} of image attention")
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", line=line, fillcolor=fill, name=kind,
                                 showlegend=kind not in legend_done, legendgroup=kind,
                                 hovertemplate=hover + "<extra></extra>"))
        legend_done.add(kind)
    tv = np.arange(1, NUM_FUTURE_STEPS + 1) * 0.1
    empty = np.zeros((0, 3))
    p_ref = ref["pred_xyz"][0] if (col != s.reference and ref is not None and ref.get("pred_xyz") is not None) else empty
    fig.add_trace(go.Scatter(x=p_ref[:, 1], y=p_ref[:, 0], mode="lines", line=dict(color=s.colors[s.reference], width=2), opacity=0.45,
                             name=f"{s.reference} prediction", showlegend=col != s.reference, customdata=tv[: len(p_ref)],
                             hovertemplate="ref t=+%{customdata:.1f}s<br>x=%{y:.1f} y=%{x:.1f}<extra></extra>"))
    p_own = rec["pred_xyz"][0] if (rec is not None and rec.get("pred_xyz") is not None) else empty
    fig.add_trace(go.Scatter(x=p_own[:, 1], y=p_own[:, 0], mode="lines", line=dict(color=s.colors[col], width=4), name=f"{col} prediction (6.4 s)",
                             customdata=tv[: len(p_own)], hovertemplate=f"{col} t=+%{{customdata:.1f}}s<br>x=%{{y:.1f}} y=%{{x:.1f}}<extra></extra>"))
    gt, truncated = s.partial_gt(t0)
    g = gt if gt is not None else empty
    fig.add_trace(go.Scatter(x=g[:, 1], y=g[:, 0], mode="lines", line=dict(color="black", width=2, dash="dash"), name="recorded future",
                             customdata=tv[: len(g)], hovertemplate="recorded t=+%{customdata:.1f}s<br>x=%{y:.1f} y=%{x:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[g[-1, 1]] if len(g) else [], y=[g[-1, 0]] if len(g) else [], mode="markers",
                             marker=dict(symbol="x", color="black", size=9), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", marker=dict(color="#ff7f0e", size=8), name="ego at t0", hoverinfo="skip"))
    # Fixed ranges and a fixed height, always. Equal aspect is enforced with constrain="domain", which
    # letterboxes the drawing area instead of changing the ranges: with the default (range-changing)
    # behaviour every responsive resize re-derived the ranges and the whole card reflowed on each step.
    lat = 12.0 if stretch_lateral else 25.0
    aspect = {} if stretch_lateral else {"scaleanchor": "x", "scaleratio": 1, "constrain": "domain"}
    fig.update_layout(
        autosize=True, height=BEV_H, margin=dict(l=50, r=10, t=10, b=40), legend=dict(x=0.99, y=0.99, xanchor="right", font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(title="left  <-  lateral y (m)  ->  right" + ("  (stretched)" if stretch_lateral else ""), range=[lat, -lat], zeroline=True, constrain="domain"),
        yaxis=dict(title="forward (m)", range=list(s.forward_range), **aspect),
        paper_bgcolor="white", plot_bgcolor="white", uirevision="keep", transition=dict(duration=0),
    )
    fig.update_xaxes(gridcolor="#eee"); fig.update_yaxes(gridcolor="#eee")
    return fig


def _fmt(v: float) -> str:
    return f"{v:.2f}" if np.isfinite(v) else "-"


def metrics_table(s: Session, col: str, idx: int, attention_split: dict[str, float] | None | bool = False):
    """Current-instant and running-mean ADE (m) at 3 s and 6.4 s, vs the real arm and vs the recorded future.
    ``attention_split`` (attention mode): the expert's attention mass on images / sentence, or None if not recorded."""
    from dash import html

    rows = []
    if attention_split is not False:  # attention / occlusion mode on: always one row, so the card height does not change while stepping
        if isinstance(attention_split, tuple):
            label, txt = attention_split
        else:
            label = "expert attention (late layers)"
            txt = (f"images {attention_split['images']:.0%} · sentence {attention_split['sentence']:.0%} · prompt {attention_split['prompt']:.0%}"
                   if attention_split else "not recorded in this run")
        rows.append(html.Tr([html.Td(label, style={"color": "#a55209", "whiteSpace": "nowrap"}), html.Td(txt, colSpan=2, style={"textAlign": "right", "color": "#a55209", "fontSize": "11px"})]))
    for label, series in (("vs recorded future", s.gt_ade_h), ("vs real", s.paired_h)):
        if label == "vs real" and col == s.reference:
            continue
        for h in ("3 s", "6.4 s"):
            arr = series[h][col]
            so_far = arr[: idx + 1]
            n = int(np.isfinite(so_far).sum())
            rows.append(html.Tr([
                html.Td(f"{label} @ {h}"),
                html.Td(_fmt(arr[idx]), style={"textAlign": "right", "fontWeight": "bold"}),
                html.Td(f"{_fmt(float(np.nanmean(so_far))) if n else '-'}  (n={n})", style={"textAlign": "right"}),
            ]))
    return html.Table(style={"fontSize": "12px", "borderCollapse": "collapse", "width": "100%"}, children=[
        html.Thead(html.Tr([html.Th("ADE (m)", style={"textAlign": "left"}), html.Th("this instant", style={"textAlign": "right"}), html.Th("mean so far", style={"textAlign": "right"})])),
        html.Tbody(rows),
    ])


def card_timeline_figure(s: Session, col: str, index: int, horizon: str = "6.4 s"):
    """Small error-over-time plot for one column (same y scale on every card); click to jump."""
    import plotly.graph_objects as go

    fig = go.Figure()
    hz = s.gt_horizon_s
    paired = s.paired_h.get(horizon, {}).get(col, s.paired[col])
    hlabel = horizon if horizon in s.paired_h else "6.4 s"
    if col == s.reference:
        fig.add_trace(go.Scatter(x=s.times, y=s.gt_ade[col], mode="lines+markers", marker=dict(size=3), line=dict(color="black", width=1.5),
                                 name="vs recorded future", hovertemplate="vs recorded future (6.4 s) %{y:.2f} m<extra></extra>"))
        fig.add_trace(go.Scatter(x=s.times, y=s.gt_ade_partial[col], mode="lines+markers", marker=dict(size=3), line=dict(color="#777777", width=1.5, dash="dot"),
                                 name="vs recorded future, partial horizon", customdata=hz,
                                 hovertemplate="vs recorded future, first %{customdata:.1f} s only: %{y:.2f} m<extra></extra>"))
        title = f"{col} vs recorded future: median {np.nanmedian(s.gt_ade[col]):.2f} m (dotted grey: pose log ends, partial horizon)"
    else:
        fig.add_trace(go.Scatter(x=s.times, y=s.gt_ade[s.reference], mode="lines", line=dict(color="black", width=1, dash="dot"),
                                 name=f"{s.reference} vs recorded future", hovertemplate="ref vs recorded future %{y:.2f} m<extra></extra>"))
        fig.add_trace(go.Scatter(x=s.times, y=s.gt_ade_partial[s.reference], mode="lines", line=dict(color="#aaaaaa", width=1, dash="dot"),
                                 name=f"{s.reference} vs recorded future, partial", customdata=hz,
                                 hovertemplate="ref vs recorded future, first %{customdata:.1f} s only: %{y:.2f} m<extra></extra>"))
        fig.add_trace(go.Scatter(x=s.times, y=paired, mode="lines+markers", marker=dict(size=3), line=dict(color=s.colors[col], width=2),
                                 name=f"vs {s.reference}", hovertemplate=f"vs {s.reference} ({hlabel}) %{{y:.2f}} m<extra></extra>"))
        title = f"{col} vs {s.reference} @ {hlabel}: median {np.nanmedian(paired):.2f} m, p90 {np.nanpercentile(paired, 90):.2f} m"
    fig.add_vline(x=float(s.times[index]), line=dict(color="#d62728", width=2))
    fig.update_layout(height=TL_H, margin=dict(l=40, r=8, t=26, b=30), title=dict(text=title, font=dict(size=11), x=0.02, y=0.97),
                      showlegend=False, xaxis=dict(title=None, gridcolor="#eee"),
                      yaxis=dict(title="ADE (m)", range=[0, s.error_y_max if hlabel == "6.4 s" else s.error_y_max_h.get(hlabel, s.error_y_max)], gridcolor="#eee"),
                      paper_bgcolor="white", plot_bgcolor="white", clickmode="event", hovermode="x unified")
    return fig


BEV_H = 480
TL_H = 190
MAX_SLOTS = 6  # image slots per card (the "all six used" camera set)


def build_app(s: Session):
    import dash
    from dash import ALL, Input, Output, State, ctx, dcc, html
    from flask import Response

    export_names = "+".join(sorted(set(s.export_of.values())))
    app = dash.Dash(__name__, title=f"alpamayo eval: {export_names}")

    # The export and run names are part of the URL so the browser cache can never hand a thumbnail
    # of one scene (or run) to a page showing another.
    img_prefix = f"/img/{export_names}/{'-'.join(sorted(set(s.run_of.values())))}"

    @app.server.route(f"{img_prefix}/<col>/<camera>/<int:index>")
    def _img(col, camera, index):
        from flask import request

        try:
            overlay = request.args.get("overlay", "0") == "1"
            attention = request.args.get("attention", "off")
            band = request.args.get("band") or None
            weighting = request.args.get("w") or None
            scale = request.args.get("scale") or "frame"
            return Response(s.thumbnail(col, camera, index, overlay=overlay, attention=attention, band=band, weighting=weighting, scale=scale), mimetype="image/jpeg", headers={"Cache-Control": "max-age=86400"})
        except (KeyError, FileNotFoundError, IndexError):
            return Response(status=404)

    n = len(s.instants)
    marks = {i: f"{s.times[i]:.0f}s" for i in range(0, n, max(1, n // 12))}
    cols = list(s.columns)
    hidden = {"display": "none"}

    # The cards exist once, for every column, with stable ids; callbacks only change their properties.
    # That keeps images and plots mounted: a new src shows the old picture until the new one has
    # decoded, and a new figure redraws in place, so nothing blinks while stepping.
    def _txt_style(differs: bool) -> dict:
        # Fixed height of three lines, so a long sentence never pushes the plots below it; the box is
        # clipped with an ellipsis (hover shows the whole sentence). 1 px grey or 2 px red border on
        # a 1 px margin, so the box keeps its size either way.
        return {"margin": "2px 8px 4px", "padding": "4px 6px", "fontSize": "12px", "lineHeight": "1.3", "background": "white",
                "height": f"{TXT_LINES * 12 * 1.3 + 2 * 4 + 2 * 2:.0f}px", "border": "2px solid #d62728" if differs else "1px solid #999",
                "boxSizing": "border-box", "overflow": "hidden", "display": "-webkit-box", "WebkitLineClamp": str(TXT_LINES),
                "WebkitBoxOrient": "vertical"}

    def card(col):
        return html.Div(id={"type": "card", "col": col}, style=hidden, children=[
            # One line, clipped with an ellipsis (hover for the full text), so a long header never shifts the card.
            html.Div(f"{col}  ·  {s.run_of[col]}" + (f"  ·  {s.export_of[col]}" if len(set(s.export_of.values())) > 1 else ""),
                     title=f"{col}  ·  {s.run_of[col]}  ·  {s.export_of[col]}",
                     style={"background": "#222", "color": "white", "padding": "3px 6px", "fontSize": "13px", "whiteSpace": "nowrap",
                            "overflow": "hidden", "textOverflow": "ellipsis", "height": "22px", "boxSizing": "border-box"}),
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Img(id={"type": "img", "col": col, "slot": k}, src="", style=hidden) for k in range(MAX_SLOTS)
            ]),
            # Text directly under the image: the eye goes image -> sentence without crossing the plot.
            html.Div(html.B("model reasoning"), title="Chain-of-Causation: the sentence the model generated from these cameras before sampling the trajectory"
                     + (f". Red border: says something materially different from the {s.reference} arm (a changed decision word such as left/right/stop/slow, or less than half the content words in common; punctuation and wording jitter are ignored)." if col != s.reference else ""),
                     style={"margin": "3px 8px 0", "fontSize": "11px", "color": "#444"}),
            html.Div(id={"type": "txt", "col": col}, children="", style=_txt_style(False)),
            dcc.Graph(id={"type": "bev", "col": col}, figure={}, config={"displayModeBar": False, "responsive": True}, style={"width": "100%", "height": f"{BEV_H}px"}),
            html.Div(id={"type": "met", "col": col}, children="", title="ADE = mean point-to-point distance between two 6.4 s trajectories (first 3 s or all 6.4 s); 'mean so far' = average over the instants up to this one", style={"margin": "2px 8px 6px", "color": "#333"}),
            dcc.Graph(id={"type": "tl", "col": col}, figure={}, config={"displayModeBar": False, "responsive": True}, style={"width": "100%", "height": f"{TL_H}px"}),
            # hidden prefetch of the next instant's images, so stepping forward is instant
            html.Div(style=hidden, children=[html.Img(id={"type": "pre", "col": col, "slot": k}, src="") for k in range(MAX_SLOTS)]),
        ])

    import time as _time

    # Identifies this server process and its scene. The page polls it (script below) and reloads itself
    # when it changes, so a tab left open across a restart never talks to a server it does not match.
    page_token = f"{export_names}|{'-'.join(sorted(set(s.run_of.values())))}|{int(_time.time())}"

    @app.server.after_request
    def _no_store(resp):  # the page, the layout and the callback registry are never served from the browser cache
        from flask import request

        if request.path in ("/", "/_dash-layout", "/_dash-dependencies") or request.path.startswith("/_dash-update-component"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    @app.server.route("/page-token")
    def _page_token():
        return Response(page_token, mimetype="text/plain", headers={"Cache-Control": "no-store"})

    app.index_string = """<!DOCTYPE html>
<html><head>{%metas%}<title>{%title%}</title>{%favicon%}{%css%}</head>
<body>{%app_entry%}<footer>{%config%}{%scripts%}{%renderer%}</footer>
<script>
(function () {
  var mine = """ + repr(page_token) + """;
  var wasDown = false;
  setInterval(function () {
    fetch('/page-token', {cache: 'no-store'}).then(function (r) { return r.text(); }).then(function (t) {
      if (t !== mine || wasDown) { location.reload(); }
    }).catch(function () { wasDown = true; });
  }, 3000);
})();
</script></body></html>"""
    app.layout = html.Div(style={"fontFamily": "sans-serif", "margin": "12px"}, children=[
        dcc.Store(id="page-token", data=page_token),
        html.Div(id="stale", children="", style={"display": "none"}),
        html.H3(f"{s.manifest['scene']}  ·  export {export_names.replace('+', ', ')}  ·  runs {', '.join(sorted(set(s.run_of.values())))}", style={"margin": "4px 0"}),
        html.Div(style={"display": "flex", "gap": "18px", "alignItems": "center", "flexWrap": "wrap", "margin": "8px 0"}, children=[
            html.Button("▶ play", id="play", n_clicks=0, style={"width": "90px"}),
            html.Button("◀ step", id="back", n_clicks=0), html.Button("step ▶", id="fwd", n_clicks=0),
            html.Span("speed"), dcc.Dropdown(id="speed", options=[{"label": f"{v} instants/s", "value": v} for v in (1, 2, 4, 10)], value=2, clearable=False, style={"width": "150px"}),
            html.Span("cameras"), dcc.Dropdown(id="cameras", options=[{"label": k, "value": k} for k in CAMERA_SETS], value="front wide", clearable=False, style={"width": "170px"}),
            html.Span("arms"), dcc.Checklist(id="cols", options=[{"label": c, "value": c} for c in cols], value=cols[:4], inline=True),
            dcc.Checklist(id="aspect", options=[{"label": "stretch lateral axis", "value": "stretch"}], value=[], inline=True),
            dcc.Checklist(id="overlay", options=[{"label": "paths on images", "value": "on"}], value=["on"], inline=True),
            # ONE overlay control. "model attention" is fixed to the combination that reads well: attention x value
            # norm (sinks removed), middle layers, each frame scaled to its own peak above its floor. Runs made
            # before the value weighting existed fall back to plain attention and say so in the label.
            html.Span("overlay", title=ATTENTION_HELP),
            dcc.Dropdown(id="attention", options=[{"label": "off", "value": "off"}]
                         + ([{"label": "model attention" + ("" if s.has_vw else " (plain, older run: weaker)"), "value": "raw"}] if s.has_attention else [])
                         + [{"label": m, "value": m} for m in ABLATION_MODES if ATTENTION_MODES[m][1][7:] in s.has_ablation],
                         value="off", clearable=False, disabled=not (s.has_attention or s.has_ablation), style={"width": "230px"}),
            html.Span("ADE horizon"), dcc.Dropdown(id="horizon", options=[{"label": h, "value": h} for h in HORIZONS], value="6.4 s", clearable=False, style={"width": "170px"}),
            html.Span(id="where", style={"fontWeight": "bold"}),
        ]),
        dcc.Slider(id="idx", min=0, max=n - 1, step=1, value=0, marks=marks, tooltip={"placement": "bottom", "always_visible": False}, updatemode="drag"),
        dcc.Interval(id="tick", interval=500, disabled=True),
        html.Div(id="cards", style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "8px", "marginTop": "10px"},
                 children=[card(col) for col in cols]),
    ])

    @app.callback(Output("tick", "disabled"), Output("play", "children"), Output("tick", "interval"),
                  Input("play", "n_clicks"), Input("speed", "value"), State("tick", "disabled"))
    def _play(n_clicks, speed, disabled):
        playing = disabled if ctx.triggered_id == "play" else (not disabled)  # the button toggles; a speed change keeps the state
        return (not playing), ("⏸ pause" if playing else "▶ play"), int(1000 / speed)

    @app.callback(Output("idx", "value"), Input("tick", "n_intervals"), Input("back", "n_clicks"), Input("fwd", "n_clicks"),
                  Input({"type": "tl", "col": ALL}, "clickData"), State("idx", "value"), prevent_initial_call=True)
    def _move(_t, _b, _f, clicks, idx):
        trig = ctx.triggered_id
        if trig == "tick" or trig == "fwd":
            return min(idx + 1, n - 1) if trig == "fwd" else (idx + 1) % n
        if trig == "back":
            return max(idx - 1, 0)
        if isinstance(trig, dict) and trig.get("type") == "tl":
            click = ctx.triggered[0]["value"]
            if click and click.get("points"):
                return int(np.argmin(np.abs(s.times - click["points"][0]["x"])))
        return idx

    def _view(idx, selected, camera_set, aspect, overlay, attention, horizon, token):
        if token != page_token:  # a tab left open across a server restart would otherwise show mixed scenes
            n_out = len(cols) * (1 + 3 * MAX_SLOTS + 5)
            banner = "THIS TAB IS STALE: the server now serves a different scene or run. Reload the page (Ctrl-Shift-R)."
            return ([dash.no_update] * len(cols), [dash.no_update] * (len(cols) * MAX_SLOTS), [dash.no_update] * (len(cols) * MAX_SLOTS),
                    [dash.no_update] * (len(cols) * MAX_SLOTS), [dash.no_update] * len(cols), [dash.no_update] * len(cols), [dash.no_update] * len(cols),
                    [dash.no_update] * len(cols), [dash.no_update] * len(cols), dash.no_update, banner,
                    {"display": "block", "background": "#d62728", "color": "white", "padding": "10px", "fontWeight": "bold", "margin": "8px 0"}, banner)
        t0 = s.t0s[idx]
        selected = set(selected or [])
        cameras = CAMERA_SETS[camera_set]
        ncol = 3 if len(cameras) >= 3 else len(cameras)
        img_style = {"width": f"{100 / ncol:.3f}%", "display": "block"}
        attention = attention if attention in ATTENTION_MODES else "off"
        band = "mid layers (16-47)" if s.has_layers else ""
        weighting = "attention x value norm" if s.has_vw else ""
        q = f"?overlay={1 if overlay else 0}&attention={attention}&band={band}&w={weighting}&scale=frame"
        attention_on = attention != "off"
        ablation = ATTENTION_MODES[attention][1][7:] if attention in ABLATION_MODES else None  # "blank" | "real" | None
        ref = s.columns[s.reference].get(t0)
        nxt = min(idx + 1, n - 1)
        card_styles, srcs, img_styles, pre_srcs, bevs, txts, txt_styles, mets, tls = [], [], [], [], [], [], [], [], []
        for col in cols:  # output lists follow the layout order: columns in order, slots 0..MAX_SLOTS-1
            on = col in selected
            rec = s.columns[col].get(t0)
            card_styles.append({"minWidth": 0, "border": "1px solid #ddd", "background": "#fafafa"} if on else hidden)
            for k in range(MAX_SLOTS):
                show = on and k < len(cameras) and rec is not None
                srcs.append(f"{img_prefix}/{col}/{cameras[k]}/{idx}{q}" if show else dash.no_update)
                img_styles.append(img_style if show else hidden)
                pre_srcs.append(f"{img_prefix}/{col}/{cameras[k]}/{nxt}{q}" if show else dash.no_update)
            if not on:
                bevs.append(dash.no_update); txts.append(dash.no_update); txt_styles.append(dash.no_update); mets.append(dash.no_update); tls.append(dash.no_update)
                continue
            if ablation and rec:
                shares, texts = _ablation_wedges(s, col, t0, rec, ablation)
                fig = bev_figure(s, col, idx, bool(aspect), cameras, shares=shares, share_text=texts)
            else:
                fig = bev_figure(s, col, idx, bool(aspect), cameras, shares=camera_shares(rec) if (attention_on and rec) else None)
            fig.update_layout(uirevision="keep")
            bevs.append(fig)
            text = _cot_text(rec) if rec else ""
            differs = bool(rec and ref and col != s.reference and cot_differs(_cot_text(rec), _cot_text(ref)))
            shown = text or ("(no output)" if rec else "not available for this instant")
            txts.append(html.Span(shown, title=shown))  # full sentence on hover when the box clips it
            txt_styles.append(_txt_style(differs))
            if ablation:
                mets.append(metrics_table(s, col, idx, _ablation_row(s, col, t0, rec, ablation)))
            else:
                mets.append(metrics_table(s, col, idx, (text_share(rec) if rec else None) if attention_on else False))
            tl = card_timeline_figure(s, col, idx, horizon)
            tl.update_layout(uirevision="keep")
            tls.append(tl)
        grid = {"display": "grid", "gridTemplateColumns": f"repeat({max(1, len(selected))}, minmax(0, 1fr))", "gap": "8px", "marginTop": "10px"}
        where = f"instant {idx + 1}/{n}   t = +{s.times[idx]:.1f} s"
        return card_styles, srcs, img_styles, pre_srcs, bevs, txts, txt_styles, mets, tls, grid, where, {"display": "none"}, ""

    app.callback(
        Output({"type": "card", "col": ALL}, "style"),
        Output({"type": "img", "col": ALL, "slot": ALL}, "src"),
        Output({"type": "img", "col": ALL, "slot": ALL}, "style"),
        Output({"type": "pre", "col": ALL, "slot": ALL}, "src"),
        Output({"type": "bev", "col": ALL}, "figure"),
        Output({"type": "txt", "col": ALL}, "children"),
        Output({"type": "txt", "col": ALL}, "style"),
        Output({"type": "met", "col": ALL}, "children"),
        Output({"type": "tl", "col": ALL}, "figure"),
        Output("cards", "style"),
        Output("where", "children"),
        Output("stale", "style"),
        Output("stale", "children"),
        Input("idx", "value"), Input("cols", "value"), Input("cameras", "value"), Input("aspect", "value"), Input("overlay", "value"),
        Input("attention", "value"), Input("horizon", "value"), State("page-token", "data"),
    )(_view)
    app.view_fn = _view  # the raw function, for tests

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="Run locations (s3:// or local); the first is the reference. Runs on another export of the same scene are tagged by export name.")
    parser.add_argument("--scene", default=None, help="Instead of run locations: a scene name from registry.py (italy, madrid, austria) = that scene's reference run, v2 run and seed floor.")
    parser.add_argument("--labels", nargs="+", default=None, help="One tag per run instead of the automatic seed / export tags (empty string = no tag), e.g. --labels '' v2; '+' merges a run with identical predictions (e.g. an occlusion run) into the existing columns")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cache", type=Path, default=Path("~/.cache/alpamayo-eval").expanduser())
    args = parser.parse_args()
    import socket

    with socket.socket() as sock:  # fail fast, before the minute of loading, with a clear message
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # like Flask: TIME_WAIT leftovers are not "in use"
        try:
            sock.bind((args.host, args.port))
        except OSError:
            raise SystemExit(
                f"port {args.port} on {args.host} is already in use, probably another dashboard still running "
                f"(the browser would keep showing THAT scene). Stop it with Ctrl-C in its terminal, or pass --port {args.port + 1}."
            )
    if args.scene:
        from pd_alpamayo_eval.registry import scene_runs

        args.runs, args.labels = scene_runs(args.scene)
    if not args.runs:
        raise SystemExit("give run locations or --scene <name>")
    session = Session(args.runs, args.cache, args.labels)
    print(f"loaded {len(session.instants)} instants, columns {list(session.columns)}; open http://{args.host}:{args.port}")
    build_app(session).run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

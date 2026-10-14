"""Interactive dashboard for inspecting runs: instant slider, arms side by side, plotly plots.

    pixi run -e dev dashboard s3://pd-nebius-ray/alpamayo-eval/full_variations/runs/alpamayo-run-27f93c3e \\
        [s3://pd-nebius-ray/alpamayo-eval/full_variations/runs/alpamayo-run-af7ff7bc ...] [--port 8050]

then open http://localhost:8050. Runs are staged into ``~/.cache/alpamayo-eval`` (same cache as
make-video); images are read lazily per instant and thumbnailed on first use. Every run must be on
the same export. The first run's ``real`` arm is the reference; arms of further runs appear as
``<arm>@seed<N>`` columns (e.g. ``real@seed1`` is the seed-to-seed noise floor).

What you get: a slider over instants with play / pause / step and speed (click the slider handle,
then the arrow keys step too); a card per arm with the camera frame(s) at t0, a bird's-eye plot
(history, recorded future as far as the pose log goes, the arm's prediction, the reference
prediction, the ego footprint and a camera-FOV inset), the Chain-of-Causation text flagged when it
differs from the reference, and the metrics of that instant; an error-over-time chart (paired ADE
of every column against the reference, dotted: reference against the recorded future) that you
can click to jump; and the summary table of compare-runs.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pd_alpamayo_eval.export_common import MANIFEST_NAME, NUM_FUTURE_STEPS, STEP_NS, read_manifest  # noqa: E402
from pd_alpamayo_eval.load_export import EgoTrajectory  # noqa: E402
from pd_alpamayo_eval.make_video import (  # noqa: E402
    ARM_COLORS,
    CAMERA_LABELS,
    EXTRA_COLORS,
    FRUSTUM_LENGTH_M,
    _camera_frustums,
    _cot_text,
    _load_run,
    _stage,
)

THUMB_W = 960
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

    def __init__(self, run_uris: list[str], cache: Path):
        self.cache = cache
        self.columns: dict[str, dict[int, dict]] = {}
        self.arm_of: dict[str, str] = {}
        self.run_of: dict[str, str] = {}
        export_uri = None
        for i, uri in enumerate(run_uris):
            run_dir = _stage(uri, cache)
            summary, records = _load_run(run_dir)
            if export_uri is None:
                export_uri = summary.get("export")
            elif summary.get("export") != export_uri:
                raise SystemExit(f"{uri} is on export {summary.get('export')}, expected {export_uri}")
            tag = "" if i == 0 else f"@seed{summary.get('seed')}"
            for (t0, arm), rec in records.items():
                col = f"{arm}{tag}"
                self.columns.setdefault(col, {})[int(t0)] = rec
                self.arm_of[col] = arm
                self.run_of[col] = summary.get("run_id", run_dir.name)
        if export_uri is None:
            raise SystemExit("no export recorded in the run summaries")
        self.export_dir = _stage(export_uri, cache, skip_suffixes=())
        if not (self.export_dir / MANIFEST_NAME).is_file():
            raise SystemExit(f"{self.export_dir} has no {MANIFEST_NAME}")
        self.manifest = read_manifest(self.export_dir)
        self.instants = self.manifest["instants"]
        self.t0s = [int(inst["t0_ns"]) for inst in self.instants]
        self.times = np.array([(t - self.t0s[0]) / 1e9 for t in self.t0s])
        self.trajectory = EgoTrajectory(self.manifest)
        self.frustums = _camera_frustums(self.manifest)
        self.reference = "real" if "real" in self.columns else next(iter(self.columns))
        order = [self.reference] + [c for c in self.columns if c != self.reference]
        self.columns = {c: self.columns[c] for c in order}
        self.colors = {}
        extra = iter(EXTRA_COLORS)
        for col in self.columns:
            self.colors[col] = ARM_COLORS.get(col, next(extra, "#7f7f7f"))
        self._series()
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
        for col, recs in self.columns.items():
            paired, gt, gtp, end, diff = [], [], [], [], []
            for t0 in self.t0s:
                rec, r = recs.get(t0), ref.get(t0)
                ok = rec is not None and rec.get("pred_xyz") is not None
                okr = ok and r is not None and r.get("pred_xyz") is not None
                paired.append(float(np.linalg.norm(rec["pred_xyz"][0, :, :2] - r["pred_xyz"][0, :, :2], axis=-1).mean()) if okr else np.nan)
                gt.append(float(rec["min_ade_vs_gt_m"]) if ok and rec.get("min_ade_vs_gt_m") is not None else np.nan)
                g = partial_gts[t0]
                gtp.append(float(np.linalg.norm(rec["pred_xyz"][0, : len(g), :2] - g[:, :2], axis=-1).mean()) if (ok and g is not None and rec.get("min_ade_vs_gt_m") is None) else np.nan)
                end.append(float(np.linalg.norm(rec["pred_xyz"][0, -1, :2])) if ok else np.nan)
                diff.append(float(_cot_text(rec) != _cot_text(r)) if (rec and r) else np.nan)
            self.paired[col] = np.array(paired)
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

    def summary_rows(self) -> list[dict]:
        rows = []
        ref_end = self.end_dist[self.reference]
        for col in self.columns:
            if col == self.reference:
                rows.append({"column": f"{col} (reference)", "run": self.run_of[col], "n": int(np.isfinite(self.end_dist[col]).sum()),
                             "paired ADE median / p90 (m)": "-", "pred. distance diff median (m)": "-", "longer / shorter": "-",
                             "ADE vs GT median (m)": f"{np.nanmedian(self.gt_ade[col]):.2f}", "text differs": "-"})
                continue
            p = self.paired[col]; d = self.end_dist[col] - ref_end
            rows.append({
                "column": col, "run": self.run_of[col], "n": int(np.isfinite(p).sum()),
                "paired ADE median / p90 (m)": f"{np.nanmedian(p):.2f} / {np.nanpercentile(p, 90):.2f}",
                "pred. distance diff median (m)": f"{np.nanmedian(d):+.2f}",
                "longer / shorter": f"{np.nanmean(d > 1):.0%} / {np.nanmean(d < -1):.0%}",
                "ADE vs GT median (m)": f"{np.nanmedian(self.gt_ade[col]):.2f}",
                "text differs": f"{np.nanmean(self.text_differs[col]):.0%}",
            })
        return rows

    def thumbnail(self, col: str, camera: str, index: int, overlay: bool = False) -> bytes:
        from PIL import Image

        arm = self.arm_of[col]
        inst = self.instants[index]
        t_ns = int(inst["frames"][camera][-1])
        src = self.export_dir / arm / camera / f"{t_ns}.jpg"
        thumb = self.cache / "thumbs" / self.export_name / arm / camera / f"{t_ns}.jpg"
        if not thumb.is_file():
            thumb.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((THUMB_W, THUMB_W))
                im.save(thumb, quality=88)
        projector = self.projectors.get(camera)
        if not overlay or projector is None:
            return thumb.read_bytes()
        # Paths at t0: history, recorded future, the reference prediction (on non-reference arms), this arm's prediction.
        t0 = self.t0s[index]
        rec = self.columns[col].get(t0)
        ref = self.columns[self.reference].get(t0)
        paths = [(self.history(t0), OVERLAY_COLORS["history"], 4, False)]
        if col != self.reference and ref is not None and ref.get("pred_xyz") is not None:
            paths.append((ref["pred_xyz"][0], _hex_rgb(self.colors[self.reference]), 3, False))
        if rec is not None and rec.get("pred_xyz") is not None:
            paths.append((rec["pred_xyz"][0], _hex_rgb(self.colors[col]), 5, False))
        gt, _ = self.partial_gt(t0)
        if gt is not None:  # last, thin and dashed, so it stays visible on top of the predictions
            paths.append((gt, OVERLAY_COLORS["gt"], 2, True))
        with Image.open(thumb) as im:
            im = im.convert("RGB")
            scale = im.width / projector.width
            draw_paths(im, projector, paths, scale)
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=88)
        return buf.getvalue()


def bev_figure(s: Session, col: str, index: int, stretch_lateral: bool, shown_cameras: list[str]):
    """Bird's-eye plot with a FIXED trace structure (same traces, order and names on every instant), so
    Plotly.react updates the data in place rather than clearing and rebuilding the plot."""
    import plotly.graph_objects as go

    t0 = s.t0s[index]
    rec = s.columns[col].get(t0)
    ref = s.columns[s.reference].get(t0)
    hist = s.history(t0)
    fig = go.Figure()
    shown = set(shown_cameras)
    legend_done: set[str] = set()
    for f in s.frustums:  # constant geometry; only the colour of the shown camera changes
        a = np.linspace(f["yaw"] - f["fov"] / 2, f["yaw"] + f["fov"] / 2, 14)
        xs = [f["y"]] + [f["y"] + f["length"] * math.sin(v) for v in a] + [f["y"]]
        ys = [f["x"]] + [f["x"] + f["length"] * math.cos(v) for v in a] + [f["x"]]
        label = CAMERA_LABELS.get(f["name"], f["name"])
        if f["name"] in shown:
            kind, line, fill = "camera shown above", dict(color="#d4a017", width=1.5), "rgba(241,196,15,0.45)"
        elif f["used"]:
            kind, line, fill = "other model-input cameras", dict(color="#1f77b4", width=1), "rgba(31,119,180,0.15)"
        else:
            kind, line, fill = "camera not used (rear tele)", dict(color="#999999", width=1, dash="dot"), "rgba(0,0,0,0)"
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself", line=line, fillcolor=fill, name=kind,
                                 showlegend=kind not in legend_done, legendgroup=kind,
                                 hovertemplate=f"{label}{'' if f['used'] else ' (not used)'}<extra></extra>"))
        legend_done.add(kind)
    tv = np.arange(1, NUM_FUTURE_STEPS + 1) * 0.1
    empty = np.zeros((0, 3))
    fig.add_trace(go.Scatter(x=hist[:, 1], y=hist[:, 0], mode="lines", line=dict(color="#1f77b4", width=4), name="ego history (1.5 s)",
                             hovertemplate="history t=%{customdata:.1f}s<br>x=%{y:.1f} y=%{x:.1f}<extra></extra>", customdata=np.arange(-15, 1) * 0.1))
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


def card_timeline_figure(s: Session, col: str, index: int):
    """Small error-over-time plot for one column (same y scale on every card); click to jump."""
    import plotly.graph_objects as go

    fig = go.Figure()
    hz = s.gt_horizon_s
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
        fig.add_trace(go.Scatter(x=s.times, y=s.paired[col], mode="lines+markers", marker=dict(size=3), line=dict(color=s.colors[col], width=2),
                                 name=f"vs {s.reference}", hovertemplate=f"vs {s.reference} %{{y:.2f}} m<extra></extra>"))
        title = f"{col} vs {s.reference}: median {np.nanmedian(s.paired[col]):.2f} m, p90 {np.nanpercentile(s.paired[col], 90):.2f} m"
    fig.add_vline(x=float(s.times[index]), line=dict(color="#d62728", width=2))
    fig.update_layout(height=TL_H, margin=dict(l=40, r=8, t=26, b=30), title=dict(text=title, font=dict(size=11), x=0.02, y=0.97),
                      showlegend=False, xaxis=dict(title=None, gridcolor="#eee"), yaxis=dict(title="ADE (m)", range=[0, s.error_y_max], gridcolor="#eee"),
                      paper_bgcolor="white", plot_bgcolor="white", clickmode="event", hovermode="x unified")
    return fig


BEV_H = 480
TL_H = 190
MAX_SLOTS = 6  # image slots per card (the "all six used" camera set)


def build_app(s: Session):
    import dash
    from dash import ALL, Input, Output, State, ctx, dash_table, dcc, html
    from flask import Response

    app = dash.Dash(__name__, title=f"alpamayo eval: {s.export_name}")

    # The export and run names are part of the URL so the browser cache can never hand a thumbnail
    # of one scene (or run) to a page showing another.
    img_prefix = f"/img/{s.export_name}/{'-'.join(sorted(set(s.run_of.values())))}"

    @app.server.route(f"{img_prefix}/<col>/<camera>/<int:index>")
    def _img(col, camera, index):
        from flask import request

        try:
            overlay = request.args.get("overlay", "0") == "1"
            return Response(s.thumbnail(col, camera, index, overlay=overlay), mimetype="image/jpeg", headers={"Cache-Control": "max-age=86400"})
        except (KeyError, FileNotFoundError, IndexError):
            return Response(status=404)

    n = len(s.instants)
    marks = {i: f"{s.times[i]:.0f}s" for i in range(0, n, max(1, n // 12))}
    cols = list(s.columns)
    hidden = {"display": "none"}

    # The cards exist once, for every column, with stable ids; callbacks only change their properties.
    # That keeps images and plots mounted: a new src shows the old picture until the new one has
    # decoded, and a new figure redraws in place, so nothing blinks while stepping.
    def swatch(color, label):
        return html.Span(style={"display": "inline-flex", "alignItems": "center", "gap": "4px"}, children=[
            html.Span(style={"display": "inline-block", "width": "18px", "height": "4px", "background": color}), label])

    def card(col):
        return html.Div(id={"type": "card", "col": col}, style=hidden, children=[
            html.Div(f"{col}  ·  {s.run_of[col]}", style={"background": "#222", "color": "white", "padding": "3px 6px", "fontSize": "13px"}),
            html.Div(style={"display": "flex", "flexWrap": "wrap"}, children=[
                html.Img(id={"type": "img", "col": col, "slot": k}, src="", style=hidden) for k in range(MAX_SLOTS)
            ]),
            html.Div(style={"fontSize": "11px", "color": "#444", "padding": "2px 8px", "display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
                html.Span("on the image:"),
                swatch("#1f77b4", "ego history 1.5 s"),
                *([swatch(s.colors[s.reference], f"{s.reference} prediction")] if col != s.reference else []),
                swatch(s.colors[col], f"{col} prediction 6.4 s"),
                swatch("#000000", "recorded future (dashed)"),
            ]),
            dcc.Graph(id={"type": "bev", "col": col}, figure={}, config={"displayModeBar": False, "responsive": True}, style={"width": "100%", "height": f"{BEV_H}px"}),
            html.Div(style={"margin": "6px 8px 0", "fontSize": "12px", "color": "#444"}, children=[
                html.B("Model reasoning (Chain-of-Causation)"),
                html.Span(" - the sentence the model generated from these cameras before sampling the trajectory"
                          + (f"; red border = differs from the {s.reference} arm" if col != s.reference else "")),
            ]),
            html.Div(id={"type": "txt", "col": col}, children="", style={"margin": "4px 8px", "padding": "6px", "background": "white", "border": "1px solid #999", "minHeight": "40px"}),
            html.Div(id={"type": "met", "col": col}, children="", title="ADE = mean distance between two trajectories over the 6.4 s horizon; predicted distance = length of the predicted path", style={"margin": "2px 8px 6px", "fontSize": "13px", "color": "#333"}),
            dcc.Graph(id={"type": "tl", "col": col}, figure={}, config={"displayModeBar": False, "responsive": True}, style={"width": "100%", "height": f"{TL_H}px"}),
            # hidden prefetch of the next instant's images, so stepping forward is instant
            html.Div(style=hidden, children=[html.Img(id={"type": "pre", "col": col, "slot": k}, src="") for k in range(MAX_SLOTS)]),
        ])

    import time as _time

    # Identifies this server process and its scene. The page polls it (script below) and reloads itself
    # when it changes, so a tab left open across a restart never talks to a server it does not match.
    page_token = f"{s.export_name}|{'-'.join(sorted(set(s.run_of.values())))}|{int(_time.time())}"

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
        html.H3(f"{s.manifest['scene']}  ·  export {s.export_name}  ·  runs {', '.join(sorted(set(s.run_of.values())))}", style={"margin": "4px 0"}),
        html.Div(style={"display": "flex", "gap": "18px", "alignItems": "center", "flexWrap": "wrap", "margin": "8px 0"}, children=[
            html.Button("▶ play", id="play", n_clicks=0, style={"width": "90px"}),
            html.Button("◀ step", id="back", n_clicks=0), html.Button("step ▶", id="fwd", n_clicks=0),
            html.Span("speed"), dcc.Dropdown(id="speed", options=[{"label": f"{v} instants/s", "value": v} for v in (1, 2, 4, 10)], value=2, clearable=False, style={"width": "150px"}),
            html.Span("cameras"), dcc.Dropdown(id="cameras", options=[{"label": k, "value": k} for k in CAMERA_SETS], value="front wide", clearable=False, style={"width": "170px"}),
            html.Span("arms"), dcc.Checklist(id="cols", options=[{"label": c, "value": c} for c in cols], value=cols[:4], inline=True),
            dcc.Checklist(id="aspect", options=[{"label": "stretch lateral axis", "value": "stretch"}], value=[], inline=True),
            dcc.Checklist(id="overlay", options=[{"label": "paths on images", "value": "on"}], value=["on"], inline=True),
            html.Span(id="where", style={"fontWeight": "bold"}),
        ]),
        dcc.Slider(id="idx", min=0, max=n - 1, step=1, value=0, marks=marks, tooltip={"placement": "bottom", "always_visible": False}, updatemode="drag"),
        dcc.Interval(id="tick", interval=500, disabled=True),
        html.Div(id="cards", style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(0, 1fr))", "gap": "8px", "marginTop": "10px"},
                 children=[card(col) for col in cols]),
        html.Details(style={"marginTop": "14px"}, open=False, children=[
            html.Summary("summary over the scene (compare-runs)", style={"cursor": "pointer", "fontWeight": "bold"}),
            dash_table.DataTable(data=s.summary_rows(), columns=[{"name": k, "id": k} for k in s.summary_rows()[0]], style_cell={"fontSize": 13, "padding": "4px"}, style_header={"fontWeight": "bold"}),
        ]),
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

    def _view(idx, selected, camera_set, aspect, overlay, token):
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
        q = "?overlay=1" if overlay else ""
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
            fig = bev_figure(s, col, idx, bool(aspect), cameras)
            fig.update_layout(uirevision="keep")
            bevs.append(fig)
            text = _cot_text(rec) if rec else ""
            differs = bool(rec and ref and col != s.reference and _cot_text(rec) != _cot_text(ref))
            txts.append(text or ("(no output)" if rec else "not available for this instant"))
            txt_styles.append({"margin": "4px 8px", "padding": "6px", "background": "white", "minHeight": "40px",
                               "border": "3px solid #d62728" if differs else "1px solid #999"})
            metrics = []
            if rec and rec.get("min_ade_vs_gt_m") is not None:
                metrics.append(f"ADE vs recorded future {rec['min_ade_vs_gt_m']:.2f} m")
            elif np.isfinite(s.gt_ade_partial[col][idx]):
                metrics.append(f"ADE vs recorded future, first {s.gt_horizon_s[idx]:.1f} s only: {s.gt_ade_partial[col][idx]:.2f} m")
            elif rec:
                metrics.append("no recorded future left (pose log ends)")
            if col != s.reference and np.isfinite(s.paired[col][idx]):
                metrics.append(f"vs {s.reference} {s.paired[col][idx]:.2f} m")
            if np.isfinite(s.end_dist[col][idx]):
                metrics.append(f"predicted distance {s.end_dist[col][idx]:.1f} m")
            mets.append("   ·   ".join(metrics))
            tl = card_timeline_figure(s, col, idx)
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
        State("page-token", "data"),
    )(_view)
    app.view_fn = _view  # the raw function, for tests

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run locations (s3:// or local), all on the same export; the first is the reference.")
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
    session = Session(args.runs, args.cache)
    print(f"loaded {len(session.instants)} instants, columns {list(session.columns)}; open http://{args.host}:{args.port}")
    build_app(session).run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

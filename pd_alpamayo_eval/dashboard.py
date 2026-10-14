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

THUMB_W = 640
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

    def _series(self) -> None:
        ref = self.columns[self.reference]
        self.paired: dict[str, np.ndarray] = {}
        self.gt_ade: dict[str, np.ndarray] = {}
        self.end_dist: dict[str, np.ndarray] = {}
        self.text_differs: dict[str, np.ndarray] = {}
        for col, recs in self.columns.items():
            paired, gt, end, diff = [], [], [], []
            for t0 in self.t0s:
                rec, r = recs.get(t0), ref.get(t0)
                ok = rec is not None and rec.get("pred_xyz") is not None
                okr = ok and r is not None and r.get("pred_xyz") is not None
                paired.append(float(np.linalg.norm(rec["pred_xyz"][0, :, :2] - r["pred_xyz"][0, :, :2], axis=-1).mean()) if okr else np.nan)
                gt.append(float(rec["min_ade_vs_gt_m"]) if ok and rec.get("min_ade_vs_gt_m") is not None else np.nan)
                end.append(float(np.linalg.norm(rec["pred_xyz"][0, -1, :2])) if ok else np.nan)
                diff.append(float(_cot_text(rec) != _cot_text(r)) if (rec and r) else np.nan)
            self.paired[col] = np.array(paired)
            self.gt_ade[col] = np.array(gt)
            self.end_dist[col] = np.array(end)
            self.text_differs[col] = np.array(diff)
        x_all = [0.0, 15.0]
        for recs in self.columns.values():
            for rec in recs.values():
                for arr in (rec.get("pred_xyz"), rec.get("gt_xyz")):
                    if arr is not None:
                        x_all += [float(np.asarray(arr).reshape(-1, 3)[:, 0].max()), float(np.asarray(arr).reshape(-1, 3)[:, 0].min())]
        self.forward_range = (min(x_all) - 3, max(x_all) + 3)

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

    def thumbnail(self, col: str, camera: str, index: int) -> bytes:
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
        return thumb.read_bytes()


def bev_figure(s: Session, col: str, index: int, equal_aspect: bool):
    import plotly.graph_objects as go

    t0 = s.t0s[index]
    rec = s.columns[col].get(t0)
    ref = s.columns[s.reference].get(t0)
    hist = s.history(t0)
    fig = go.Figure()
    # ego footprint
    fig.add_trace(go.Scatter(x=[1, -1, -1, 1, 1], y=[-1, -1, 3.5, 3.5, -1], fill="toself", mode="lines",
                             line=dict(color="#ff7f0e"), fillcolor="rgba(255,127,14,0.4)", name="ego footprint", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist[:, 1], y=hist[:, 0], mode="lines", line=dict(color="#1f77b4", width=4), name="ego history (1.5 s)",
                             hovertemplate="history t=%{customdata:.1f}s<br>x=%{y:.1f} y=%{x:.1f}<extra></extra>", customdata=np.arange(-15, 1) * 0.1))
    tv = np.arange(1, NUM_FUTURE_STEPS + 1) * 0.1
    if col != s.reference and ref is not None and ref.get("pred_xyz") is not None:
        p = ref["pred_xyz"][0]
        fig.add_trace(go.Scatter(x=p[:, 1], y=p[:, 0], mode="lines", line=dict(color=s.colors[s.reference], width=2), opacity=0.45,
                                 name=f"{s.reference} prediction", customdata=tv, hovertemplate="ref t=+%{customdata:.1f}s<br>x=%{y:.1f} y=%{x:.1f}<extra></extra>"))
    if rec is not None and rec.get("pred_xyz") is not None:
        for k, p in enumerate(rec["pred_xyz"]):
            fig.add_trace(go.Scatter(x=p[:, 1], y=p[:, 0], mode="lines", line=dict(color=s.colors[col], width=4), name=f"{col} prediction (6.4 s)",
                                     showlegend=k == 0, customdata=tv, hovertemplate=f"{col} t=+%{{customdata:.1f}}s<br>x=%{{y:.1f}} y=%{{x:.1f}}<extra></extra>"))
    gt, truncated = s.partial_gt(t0)
    if gt is not None:
        label = f"recorded future (ends at +{len(gt) / 10:.1f} s)" if truncated else "recorded future (6.4 s)"
        fig.add_trace(go.Scatter(x=gt[:, 1], y=gt[:, 0], mode="lines", line=dict(color="black", width=2, dash="dash"), name=label,
                                 customdata=tv[: len(gt)], hovertemplate="recorded t=+%{customdata:.1f}s<br>x=%{y:.1f} y=%{x:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[gt[-1, 1]], y=[gt[-1, 0]], mode="markers", marker=dict(symbol="x", color="black", size=9), showlegend=False, hoverinfo="skip"))
    # camera FOV inset (equal aspect, own axes)
    for f in s.frustums:
        a = np.linspace(f["yaw"] - f["fov"] / 2, f["yaw"] + f["fov"] / 2, 12)
        xs = [f["y"]] + [f["y"] + f["length"] * math.sin(v) for v in a] + [f["y"]]
        ys = [f["x"]] + [f["x"] + f["length"] * math.cos(v) for v in a] + [f["x"]]
        fig.add_trace(go.Scatter(x=xs, y=ys, xaxis="x2", yaxis="y2", mode="lines", fill="toself", showlegend=False,
                                 line=dict(color="#1f77b4" if f["used"] else "#999999", width=1, dash=None if f["used"] else "dot"),
                                 fillcolor="rgba(31,119,180,0.14)" if f["used"] else "rgba(0,0,0,0)",
                                 hovertemplate=f"{CAMERA_LABELS.get(f['name'], f['name'])}{'' if f['used'] else ' (not used)'}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1, -1, -1, 1, 1], y=[-1, -1, 3.5, 3.5, -1], xaxis="x2", yaxis="y2", fill="toself", mode="lines",
                             line=dict(color="#ff7f0e"), fillcolor="rgba(255,127,14,0.5)", showlegend=False, hoverinfo="skip"))
    n_used = sum(f["used"] for f in s.frustums)
    lat = 12.0
    fig.update_layout(
        height=430, margin=dict(l=50, r=10, t=10, b=40), legend=dict(x=0.01, y=0.99, font=dict(size=10), bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(title="left  <-  lateral y (m)  ->  right", range=[lat, -lat], zeroline=True),
        yaxis=dict(title="forward (m)", range=list(s.forward_range), **({"scaleanchor": "x", "scaleratio": 1} if equal_aspect else {})),
        xaxis2=dict(domain=[0.70, 0.99], anchor="y2", range=[11, -11], showticklabels=False, showgrid=False, zeroline=False),
        yaxis2=dict(domain=[0.02, 0.34], anchor="x2", range=[-11, 11], scaleanchor="x2", scaleratio=1, showticklabels=False, showgrid=False, zeroline=False),
        annotations=[dict(x=0.845, y=0.345, xref="paper", yref="paper", text=f"model input: {n_used} camera FOVs", showarrow=False, font=dict(size=10))],
        shapes=[dict(type="rect", xref="paper", yref="paper", x0=0.70, y0=0.02, x1=0.99, y1=0.34, line=dict(color="#444", width=1), fillcolor="rgba(255,255,255,0.85)", layer="below")],
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(gridcolor="#eee"); fig.update_yaxes(gridcolor="#eee")
    return fig


def timeline_figure(s: Session, cols: list[str], index: int):
    import plotly.graph_objects as go

    fig = go.Figure()
    for col in cols:
        if col == s.reference:
            fig.add_trace(go.Scatter(x=s.times, y=s.gt_ade[col], mode="lines", line=dict(color="black", width=1.5, dash="dot"),
                                     name=f"{col} vs recorded future (median {np.nanmedian(s.gt_ade[col]):.2f} m)"))
        else:
            fig.add_trace(go.Scatter(x=s.times, y=s.paired[col], mode="lines+markers", marker=dict(size=4), line=dict(color=s.colors[col], width=2),
                                     name=f"{col} vs {s.reference} (median {np.nanmedian(s.paired[col]):.2f} m)"))
    fig.add_vline(x=float(s.times[index]), line=dict(color="#d62728", width=2))
    fig.update_layout(height=280, margin=dict(l=50, r=10, t=30, b=40), legend=dict(orientation="h", y=1.15, font=dict(size=10)),
                      xaxis=dict(title="time in scene (s)", gridcolor="#eee"), yaxis=dict(title="ADE (m)", gridcolor="#eee", rangemode="tozero"),
                      paper_bgcolor="white", plot_bgcolor="white", clickmode="event", hovermode="x unified")
    return fig


def build_app(s: Session):
    import dash
    from dash import Input, Output, State, ctx, dash_table, dcc, html
    from flask import Response

    app = dash.Dash(__name__, title=f"alpamayo eval: {s.export_name}")

    @app.server.route("/img/<col>/<camera>/<int:index>")
    def _img(col, camera, index):
        try:
            return Response(s.thumbnail(col, camera, index), mimetype="image/jpeg", headers={"Cache-Control": "max-age=86400"})
        except (KeyError, FileNotFoundError, IndexError):
            return Response(status=404)

    n = len(s.instants)
    marks = {i: f"{s.times[i]:.0f}s" for i in range(0, n, max(1, n // 12))}
    cols = list(s.columns)
    app.layout = html.Div(style={"fontFamily": "sans-serif", "margin": "12px"}, children=[
        html.H3(f"{s.manifest['scene']}  ·  export {s.export_name}  ·  runs {', '.join(sorted(set(s.run_of.values())))}", style={"margin": "4px 0"}),
        html.Div(style={"display": "flex", "gap": "18px", "alignItems": "center", "flexWrap": "wrap", "margin": "8px 0"}, children=[
            html.Button("▶ play", id="play", n_clicks=0, style={"width": "90px"}),
            html.Button("◀ step", id="back", n_clicks=0), html.Button("step ▶", id="fwd", n_clicks=0),
            html.Span("speed"), dcc.Dropdown(id="speed", options=[{"label": f"{v} instants/s", "value": v} for v in (1, 2, 4, 10)], value=2, clearable=False, style={"width": "150px"}),
            html.Span("cameras"), dcc.Dropdown(id="cameras", options=[{"label": k, "value": k} for k in CAMERA_SETS], value="front wide", clearable=False, style={"width": "170px"}),
            html.Span("arms"), dcc.Checklist(id="cols", options=[{"label": c, "value": c} for c in cols], value=cols[:4], inline=True),
            dcc.Checklist(id="aspect", options=[{"label": "equal aspect BEV", "value": "eq"}], value=[], inline=True),
            html.Span(id="where", style={"fontWeight": "bold"}),
        ]),
        dcc.Slider(id="idx", min=0, max=n - 1, step=1, value=0, marks=marks, tooltip={"placement": "bottom", "always_visible": False}, updatemode="drag"),
        dcc.Interval(id="tick", interval=500, disabled=True),
        html.Div(id="cards", style={"display": "flex", "flexWrap": "wrap", "gap": "10px", "marginTop": "10px"}),
        dcc.Graph(id="timeline", config={"displayModeBar": False}),
        html.H4("summary over the scene (compare-runs)"),
        dash_table.DataTable(data=s.summary_rows(), columns=[{"name": k, "id": k} for k in s.summary_rows()[0]], style_cell={"fontSize": 13, "padding": "4px"}, style_header={"fontWeight": "bold"}),
    ])

    @app.callback(Output("tick", "disabled"), Output("play", "children"), Output("tick", "interval"),
                  Input("play", "n_clicks"), Input("speed", "value"), State("tick", "disabled"))
    def _play(n_clicks, speed, disabled):
        playing = disabled if ctx.triggered_id == "play" else (not disabled)  # the button toggles; a speed change keeps the state
        return (not playing), ("⏸ pause" if playing else "▶ play"), int(1000 / speed)

    @app.callback(Output("idx", "value"), Input("tick", "n_intervals"), Input("back", "n_clicks"), Input("fwd", "n_clicks"),
                  Input("timeline", "clickData"), State("idx", "value"), prevent_initial_call=True)
    def _move(_t, _b, _f, click, idx):
        trig = ctx.triggered_id
        if trig == "tick" or trig == "fwd":
            return min(idx + 1, n - 1) if trig == "fwd" else (idx + 1) % n
        if trig == "back":
            return max(idx - 1, 0)
        if trig == "timeline" and click:
            t = click["points"][0]["x"]
            return int(np.argmin(np.abs(s.times - t)))
        return idx

    @app.callback(Output("cards", "children"), Output("timeline", "figure"), Output("where", "children"),
                  Input("idx", "value"), Input("cols", "value"), Input("cameras", "value"), Input("aspect", "value"))
    def _view(idx, selected, camera_set, aspect):
        t0 = s.t0s[idx]
        selected = [c for c in cols if c in (selected or [])]
        cameras = CAMERA_SETS[camera_set]
        ncol = 3 if len(cameras) >= 3 else len(cameras)
        img_w = THUMB_W // ncol
        cards = []
        ref = s.columns[s.reference].get(t0)
        for col in selected:
            rec = s.columns[col].get(t0)
            available = rec is not None
            imgs = [html.Img(src=f"/img/{col}/{cam}/{idx}", style={"width": f"{img_w}px", "display": "block"}, title=CAMERA_LABELS.get(cam, cam)) for cam in cameras] if available else [html.Div("not available for this instant", style={"width": f"{THUMB_W}px", "height": "120px", "background": "#ccc"})]
            text = _cot_text(rec) if rec else ""
            differs = bool(rec and ref and col != s.reference and _cot_text(rec) != _cot_text(ref))
            metrics = []
            if rec and rec.get("min_ade_vs_gt_m") is not None:
                metrics.append(f"ADE vs recorded future {rec['min_ade_vs_gt_m']:.2f} m")
            if col != s.reference and np.isfinite(s.paired[col][idx]):
                metrics.append(f"vs {s.reference} {s.paired[col][idx]:.2f} m")
            if np.isfinite(s.end_dist[col][idx]):
                metrics.append(f"predicted distance {s.end_dist[col][idx]:.1f} m")
            cards.append(html.Div(style={"width": f"{THUMB_W}px", "border": "1px solid #ddd", "background": "#fafafa"}, children=[
                html.Div(f"{col}  ·  {s.run_of[col]}", style={"background": "#222", "color": "white", "padding": "3px 6px", "fontSize": "13px"}),
                html.Div(imgs, style={"display": "flex", "flexWrap": "wrap"}),
                dcc.Graph(figure=bev_figure(s, col, idx, bool(aspect)), config={"displayModeBar": False}),
                html.Div(text or "(no output)", style={"margin": "4px 8px", "padding": "6px", "background": "white", "border": f"{'3px solid #d62728' if differs else '1px solid #999'}", "minHeight": "40px"}),
                html.Div("   ·   ".join(metrics), style={"margin": "2px 8px 8px", "fontSize": "13px", "color": "#333"}),
            ]))
        where = f"instant {idx + 1}/{n}   t = +{s.times[idx]:.1f} s"
        return cards, timeline_figure(s, selected, idx), where

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="Run locations (s3:// or local), all on the same export; the first is the reference.")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--cache", type=Path, default=Path("~/.cache/alpamayo-eval").expanduser())
    args = parser.parse_args()
    session = Session(args.runs, args.cache)
    print(f"loaded {len(session.instants)} instants, columns {list(session.columns)}; open http://{args.host}:{args.port}")
    build_app(session).run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())

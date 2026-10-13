# Alpamayo eval input export: format and minimum content

What `pd_alpamayo_eval.load_export()` reads. One export = one scene (one replica), both arms.
`pd_alpamayo_eval/export_from_pdncc.py` writes it from a pdncc bundle's replipair artifact (run in
pdncc's pixi env); anything else may write it too. This repo never touches HF for it. Derived
from the model's input contract in `src/alpamayo2_super/load_physical_aiavdataset.py`.

## Layout

```
<export>/
├── manifest.json
├── real/<camera>/<t_ns>.jpg             # real frames, as recorded (replipair `original/`)
├── rendered/<camera>/<t_ns>.jpg         # same camera, same pose, same t_ns, rendered by the replica
└── rendered_<variation>/<camera>/...    # optional arms: replipair `variations/<variation>/` renders
                                         # (e.g. rendered_snowy), same poses, altered appearance
```

`real` and `rendered` are the required arms and define which instants exist. Variation arms are
optional: an instant lists in its `arms` only the arms that have all 28 of its frames, and the
runner skips an arm where it is missing.

- `<camera>` is the Alpamayo canonical name, all seven, exactly:
  `camera_cross_left_120fov camera_front_wide_120fov camera_cross_right_120fov
  camera_rear_left_70fov camera_rear_tele_30fov camera_rear_right_70fov camera_front_tele_30fov`
  (this is what replipair already emits for the PAI-AV scenes).
- `<t_ns>` is the frame's capture timestamp in integer nanoseconds (the same clock as the ego
  poses). Cameras may be unsynchronized; the loader passes each camera's real timestamps to the
  model, as the NVIDIA loader does.
- Images: 1920x1080, original (distorted) framing, JPEG q95, written by the **same code path for
  both arms** (hard rule). No PNG, no undistortion, no resizing. A rendered frame must exist for
  every real frame used (same filename).
- A `rendered/` tree may be absent (real-only export, e.g. the first functional test).

## manifest.json

```json
{
  "version": 1,
  "scene": "pai_av_italy_tram_street_c3608270",
  "arms": ["real", "rendered", "rendered_snowy"],
  "time_origin_ns": 1699999999888906000,
  "ego_pose_convention": {"direction": "ego_to_world", "axes": "FLU"},
  "ego_poses": {"1699999999888906000": {"x": 0, "y": 0, "z": 0, "qw": 1, "qx": 0, "qy": 0, "qz": 0}},
  "instants": [
    {"t0_ns": 1700000005007035000,
     "frames": {"camera_front_wide_120fov": [1700000004722219000, 1700000004822213000, 1700000004922206000, 1700000005022200000]},
     "arms": ["real", "rendered"]}
  ],
  "nav_text": null,
  "cameras": {"camera_front_wide_120fov": {"width": 1920, "height": 1080, "distortion_model": "kannala_brandt", "K": [], "D": [], "cam_to_ego": {}}},
  "source": {"bundle_id": "...", "replipair_artifact": "s3://..."}
}
```

| key | required | meaning |
|---|---|---|
| `ego_poses` | yes | dense ego pose log, `{"<t_ns>": {"x","y","z","qw","qx","qy","qz"}}` (quaternion w,x,y,z). Any rate >= 10 Hz; the loader interpolates (linear xyz, slerp rotation) at `t0 - 1.5 .. t0` (16 points, 0.1 s) and `t0 + 0.1 .. t0 + 6.4` (64 points). Must cover `t0 - 1.5 s` for every instant; `t0 + 6.4 s` only if a GT-future readout is wanted (otherwise the loader omits the future and sets `ego_future_available = False`). |
| `ego_pose_convention` | yes | `direction`: `ego_to_world` (pose of the ego in the world) or `world_to_ego`. `axes`: `FLU` (x forward, y left, z up: the model's frame and the MCAP convention) or `RFU` (x right, y forward, z up). The loader converts. |
| `time_origin_ns` | yes | clock origin; the model's microsecond timestamps are relative to it (the exporter uses the first ego pose). |
| `instants` | yes | the `t0` list with, per camera, the four context frame timestamps (nearest frames to `t0 - 0.3/0.2/0.1/0 s`, within 20 ms) and the arms complete for that instant. `export_common.select_instants()` builds it. |
| `arms` | yes | every arm directory present. |
| `nav_text` | no | `null` for now (no CFG on the 1-GPU pool). |
| `cameras` | no | calibration per camera (K, D, model, cam-to-ego), informational only; the model never sees it and the loader does not pass it on. |
| `source` | no | provenance (bundle and artifact ids, window). |

## Minimum content, by what it buys

| goal | what to provide | size |
|---|---|---|
| loader + model wiring works ("functionality") | **one instant**: 4 frames x 7 cameras, real only, plus poses for `t0 - 1.6 s .. t0 + 6.4 s` | 28 JPEGs, ~6 MB |
| real vs rendered end to end, timing per instant, blind test | **one 3 s window** at 10 Hz, all 7 cameras, both arms, plus the pose log of the whole clip | 30 x 7 x 2 = 420 JPEGs, ~100 MB, gives ~27 instants |
| the actual per-scene regression number | the whole clip (~20 s), both arms | ~2800 JPEGs, ~1 GB per scene |

What makes an export unusable: non-consecutive frames (replipair's default random sample gives at
most 3 in a row), cameras sampled at different timestamps, frames resized or undistorted, poses
sparser than 10 Hz or not covering 1.6 s before the first instant.

## Why no calibration-dependent inputs

`select_task_input(data, "trajectory")` drops the rear tele and feeds six cameras x four frames,
ego history (16 x xyz + rotation in the ego frame at t0) and per-camera timestamps. Nothing else
reaches the model. The GT future is only used to compute minADE. See PD_EVAL_PLAN.md,
"The model's input contract".

# Handoff: Alpamayo 2 behaviour evaluation as a pdncc pipeline step

Written for the agent that will implement `BehaviourEvalStep` (working name) in
`pd-neural-content-creation` (pdncc) and a matching tab in `ncc bundle visualize`. Nothing in this
document has been implemented in pdncc; everything referenced lives in this repo
(`~/workspace/alpamayo2`, branch `karoly/test-pd-inference`) and was run end to end on three scenes
between 2026-10-12 and 2026-10-22. Findings are in `PD_EVAL_PLAN.md`; every run and dataset is in
`INVENTORY.md`; the manual recipe is `HOWTO.md`.

Read in this order: §1 (what the thing measures), §2 (the replipair render it needs), §3 (the
inference pipeline, file by file), §4 (the proposed step), §5 (the visualisation), §6 (gotchas).

---

## 1. What this measures, in one paragraph

NVIDIA's Alpamayo 2 Super (a 32B vision-language model plus a 2B diffusion "expert") is a driving
policy: given 6 cameras x 4 frames (t0-0.3 ... t0 s) and 1.6 s of ego history, it writes a
one-sentence reasoning ("Stop due to the red traffic light") and samples a 6.4 s future trajectory.
We run it open loop on the same instants twice: on the real camera frames and on the replica's
rendered frames at the recorded poses (plus weather variations). The difference between the two
predictions is the replica's *behavioural* effect on a downstream policy, which is what a replica is
for. Headline results so far: the metric is paired ADE (mean point distance between the two
trajectories, same seed) and the signed end-distance bias; a seed-to-seed noise floor is required
to read any number; on Madrid the entire effect was one rendering defect (the red traffic light was
not reproduced, the model ran it) found with attention maps and proven with camera occlusion, and
the fixed replica (v3) halved the paired ADE. The step should make exactly this loop a pipeline
artifact: real-vs-render trajectories, metrics against the floor, and the per-instant evidence
(attention, occlusion) for the viewer.

---

## 2. The replipair render the model needs

The model consumes, per instant, the four frames at t0, t0-0.1, t0-0.2, t0-0.3 s of all seven
cameras (it uses six; the rear tele is dropped internally but the loader requires the full ring).
A default replipair artifact is a random subsample of frames and is useless for that. The checked-in
step config `pd_alpamayo_eval/replipair_step_alpamayo.json` is what works:

```json
{"type": "ReplipairCreationStep", "skip_if_already_ran": false, "replipair_type": "pdncc",
 "replipair_kwargs": {"max_eval_frames": null, "camera_mode": "original", "rolling_shutter_mode": "auto",
   "color_mapping_mode": "auto", "render_variations": true, "off_axis_offsets": null,
   "write_valid_pixel_masks": false, "write_dynamic_agent_masks": false, "apply_occlusion_mask": false}}
```

Meaning: every frame of the clip (`max_eval_frames: null`), the native f-theta camera model and
resolution (`camera_mode: original`; the model resizes to ~504 x 280 px itself), rolling shutter and
colour mapping as trained, appearance variations when the splat has them (rainy/snowy), no off-axis
passes, no masks. Run as `ncc step run --bundle-id <bundle with replica_splat> --step-config @...`;
it produces a new bundle id. 15-30 min, x3 with two variations. Output layout the exporter reads
(`<replipair artifact>/`): `<camera>/original/*.jpg`, `<camera>/rendered/*.jpg`,
`variations/<rainy|snowy>/<camera>/rendered/*.jpg`, file stems `<session>__<camera>__<t_ns>`.

Requirements on the bundle: the original data artifact must carry 10 Hz ego poses and camera
calibrations in the MCAP (PhysicalAI-AV val clips do; `session.iter_ego_poses()` and
`session.camera_calibrations` in pdncc are what the exporter uses). The replipair must cover the
same clip so that for each 10 Hz pose instant the 28 context frames exist in `original/` and
`rendered/`; the exporter keeps only instants where they do (~175-186 of 202 on a 20 s clip; the
first 0.3 s and the last 0.1 s drop out, plus any gaps).

Lesson from Madrid v2 -> v3: emissive small objects (traffic lights, brake lights) must keep their
colour state in the render. That is a checkable requirement for the splat, and the evaluation
detects it (§3.6).

---

## 3. The inference pipeline, file by file

All code is under `pd_alpamayo_eval/` (ours) and `src/alpamayo2_super/` (NVIDIA's, unmodified).
NVIDIA's environment is `pyproject.toml` + `uv.lock` (torch 2.8.0+cu128, transformers 4.57.1,
flash-attn 2.8.3 from the prebuilt wheel); our `pixi.toml` only bootstraps python 3.12 + uv and adds
laptop-side envs. Do not fork NVIDIA's code; import it.

### 3.1 Data contract: the export (`EXPORT_FORMAT.md`, `export_common.py`)

An **export** is a directory (local or S3) with `manifest.json` and `<arm>/<camera>/<t_ns>.jpg`
frames. Arms: `real` (= replipair `original/`), `rendered`, `rendered_<variation>`. The manifest
(version 1) holds the scene id, the arms, `time_origin_ns`, the ego poses (10 Hz, FLU, world frame,
quaternion wxyz), the instants (each: `t0_ns`, the four frame timestamps per camera, which arms are
complete), the camera calibrations (Kannala-Brandt `K`, `D[4]`, `cam_to_ego` as translation +
quaternion, RDF camera convention) and the source ids. `select_instants()` picks the instants on
the pose grid whose 28 context frames exist within a tolerance; `build_manifest()` writes it.

### 3.2 Exporter: `export_from_pdncc.py` (runs in the pdncc env)

`python export_from_pdncc.py --bundle-id <id> --name full [--start-s --duration-s --max-instants
--arms --no-variations --dry-run]`. It uses only three pdncc things: `LocationBundle.from_artifact_id`,
`bundle.original_data_artifact` (session: `iter_ego_poses()`, `camera_calibrations`) and
`bundle.pdncc_replipair_artifact` (`.path` on S3). It lists frames per arm/camera, selects
instants, builds the manifest and **server-side copies** the frames into
`<replipair artifact>/tmp_alpamayo_eval/<name>/` (16 threads, ~5000 objects in a few minutes).
Nothing outside that new folder is touched. In the step this becomes: build the manifest and either
copy frames into the step's own artifact or, better, reference the replipair frames in place and
only write the manifest (the loader only needs paths; see §4).

### 3.3 Loader: `load_export.py`

`load_export(export_dir, t0_ns, arm, include_future=True)` returns exactly the dict NVIDIA's
`load_physical_aiavdataset()` produces: `image_frames` (7 cameras sorted by NVIDIA's camera index, 4
frames, C, H, W, uint8), `camera_indices`, timestamps, `ego_history_xyz/rot` (16 steps at 10 Hz in
the ego frame at t0), `ego_future_xyz/rot` (64 steps) when the pose log covers 6.4 s ahead
(`ego_future_available`), FLU/RFU conversions handled by `EgoTrajectory`. Then NVIDIA's
`select_task_input(data, "trajectory")` drops the rear tele. Tests: `tests/test_pd_load_export.py`
(synthetic export, conventions, timing). The loader is the contract; if the step keeps the export
format, the loader is reusable as is.

### 3.4 Runner: `run_export.py` (runs in NVIDIA's env, one H100)

`python -m pd_alpamayo_eval.run_export --model-id <weights dir> --export <dir or s3> --out-dir <dir>
--run-id <id> [--upload-to s3://...] [--arms ...] [--max-instants N | --instant-indices ... |
--instant-times ...] [--num-traj-samples 1] [--diffusion-steps 10] [--seed 42] [--save-viz]
[--no-attention] [--no-attention-per-layer] [--ablate blank real] [--upload-every 10]
[--resume-from s3://...]`. Per instant and arm it: loads the data (3 s, CPU; the main cost),
builds the model inputs with NVIDIA's `helper.prepare_model_inputs`, seeds CUDA with the run seed
(same seed for every arm = common random numbers, so differences come from the pictures), calls
`model.sample_trajectories_from_data(top_p=0.98, temperature=0.6, num_traj_samples, diffusion
inference_step=10, return_extra=True)` (1.7-2.0 s on an H100), and writes:

- `<t0_ns>_<arm>.json`: sentence (`cot`), timings, `peak_vram_gib`, `ade_vs_gt_m` /
  `min_ade_vs_gt_m` when the future exists, attention mass split, occlusion summaries;
- `<t0_ns>_<arm>.npz`: `pred_xyz` (samples, 64, 3) in the ego frame at t0, `gt_xyz` (64, 3) when
  available, attention arrays (§3.6), occlusion arrays (§3.7);
- `results.jsonl` (one line per record), `summary.json` (config, device, per-arm percentiles,
  paired ADE real-vs-arm, `partial` flag), optional NVIDIA PNG per record with `--save-viz`.

The run checkpoints to the upload target every 10 instants and resumes with `--resume-from` (the
cluster loses nodes). Determinism: the real arm is bit-identical across jobs for a seed (verified on
every rerun), so a run never needs repeating for the same (export, arms, seed).

Model: `Alpamayo2Super.from_pretrained(dir, dtype=bf16, device_map="cuda:0")`, 67 GB of weights
(gated HF repo `nvidia/Alpamayo2-Super`; a copy sits on the Nebius volume
`/mnt/pd-general/data/alpamayo-eval/models/alpamayo2-super`), 69 GiB peak VRAM: needs an
80 GB card (H100) or 96 GB (RTX PRO 6000); not a single L40S.

### 3.5 Metrics: `compare_runs.py`, `dashboard.py::Session._series`

Per instant and arm vs the reference (the first run's `real`): paired ADE over 64 steps and per
horizon (1/2/3/6.4 s, exp(-t/2 s)-weighted), signed difference of the 6.4 s end distance (the
"speed bias": seeds are symmetric, renders are one-sided), ADE vs the recorded future, sentence
difference (`cot_differs`: ignores punctuation and wording, flags a changed action/direction word or
<50 % content overlap). Scene level: medians, p90, share longer/shorter by >1 m, shift-vs-spread (t
statistic and sign test). **A seed floor run** (`--arms real --seed 1`) is mandatory per scene; the
gate proposal in the plan is "signed bias over many instants, not per-instant ADE".

### 3.6 Attention maps: `attention.py`

The diffusion expert attends directly into the VLM's KV cache (all image patch tokens of the 24
frames, 10 x 18 tokens of 28 px each). `ExpertAttentionRecorder` wraps the attention function
transformers dispatches to, calls the original unchanged (predictions bit-identical) and records,
per layer, the mean softmax weight over the 64 trajectory tokens, heads and 10 diffusion steps, both
plain and multiplied by the value-vector norm (removes attention sinks). `image_token_layout()` maps
tokens back to (camera, frame, row, col). Stored per record: `attn_img_layers` and
`attn_vw_img_layers` (64 x 4320, float16, ~0.5 MB each), `attn_layout` (json), mass splits. The
usable view is value-weighted, middle layers (16-47), per frame scaled above its median. Tests:
`tests/test_pd_attention.py` (recorded weights equal transformers' eager attention; outputs
unchanged). Caveat: a map of where the expert reads, not causal evidence; crisp on small decisive
objects (traffic light), smeared on large ones.

### 3.7 Camera occlusion: `ablation.py`, `run_export.py --ablate`

Per record one extra forward per camera with that camera's four frames replaced by their mean
colour (`blank`) or, for render arms, by the real frames (`real`), same seed. Stored:
`ablate_<mode>_xyz` (6, 64, 3) and per-camera ADE/distance/sentence in the json. This is the causal
readout (no model internals, immune to sinks); 13 forwards per instant pair, ~75 s per instant with
the current CPU-bound loop. `python -m pd_alpamayo_eval.ablation <run>` prints the scene table.

### 3.8 Cluster plumbing (`scripts/`), for reference only

`scripts/submit_nebius.py` (Ray Jobs API, uploads the code tree, forwards the Nebius S3 key as job
env) and `scripts/nebius_entrypoint.sh` (builds NVIDIA's env with `uv sync --locked
--no-install-package flash-attn` + the prebuilt flash-attn wheel in ~45 s warm, checks weights,
stages the export to the pod SSD, runs the runner, uploads). In pdncc the step framework replaces
all of this; what carries over is the env recipe and the weights location.

### 3.9 Laptop tools

`quick.py` (run a few chosen instants and get contact sheets in ~10 min), `mirror_export.py`
(AWS <-> Nebius copy), `make_video.py` (side-by-side video), `registry.py` (scene -> runs for the
dashboard). `tests/manual/probe_dashboard_layout.py` (headless layout check).

---

## 4. Proposed `BehaviourEvalStep`

### 4.1 Inputs and outputs

- Input artifacts: `RepliPairDataArtifact` (pdncc type, with the §2 render) and the
  `OriginalDataArtifact` (MCAP: ego poses, calibrations). Precondition check: every-frame render.
- Output artifact (new DataArtifact type, e.g. `BehaviourEvalArtifact`, under the bundle like
  `replica_replipair_pdncc`): `manifest.json` (the export manifest of §3.1 with frame paths pointing
  at the replipair artifact, no frame copies), `runs/<run id>/` as written by the runner (§3.4), and
  a `metrics.json` with the scene-level table of §3.5 (reference real, every arm, the floor when a
  seed-1 run is part of the step). Keep the runner's file layout unchanged so the existing dashboard,
  `compare_runs.py`, `quick.py` and `ablation.py` open the artifact directly.
- Registry row: scene, bundle, arms, seed(s), run id, headline numbers (mirror `INVENTORY.md`'s
  experiment log so nobody reruns what exists).

### 4.2 Config (step kwargs)

`arms` (default: all present), `seeds` (default `[42]`; `[42, 1]` adds the floor on the real arm),
`num_traj_samples` (1), `diffusion_steps` (10), `window` (`start_s`, `duration_s`, or instant
list), `record_attention` (True), `attention_per_layer` (True), `ablate` (`[]`, `["blank"]`,
`["real"]`), `save_viz` (False), `model_id` (weights path), `upload_every` (10).

### 4.3 Execution

GPU step (`num_gpus=1`, pod with an 80 GB card; `memory` ≥ 80 GB host RAM for safety), its own pixi
env built from NVIDIA's lock (a `[feature.alpamayo]` env in pdncc's `pixi.toml` mirroring
`pyproject.toml`/`uv.lock`, flash-attn from the prebuilt wheel; or run NVIDIA's uv env as a
subprocess like the entrypoint does). Phases: (1) export manifest from the artifacts (seconds; port
`export_from_pdncc.py::main` minus the copy); (2) stage frames to local SSD (the runner reads
local jpgs; 2 GB per scene); (3) load the model once; (4) loop instants x arms x seeds with
checkpoint uploads; (5) metrics table; (6) register the artifact. Idempotent: `already_ran` =
artifact exists with a non-partial `summary.json` for the configured (arms, seeds); `--resume-from`
semantics for a partial one. Wall clock: ~1 GPU-hour per scene for four arms at 2 s/record plus 3 s
load; a prefetching data loader (background thread decoding the next instant's 24-96 jpgs) would
halve it and is the first optimisation to make (`load_s` 2.9 s vs `inference_s` 1.7 s per record).

### 4.4 Things to keep exactly

Camera order and the rear-tele rule (`CANONICAL_CAMERA_NAMES`, `CAMERA_NAMES_TO_INDICES`), the
t0-relative 10 Hz grid (`STEP_NS`), FLU ego frame at t0 for history/future/predictions, one seed for
all arms, `torch.cuda.manual_seed_all(seed)` before every sample, bf16, `top_p 0.98 / temperature
0.6` (NVIDIA's defaults), the npz/json field names (the viewer and tools read them).

---

## 5. Visualisation: what the dashboard shows and how to port it

Current implementation: `dashboard.py` (Dash 4 / plotly 7, ~900 lines), one page per scene,
loading one or more runs (`--scene <name>` via `registry.py`, or run uris; `--labels` to tag runs,
`+` merges a run with identical predictions, e.g. an occlusion run, into the same cards). Verified
layout-stable with a headless probe. Screens and behaviour worth reproducing in the pdncc viewer:

- **Controls**: instant slider over the clip (play/pause/step, speed), camera set (front wide /
  front three / all six / one camera), arm checklist (one card per arm, cards in one row sized to
  the window), "paths on images", one **overlay** dropdown (off / model attention / occlusion blank /
  occlusion real swap), ADE horizon.
- **Card per arm**: the camera frame(s) at t0 with the paths projected into the image
  (`CameraProjector`: Kannala-Brandt forward projection with the manifest calibration; the arm's 6.4 s
  prediction thick in its colour, the reference prediction faint green, the recorded future dashed
  black) and the overlay blended underneath; the reasoning sentence (fixed three-line box, red border
  when `cot_differs` from the reference); a bird's-eye plot (equal aspect, camera FOV wedges around
  the ego, shown camera in gold, wedges shaded by attention share or occlusion effect, predictions
  and partial recorded future); a four-row metrics table (ADE vs recorded future @3 s/6.4 s, vs real
  @3 s/6.4 s; this instant and mean so far); an error-over-time strip (paired ADE per instant, click
  to jump, red cursor, all cards on one scale).
- **Colours**: one hue per arm (real green, rendered orange, rainy blue, snowy purple), darker shades
  for further runs of the same arm (`@v2`, `@seed1`); red reserved for the cursor and "differs".
- **Heatmap** (§3.6): `_attention_grids` → mid-layer, value-weighted grid → `_blend_heatmap`
  (median subtracted, per-frame peak, gamma 0.7, inferno, alpha 0.8, bilinear upsample of the 10 x 18
  grid). **Occlusion**: `_ablation_effects` (blank: ADE shift; real swap: share of the gap to real
  removed) → wedge opacity + hover text + table row.
- **Data it needs** from the artifact: `manifest.json` (poses, calibration, instants), the frames
  (fetched per view, cached), `results.jsonl` + npz per record. Precomputed once per session:
  paired ADE series per horizon, GT ADE, end distances, text-differs, projectors, scene-average
  attention (for the "minus scene average" variant, not exposed in the UI).
- **Engineering notes that cost days**: pattern-matching component ids built once (no remounts while
  stepping, otherwise everything blinks), fixed graph heights and fixed trace structure, image URLs
  namespaced by export+run (browser cache), a page token with self-reload so a stale tab never talks
  to a restarted server, lazy per-frame staging, thumbnails cached on disk.

The pdncc viewer is Viser-based (`ncc bundle visualize`, one tab per artifact,
`cli/commands/bundle_visualize.py` → `_run_visualizer`). A faithful port as a Viser tab: a time
slider, per-arm image panes with the overlay and projected paths as pre-rendered jpgs (the same
`Session.thumbnail` function produces them, it is pure PIL), the bird's-eye plot as 2D line
primitives, and the metrics as text; or embed the Dash app as an iframe tab if Viser allows. Either
way, reuse `Session` (it is the model of the view: all series, projectors and thumbnails) and keep
the overlay semantics; the Dash layer is replaceable.

---

## 6. Gotchas, in the order they bit us

1. Random-subsample replipairs look fine and give zero usable instants; use the §2 config.
2. The model requires the full seven-camera ring in the input dict even though it drops one.
3. Per-instant ADE is swamped by sampling noise on scenes where the model is bimodal (stop/go at
   an intersection); always run the seed floor and read the signed distance bias and the tails.
4. Sentence comparison is noise unless normalised (`cot_differs`); even then ~40-70 % differ.
5. Raw attention is attention sinks (sky patches); only value-weighted middle layers are readable.
6. Attention points at where the model reads, occlusion at what causes the decision; they can name
   different cameras for the same cue (Madrid: tele vs wide).
7. Results must be uploaded incrementally; the Ray cluster loses nodes (two runs died).
8. Image decoding is the bottleneck (3 s vs 1.7 s inference per record); prefetch.
9. Never move data through Ray `working_dir`; data goes through S3. Never write into shared
   volumes that are full (`/mnt/pdfm-data` was at 100 %).
10. Keep file names and npz keys stable: five tools read them.

## 7. Pointers

| what | where |
|---|---|
| plan, findings, numbers | `PD_EVAL_PLAN.md` |
| every bundle / artifact / export / run / video | `pd_alpamayo_eval/INVENTORY.md` |
| manual recipe | `pd_alpamayo_eval/HOWTO.md` |
| export format | `pd_alpamayo_eval/EXPORT_FORMAT.md` |
| replipair config | `pd_alpamayo_eval/replipair_step_alpamayo.json` |
| exporter / loader / runner | `export_from_pdncc.py`, `load_export.py`, `run_export.py` |
| metrics / attention / occlusion | `compare_runs.py`, `attention.py`, `ablation.py` |
| dashboard / quick / video / registry | `dashboard.py`, `quick.py`, `make_video.py`, `registry.py` |
| tests | `tests/test_pd_*.py`, `tests/manual/probe_dashboard_layout.py` |
| NVIDIA model code (unmodified) | `src/alpamayo2_super/` (`models/alpamayo2_super.py`, `models/expert.py`, `helper.py`, `input_profiles.py`) |
| pdncc analogues | `pipeline_steps/pipeline_step.py` (base), `pipeline_steps/geometry_evaluation_step.py` (an evaluation step with its own artifact), `pipeline_steps/replipair_step.py`, `common/data_artifact/replipair_data_artifact.py`, `cli/commands/bundle_visualize.py` |

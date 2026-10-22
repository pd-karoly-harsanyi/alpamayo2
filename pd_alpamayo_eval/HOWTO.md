# How to run Alpamayo 2 Super on a PD replica and inspect the results

End-to-end recipe, PD-internal. Every step was run on 2026-10-13/14 for the three PAI-AV scenes
(Italy tram street, Madrid intersection, Austria roundabout); the ids used appear in the examples so
you can repeat them exactly. Background and findings: `PD_EVAL_PLAN.md`. Input format:
`pd_alpamayo_eval/EXPORT_FORMAT.md`.

## 0. One-time setup on your laptop

- **VPN**: the Ray cluster (`ray-nebius.internal.paralleldomain.com:8265`) is VPN-only.
- **pixi** (`~/.pixi/bin/pixi`): this repo's `pixi.toml` defines three envs. `launch` (submit jobs,
  mirror data), `dev` (tests, dashboard, video; CPU only), `default` (what the cluster worker uses).
  `pixi install -e launch -e dev` once; pixi tasks below install on demand anyway.
- **AWS `default` profile** = your PD SSO login (`aws sso login`). Used by the exporter to read the
  pdncc artifact bucket `paralleldomain-pipeline` and by the mirror as the source.
- **AWS `nebius` profile** = the Nebius Object Storage key of service account `rayio-storage`
  (key `pdfm-ray-shell-forward`, id `NAKIGKUBBE6D4NO1EF5U`; secret from
  `nebius iam v2 access-key list --parent-id project-e00ate7dpr00brrpem4zqa` once you have IAM
  read on the project, which Jannik grants). Store it as a separate profile, never in `default`:

  ```
  # ~/.aws/credentials
  [nebius]
  aws_access_key_id = NAKI...
  aws_secret_access_key = ...
  # ~/.aws/config
  [profile nebius]
  region = eu-north1
  endpoint_url = https://storage.eu-north1.nebius.cloud
  ```

  Check: `AWS_PROFILE=nebius aws s3 ls s3://pd-nebius-ray/alpamayo-eval/`.
- **pdncc checkout** at `~/workspace/pd-neural-content-creation` with its own pixi env (the exporter
  runs there because it needs the bundle/registry stack).
- **Hugging Face**: nothing. The 67 GB weights already sit on the cluster volume
  (`/mnt/pdfm-data/pdfm-data-cache/alpamayo-eval/models/alpamayo2-super`); the PhysicalAI-AV dataset
  is not used.

## 1. Render the replica frames the model needs (pdncc, GPU cluster)

The model wants, per instant, the four frames at t0, t0-0.1, t0-0.2, t0-0.3 s of all seven cameras,
real and rendered. A default replipair artifact is a random subsample and is useless for that; rerun
the replipair step on a bundle that has a trained dynamic splat, with the checked-in config
(every frame, native f-theta camera model, rolling shutter and colour mapping as trained, weather
variations if the splat has them, no off-axis passes, no masks):

```
cd ~/workspace/pd-neural-content-creation
pixi run ncc step run --bundle-id <bundle with replica_splat> \
  --step-config @/home/karoly.harsanyi/workspace/alpamayo2/pd_alpamayo_eval/replipair_step_alpamayo.json
```

It prints a NEW bundle id (`location_artifact_id`); that is the bundle you use from here on.
Takes 15-30 min without variations, about three times that with rainy+snowy.
Bundles used so far: Italy `ceb8e94d-de22-4558-be1d-d7da397e75f0`, Madrid
`ea732026-8a79-474e-9bf4-70b2f6438256`, Austria `890838c3-7aa1-4278-b8e4-e4040b0688fb`.

## 2. Export the model inputs into the artifact (laptop, pdncc env)

```
cd ~/workspace/pd-neural-content-creation
pixi run python ~/workspace/alpamayo2/pd_alpamayo_eval/export_from_pdncc.py \
  --bundle-id <new bundle id> --name full            # whole scene
# or a window:  --name w5s_3s --start-s 5 --duration-s 3
# add --dry-run to only list the instants and arms it would export
```

What it does: reads the bundle's replipair artifact and original-data MCAP (10 Hz ego poses,
calibration), picks every instant on the ego-pose grid whose 28 context frames exist in `real`
(= replipair `original/`) and `rendered`, adds `rendered_<variation>` arms where complete, and
writes `<replipair artifact>/tmp_alpamayo_eval/<name>/` in the artifact bucket by server-side copy
(a couple of minutes for ~5000 frames). Nothing outside that new folder is touched. A whole 20 s
scene is ~175-185 instants, ~5000 frames, ~2 GB with two variations.

## 3. Mirror the export to the Nebius bucket (laptop)

Nebius pods cannot reach AWS, so copy once (one egress), from the laptop, with the two profiles:

```
cd ~/workspace/alpamayo2
pixi run -e launch mirror-export \
  s3://paralleldomain-pipeline/artifacts/replica_replipair_pdncc/<replipair artifact uid>/tmp_alpamayo_eval/full \
  s3://pd-nebius-ray/alpamayo-eval/<export name>
```

The replipair artifact uid is printed by the exporter (`replipair s3://...`). Use a distinct
`<export name>` per scene (`full_variations`, `madrid_full_variations`, `austria_full_variations`).
~2 GB takes 5-10 min; reruns skip objects already present.

## 4. Run the model on the cluster (laptop submits, cluster runs)

First check `INVENTORY.md` ("Experiments already run"): if the (scene, replica version, arms, seed)
combination is listed, the results are in the bucket and staged in `~/.cache/alpamayo-eval`; use
them. The real arm is deterministic per seed, so it never needs rerunning for a seed already listed.

```
cd ~/workspace/alpamayo2
pixi run -e launch submit-nebius run --wait --forward-nebius-credentials -- \
  --export s3://pd-nebius-ray/alpamayo-eval/<export name> [--save-viz] [--arms real rendered] [--seed 42]
```

- Instant selection: `--max-instants N` (first N), `--instant-indices i j k`, `--instant-times 0.0 4.2`
  (clip seconds, nearest instants). The `quick` task (4b) wraps this.

- `--forward-nebius-credentials` reads the local `nebius` profile and forwards the key as job env,
  pinned to the Nebius endpoint, like every pd-fm Nebius job. (The job spec is readable on the
  unauthenticated dashboard while the job record exists; the key is scoped to one bucket.)
- Since 2026-10-20 every run also records the expert's attention over the images into the npz
  (`attn_img`, `attn_layout`, `attn_mass`, `attn_image_mass`, ~100 KB per record; `--no-attention`
  turns it off). The recording wraps the attention function and recomputes the softmax weights on the
  side, so predictions are bit-identical with or without it (`tests/test_pd_attention.py`).
- **Camera occlusion** (`--ablate blank`, `--ablate real`, or both): per record one extra forward per
  camera, same seed, with that camera's four frames replaced by their mean colour (blank: how much the
  prediction depends on the camera) or, for render arms, by the real frames (real swap: how much of the
  gap to the real prediction that camera's render carries). Causal, no model internals, immune to the
  attention-sink problem of the heatmaps; 7 x the cost (about 2 h per scene for four arms, 1 h for
  `--arms real rendered`). Results: npz `ablate_<mode>_xyz` (cameras, 64, 3), json `ablate_<mode>` per
  camera; scene summary with `python -m pd_alpamayo_eval.ablation <run>`; dashboard "model attention"
  dropdown entries "occlusion: blank camera" / "occlusion: real swap" (wedges shaded by the effect,
  hover for metres, one table row).
- The worker uploads only the code tree (never data), builds NVIDIA's env from `uv.lock` on the pod
  (prebuilt flash-attn wheel, ~45 s warm / 2 min cold), stages the export onto the pod's local SSD,
  loads the model (28 s, 69 GiB), runs every instant x arm with ONE seed for all arms (common random
  numbers), and uploads results to `s3://pd-nebius-ray/alpamayo-eval/<export name>/runs/<run id>/`.
- Cost: 1.75 s inference per instant per arm; ~11 s per instant for four arms without `--save-viz`
  (~35 min per scene), ~19 s with the NVIDIA PNGs. Cold pod start adds ~5 min.
- The log ends with `uploaded ... -> s3://.../runs/<run id>` and a `SUMMARY {...}` line.
- **Node deaths** (the cluster loses a worker now and then: "health check failed due to missing too
  many heartbeats"): since 2026-10-21 the run checkpoints to its bucket folder every 10 instants
  (`--upload-every`, partial `summary.json` included, so the dashboard can open a partial run). If the job
  dies, resubmit the same command plus `--resume-from s3://pd-nebius-ray/alpamayo-eval/<export>/runs/<dead run id>`:
  the new run takes over the finished records and continues; at most 10 instants are redone.
- Other modes: `probe` (env + weights only), `smoke` (NVIDIA's own sample; needs HF dataset access,
  which the cluster token lacks), `exec -- <cmd>` (anything inside the model env).

**Seed-to-seed noise floor** (needed to read any paired number): same export, real arm only,
another seed:

```
pixi run -e launch submit-nebius run --wait --forward-nebius-credentials -- \
  --export s3://pd-nebius-ray/alpamayo-eval/<export name> --arms real --seed 1
```

## 4b. Small experiments in minutes: quick (laptop, dev env)

```
pixi run -e dev quick madrid --times 0.0 4.2 8.0                        # a few instants, all arms, attention maps
pixi run -e dev quick madrid --times 4.2 --arms real rendered --ablate blank real   # plus camera occlusion
pixi run -e dev quick --render-only s3://pd-nebius-ray/alpamayo-eval/<export>/runs/<run id>   # just the pictures
```

Submits the job for the chosen instants only (`--times` in clip seconds as the dashboard shows them,
or `--indices`), waits (about 5 min pod start + 2 s per instant and arm, 14 s per occluded instant),
stages the results and writes one contact sheet per instant to `outputs/quick/<name>/`: front wide,
front tele, cross left, cross right x arms, with the paths, the attention overlay (value-weighted,
middle layers) and, with occlusion, the per-camera effect printed on each tile; `summary.txt` has the
sentences and numbers. Use this before any full-scene run: a hypothesis about a few instants costs
ten minutes and produces something to look at.

## 5. Numbers: compare-runs (laptop, dev env)

```
pixi run -e dev compare-runs --scene italy       # runs from registry.py
pixi run -e dev compare-runs s3://pd-nebius-ray/alpamayo-eval/<export>/runs/<run A> [s3://.../runs/<run B> ...]
```

Reference = the first run's `real`. Per other column (arms of all runs; later runs' arms are suffixed
`@seed<N>`): paired ADE vs reference (median / p90), signed difference of the distance covered by the
predicted 6.4 s trajectory (median; share of instants longer / shorter by >1 m), ADE vs the recorded
future, and how often the Chain-of-Causation sentence differs. `real@seed1` vs `real` is the floor;
a replica effect is only a finding where it exceeds the floor, and the distance bias is the cleanest
discriminator because seeds are symmetric and renders are not. Runs that share a seed (an old and a
new render of the same scene) are tagged by export name instead; `--labels old new` overrides. Runs and exports are staged into
`~/.cache/alpamayo-eval` on first use (same cache as the dashboard and the video).

## 6. Look: the dashboard (laptop, dev env)

```
pixi run -e dev dashboard --scene italy          # or madrid, austria: reference run, v2 run, seed floor
pixi run -e dev dashboard \
  s3://pd-nebius-ray/alpamayo-eval/<export>/runs/<run A> [s3://.../runs/<run B> ...] [--port 8050]
```

then open http://localhost:8050 (one server shows one scene; start a second on `--port 8051` for
another). Nothing is rerun: the dashboard reads the stored results, staged into the laptop cache on
first use. `--scene` takes the run list from `pd_alpamayo_eval/registry.py`; add a row there when a
new run lands. The first run's `real` is the reference. Further runs on the same export are tagged by
seed (`real@seed1`); a run on ANOTHER export of the same scene (a newer render of the same clip,
e.g. `italy_full_variations_v2`) is tagged by export name and its cards show that export's images,
so old and new renders sit side by side on the same instant. `--labels '' v2` sets your own tags; the label `+` merges a run with identical predictions (an
occlusion or attention run of the same export and seed) into the existing columns, so its extra
fields show on the same cards.

- Slider over instants with play / pause / step, speed, and a time readout. Click the slider handle
  and the arrow keys step too.
- Colours: one hue per arm (real green, rendered orange, rainy blue, snowy purple); further runs of the
  same arm (`@v2`, `@seed1`) use darker shades of that hue.
- One card per arm in a single row that fills the window (tick the arms you want): camera frame(s)
  at t0 with the paths projected onto them (reference prediction faint green, the arm's prediction
  thick in its colour, recorded future dashed black; the bird's-eye legend names them), the model's
  sentence right under the image (red box when it says something materially different from the reference: a changed action or direction word, or less than half the content words in common; punctuation and wording jitter are ignored), a bird's-eye plot (equal
  aspect, zoomable, camera FOV wedges around the ego with the shown camera in gold), the instant's ADE
  table, and that arm's error-over-time plot (click to jump; all on one scale).
- Cameras dropdown: front wide, the front three, all six used, or any single camera.
- **overlay** dropdown: `model attention` = where the trajectory expert read from each image
  (attention x value norm so the attention sinks drop out, middle layers, each frame scaled to its own
  peak above its floor; runs made before 2026-10-22 only have plain attention and say so); the
  bird's-eye wedges are shaded by each camera's share and the table gets an attention row.
  `occlusion: blank camera` / `occlusion: real swap` (runs made with `--ablate`): wedges shaded by the
  causal effect of each camera, hover for metres. One control, no sub-options; the underlying choices
  (layer band, weighting, scale, scene-average subtraction) remain available as URL parameters of the
  image route for scripts.
- Scene-level numbers are not on the dashboard; use `compare-runs` (section 5).
- After pulling new code: restart the server AND hard-reload the tab (Ctrl-Shift-R). A stale tab gives
  `IndexError ... _prepare_grouping` 500s in the server log.
- If anything jumps while stepping: `pixi run -e dev python tests/manual/probe_dashboard_layout.py
  http://127.0.0.1:8050/` reports the element that moves.

## 7. Share: the video (laptop, dev env)

```
pixi run -e dev make-video s3://pd-nebius-ray/alpamayo-eval/<export>/runs/<run id> --out outputs/<name>.mp4 [--fps 2]
```

One frame per instant: a tile per arm (frame at t0, bird's-eye plot with the FOV inset, text) and a
per-arm error strip with a cursor. ~10 min per scene; `outputs/` is gitignored. Always pass `--out`
with a distinct name per run (`<scene>_<export>_<run suffix>.mp4`), so a re-render of a scene never
overwrites last week's video.

## 8. Where things are

| what | where |
|---|---|
| model inputs (per scene) | `s3://paralleldomain-pipeline/artifacts/replica_replipair_pdncc/<uid>/tmp_alpamayo_eval/<name>/` (AWS) and `s3://pd-nebius-ray/alpamayo-eval/<export name>/` (Nebius) |
| run outputs | `s3://pd-nebius-ray/alpamayo-eval/<export name>/runs/<run id>/`: `results.jsonl`, `<t0>_<arm>.json` + `.npz` (pred and GT xyz), `summary.json`, PNGs with `--save-viz`; copy back with `mirror-export` reversed |
| laptop cache | `~/.cache/alpamayo-eval/` (safe to delete) |
| cluster volume | `/mnt/pdfm-data/pdfm-data-cache/alpamayo-eval/` holds only the weights and the HF cache; the volume is shared and was 100 % full on 2026-10-13, so nothing else is written there |
| Ray dashboard | `http://ray-nebius.internal.paralleldomain.com:8265/#/jobs/<run id>` |

Runs so far (all seed 42, 1 sample, 10 diffusion steps): Italy `full_variations` run
`alpamayo-run-27f93c3e` + floor `alpamayo-run-af7ff7bc` (seed 1); Madrid `madrid_full_variations`
run `alpamayo-run-e368b20f`; Austria `austria_full_variations` run `alpamayo-run-17c45f0a`.
Improved replicas (2026-10-19): Italy `italy_full_variations_v2` run `alpamayo-run-928b4727`,
Austria `austria_full_variations_v2` run `alpamayo-run-9fc99df6` + floor `alpamayo-run-b357ae61`
(seed 1), Madrid `madrid_full_variations_v2` run `alpamayo-run-99bb841a` + floor `alpamayo-run-e6d4f80e`
(seed 1) + the same predictions with expert attention `alpamayo-run-ed10a713` (2026-10-21; the
reference run of `--scene madrid`).

## 9. Tests

`pixi run -e dev test-pd` runs the CPU contract tests of the loader (synthetic export, FLU/RFU
conventions, timing, optional arms), of the sentence comparison behind the red text box, and of the
attention recorder (a tiny Qwen2 decoder on CPU: recorded weights equal transformers' eager attention,
outputs unchanged; needs `transformers` in the dev env, same version as the model env). The loader is `pd_alpamayo_eval/load_export.py`; it produces
exactly the dict NVIDIA's `load_physical_aiavdataset()` produces, and NVIDIA's code under
`src/alpamayo2_super/` is unmodified.

# PD: Alpamayo 2 Super as a behavioral replica probe — plan and state

Working document for picking this up in a new session. Written 2026-10-12 from a session in
`~/workspace/pd-fm`. This file is PD-internal and sits untracked in an NVIDIA checkout
(`origin` = `github.com/NVlabs/alpamayo2`); move it with the code when the eval repo is created.

## Goal

Run Alpamayo 2 Super on (a) real multi-camera footage and (b) pdncc splat renders of the same
scene at the same poses, and compare its predictions (trajectory = quantitative gate, Chain-of-Causation
text = explanation). Same behavior => the replica is fit to drive in. Internal regression use first;
no fine-tuning, ever. The full rationale, decisions and metric design live in
`~/workspace/pd-neural-content-creation/ALPAMAYO_EVAL_BRAINSTORM.md` (§1 state, §3 decisions,
§6.9 metrics, §6.10 architecture). This file does not repeat them.

Division of labour: **pdncc** builds the replica and exports clip inputs (real + rendered frames,
calibration, ego poses). **A separate eval repo** (to be created, based on this checkout) runs the
model on the Nebius H100 cluster, writes raw outputs to S3, and scores offline. The user produces
the renders; the eval side is what this plan covers.

## What is already in place (verified 2026-10-12)

- **Compute.** The Nebius Ray cluster (`ray-nebius.internal.paralleldomain.com:8265`, VPN-only)
  has two H100 80 GB pools: a 1-GPU pool (Ray tag `{h100: 1}`, 16 vCPU, 200 GB RAM, 558 GB SSD)
  and one 8-GPU node (`{h100x8: 1}`, `maxReplicas: 1`, usually held by pd-fm training). The user
  can submit: a pd-fm bench job ran end to end on the 1-GPU pool. The brainstorm's §5 blocker
  ("do we have 80 GB nodes?") is therefore resolved.
- **Model fit.** The single-GPU batch-1 path needs ~72 GB (README, this repo) -> fits one H100.
  The two-GPU `nav_text` CFG demo (`examples/two_gpu_nav_cfg_demo.py`) needs 2x80 GB -> only the
  8-GPU node. Start single-GPU, no CFG.
- **Data.** Three NVIDIA PhysicalAI-AV val-split scenes (Austria roundabout, Madrid intersection,
  Italy tram street) were converted to MCAP with 3D cuboid tracks and dynamic replicas were built
  ("reconstructed successfully", `~/workspace/pai_av_handover/README.md`, table "The three scenes
  that were built" has bundle UUIDs + 10 Hz curated bundle UUIDs). Exact NVIDIA rig => zero rig gap.
  Jannik's unmerged pdncc branch `jannik/nvidia-av-vehicles-export` went on to 60 scenes after the
  brainstorm doc stopped being updated — ask him what exists beyond the three. Raw scene folders
  (`~/data/pai_av_scenes`) are gone from this box; HF streaming still works.
- **Hugging Face.** The local token (`~/.cache/huggingface/token`, account `harsandor`) returns
  200 on both gated repos: `nvidia/Alpamayo2-Super` and `nvidia/PhysicalAI-Autonomous-Vehicles`.
- **Licensing.** Model weights OpenMDW-1.1, code Apache-2.0 (forking is fine, keep LICENSE +
  notices). PhysicalAI-AV-derived numbers and renders stay internal until the NVIDIA outreach
  (brainstorm §8, "awaiting reply" as of 2026-08-25) resolves. Internal regression use is fine.
- **Local box.** No `uv`, no `nvcc`, 4 GB laptop GPU, ~92 GB free disk. Nothing in phase 1 can run
  locally; do NOT download the 68 GB weights here.

## The model's input contract (read from source, this repo)

Per instant `t0`, `load_physical_aiavdataset()` (`src/alpamayo2_super/load_physical_aiavdataset.py`,
docstring lines 32-75 lists every key) produces one dict:

| key | shape | note |
|---|---|---|
| `image_frames` | (7 cams, 4 frames, 3, H, W) uint8 | frames at t0-0.3/-0.2/-0.1/0 s; 1920x1080 f-theta native |
| `camera_indices` / `camera_names` | (7,) | MUST be the canonical ring in order 0..6 (`input_profiles.validate_source_camera_ring`) |
| `absolute_timestamps` / `relative_timestamps` | (7, 4) | per camera per frame, µs clip-relative / s from min |
| `ego_history_xyz` / `_rot` | (1,1,16,3) / (1,1,16,3,3) | 1.6 s @ 10 Hz ending at t0, in the ego frame at t0 |
| `ego_future_xyz` / `_rot` | (1,1,64,3) / (1,1,64,3,3) | 6.4 s GT future, **scoring only** |
| `ego_t0`, `ego_t0_frame_idx`, `camera_tmin`, `*_tvals`, `ego_t0_xyz`, `ego_t0_inv_quat` | | bookkeeping, derived |
| `camera_calibrations` | dict | optional, visualization overlays only |

Then, unchanged from NVIDIA's path: `select_task_input(data, "trajectory")` picks cameras
`[0,1,2,3,5,6]` (drops rear-tele) -> `helper.prepare_model_inputs` -> `model.sample_trajectories_from_data`
(`num_traj_samples`, `diffusion_kwargs={"inference_step": N}`, `torch.cuda.manual_seed_all(seed)` first).

**Verified: the GT future is NOT an inference input.** `prepare_model_inputs` forwards only history
(`helper.py:98-102`); `sample_trajectories_from_data` builds `traj_data` from history only
(`models/alpamayo2_super.py:268-270`); `fuse_traj_tokens` guards the future with an `if`. The future
is required only by training and the auto-labeling text task. Keep it in the export when available
(needed for the three-way readout), `null` otherwise.

**Hard inputs the export must carry besides images:** ego pose history (not an annotation — it is
the recorded trajectory, present in the MCAP / `egomotion.offline`; the same trajectory serves both
arms since renders are at recorded poses), and per-camera frame timestamps (PAI-AV cameras are not
synchronized). 10 Hz curated bundles match the model's 0.1 s frame spacing.

## Corrections to the first sketch

1. Do not modify NVIDIA's package. Add a sibling package `pd_alpamayo_eval` with
   `load_pdncc_export(manifest, clip_id, t0_us)` returning the same dict; the other three calls run
   untouched. ~300 lines. Export the full 7-camera ring; the profile selection stays NVIDIA's.
2. Fork into a private `parallel-domain/alpamayo-eval` (name TBD by user), NVIDIA code vendored
   under `third_party/alpamayo2/` or kept at root; do not branch inside the NVLabs clone.
3. Identical preprocessing in both arms (brainstorm hard rule): same JPEG path, same resolution
   (original 1920x1080 f-theta framing — the KB-fit replica renders in the original distorted model),
   same slot mapping. Save renders as JPEG q95 like the MCAP frames, not PNG.
4. Do not reuse `pd_fm.launch.ray`: it rewrites the command for pd-fm's Hydra CLI. Write a small
   submitter instead (below).

## Infra risks, in order

1. **flash-attn builds from source.** `uv.lock` pins `flash-attn==2.8.3` from the PyPI sdist ->
   needs `nvcc`, 30-120 min. Unknown whether the Ray worker image (`image: null` = node base env)
   has nvcc; pd-fm avoids it everywhere. Fixes: point the lock at the prebuilt GitHub-release wheel
   (torch 2.8 / cu12 / py3.12 / cxx11abi), or add `cuda-nvcc` via pixi. Build once; cache the venv.
2. **68 GB weights, HF-gated.** Nebius `/tmp` is a per-POD emptyDir (lost on recycle). Download
   once to the persistent virtiofs `/mnt/pdfm-data/` (2 TiB, mounted on both H100 pools; pd-fm
   uses `/mnt/pdfm-data/pdfm-data-cache`) and pass the directory as `--model-id`
   (`inference_smoke.validate_model_id` checks the layout). Then the HF token never has to ride in
   a job env again — Ray's dashboard serves `runtime_env.env_vars` UNAUTHENTICATED, so any
   forwarded token/key is exposed (same caveat pd-fm documents for the Nebius S3 key).
3. **Private-repo clone from the pod.** pd-fm pods clone over SSH (`_remote.to_ssh_url`), so some
   deploy credential exists; whether it covers a new repo is unknown. Ray `--working-dir` upload
   (pattern: `pd-fm/scripts/submit_mirror_job.py`) sidesteps it for iteration.
4. **Cold start.** Autoscale ~2 min + 7.5 GB image pull ~3.5 min; `RAY_JOB_START_TIMEOUT_SECONDS`
   900 s kills a job still PENDING at ~15 min (pd-fm `docs/clusters.md`). Resubmit onto a warm node.
5. **Results retrieval** needs a Nebius S3 key (`pd-nebius-ray`, service account in project
   `project-e00ate7dpr00brrpem4zqa`, scope `storage.object-editor`) or the job log. The user was
   invited to the tenant; key creation was in progress at the time of writing. No `nebius` AWS
   profile exists yet on this box (`~/.aws/config` has only `default` = AWS SSO).

## Phase 1 results (2026-10-12, jobs `alpamayo-probe-f5d8ec4f`, `alpamayo-smoke-16f38bee`)

Files: `pixi.toml` (+ lock), `scripts/submit_nebius.py`, `scripts/nebius_entrypoint.sh`,
`pd_alpamayo_eval/smoke.py`. Submit with `pixi run -e launch submit-nebius {probe|smoke|exec} --wait`.
Repo goes up via Ray `working_dir` (2.3 MB), no clone on the pod.

| what | measured |
|---|---|
| worker image | has `pixi`, `uv`, `git` **and** `nvcc` (`/usr/local/cuda`); driver 580; runs as uid 1000 `ray` |
| `/mnt/pdfm-data` | mount root is root-owned, NOT writable; `/mnt/pdfm-data/pdfm-data-cache` is world-writable -> everything lives under `pdfm-data-cache/alpamayo-eval/` |
| env build, cold | 112 s (`uv sync --locked`, flash-attn from the prebuilt cu12/torch2.8/cxx11abiTRUE wheel, no compile) |
| env build, warm pod | 1 s (venv on the pod's `/tmp` survives between jobs on the same pod) |
| weights | 67 GB in 51 s to `pdfm-data-cache/alpamayo-eval/models/alpamayo2-super`; persisted |
| model load | 28.5 s from virtiofs, 66.71 GiB VRAM after load (bf16, `device_map=cuda:0`) |
| cold start to running | ~5 min (first probe: PENDING -> RUNNING on a fresh `gpuh100` pod) |
| HF access | the pod env carries a cluster `HF_TOKEN` that can read the model but is **403 on the PhysicalAI-AV dataset** |

So risks 1-4 are closed. The remaining blocker is dataset access: the smoke died in
`PhysicalAIAVDatasetInterface()` with `GatedRepoError`. Fix = forward the `harsandor` token once
(`--forward-hf-token`); the entrypoint persists it at `pdfm-data-cache/alpamayo-eval/hf-cache/token`
and unsets the pod's `HF_TOKEN` at runtime so the file governs. Claude cannot submit the
token-bearing job (auto-mode classifier); the user runs it. Inference timing and minADE are still
to be measured (next run).

## Phase 2 state (2026-10-13): PD data path, no HF dataset needed

Decision (user): run on our own frames, never on PhysicalAI-AV via HF. Data moves like pd-fm's:

1. `pd_alpamayo_eval/export_from_pdncc.py --bundle-id <uid> --name <name> --start-s --duration-s`
   (pdncc pixi env) selects instants (ego-pose grid, 4 context frames x 7 cameras per arm, 10 Hz)
   and writes `<replipair artifact>/tmp_alpamayo_eval/<name>/` in the artifact bucket by
   server-side copy. Arms: `real` (= replipair `original/`), `rendered`, optional
   `rendered_<variation>`. Format: `pd_alpamayo_eval/EXPORT_FORMAT.md`; loader
   `pd_alpamayo_eval/load_export.py` (tests: `pixi run -e dev test-pd`).
2. `pixi run -e launch mirror-export <artifact export uri> s3://pd-nebius-ray/alpamayo-eval/<name>`
   copies once from the laptop (AWS `default` profile -> local `nebius` profile; one egress).
3. `pixi run -e launch submit-nebius run --wait --forward-nebius-credentials -- --export
   s3://pd-nebius-ray/alpamayo-eval/<name> --save-viz` (user runs it: carries the Nebius key, the
   `pdfm-ray-shell-forward` key of SA `rayio-storage`, as every pd-fm Nebius job does). The worker
   stages the export to `/mnt/pdfm-data/pdfm-data-cache/alpamayo-eval/exports/<name>` once, runs
   every instant x arm with one seed (common random numbers), writes results to
   `s3://pd-nebius-ray/alpamayo-eval/<name>/runs/<run id>/` (results.jsonl, per-instant json+npz,
   summary.json, PNGs). `mirror-export` in reverse puts a run next to the export in the artifact.

Never ship data through Ray `working_dir` (user decision 2026-10-13; one such package was deleted).

First export: bundle `ce48f271-05b1-4f04-a1d1-af1472473ed9` (nightly Italy dynamic replica,
replipair re-rendered with `pd_alpamayo_eval/replipair_step_alpamayo.json`: all 30 fps frames,
original f-theta, both arms), window 5-8 s -> `w5s_3s`, 30 instants, 462 frames, 192 MB, in both
buckets.

**First run `alpamayo-run-0d74dd90` (2026-10-13), 30 instants x {real, rendered}, 1 sample, seed 42,
10 diffusion steps, 0 errors.** Results: `s3://pd-nebius-ray/alpamayo-eval/w5s_3s/runs/alpamayo-run-0d74dd90/`.

| measured | value |
|---|---|
| inference per instant (CoT + 10-step expert) | 1.73 s median; CoT is one sentence (~79 chars), e.g. "Keep distance to the lead vehicle because traffic ahead is moving slowly." |
| peak VRAM | 69.0 GiB |
| staging 463 files Nebius -> volume | 21 s (once) |
| minADE vs GT, real | median 1.12 m, p90 1.62 m |
| minADE vs GT, rendered | median 0.69 m, p90 2.25 m |
| paired ADE real vs rendered (same seed) | median 0.61 m, p90 1.43 m |
| identical CoT text in both arms | 18 / 30 instants |

Reading: ego frame, timestamps and image path are right (predictions overlap GT in direction and
scale on the PNGs; a convention bug would show as ~10 m). The scene is stop-and-go behind a lead
car; the biggest arm disagreement (t0 1700000005807012000: real 1.6 m, rendered 3.8 m) is the
rendered arm saying "Resume speed from stop" where the real arm keeps distance. A full ~20 s scene
costs ~10 min of H100, so sampling is cheap. **Nothing is interpretable yet without the
seed-to-seed noise floor** (same images, different seeds): the paired 0.6 m is the same size as
each arm's own GT error. Next runs: (1) noise floor, real arm, seeds 0..4 or `--num-traj-samples`;
(2) blind test (wrong/blank images, same history); (3) whole scene.

**Full scene with weather variations, run `alpamayo-run-27f93c3e` (2026-10-13).** Bundle
`ceb8e94d-de22-4558-be1d-d7da397e75f0` (replipair re-rendered with `render_variations: true`),
export `full_variations` (174 instants x {real, rendered, rendered_rainy, rendered_snowy}, 4956
frames, 1.9 GB), 1 sample, seed 42, 0 errors, ~55 min wall (19 s/instant incl. 4 PNGs; inference
1.75 s). Results: `s3://pd-nebius-ray/alpamayo-eval/full_variations/runs/alpamayo-run-27f93c3e/`.

| arm | minADE vs GT median / p90 (n=121 with future) | paired ADE vs real median / p90 (n=174) | CoT sentence differs from real |
|---|---|---|---|
| real | 0.86 / 2.47 m | - | - |
| rendered | 1.33 / 5.96 m | 1.14 / 5.78 m | 121 / 174 |
| rendered_rainy | 1.64 / 6.25 m | 1.73 / 5.93 m | 155 / 174 |
| rendered_snowy | 1.17 / 5.64 m | 1.34 / 6.03 m | 159 / 174 |

Where it diverges (median paired ADE per 2 s bin): 0-8 s all arms within ~1 m of real; **8-10 s
rendered 2.4 m, rainy 5.9 m, snowy 4.7 m**; 12-18 s 1.7-3.8 m for the variations, up to 5.4 m
(snowy, 16-18 s, "yellow traffic light ahead"). The large disagreements are bimodal: at t=+8..9 s
the real arm is 0.3-0.4 m from GT ("keep distance ... moving slowly") while rendered is 6.3-6.4 m
off (same keep-distance text but a go trajectory), and snowy sits at 0.6 m on some of the same
instants. I.e. the model flips between a stop and a go mode; renders flip it more often. The
sentence-level text diff overstates disagreement (85 distinct sentences, mostly paraphrases of
keep-distance); a meta-action-level comparison is needed.

**Directional finding: on renders the model drives faster.** Distance covered by the predicted
6.4 s trajectory, median over 174 instants: real 23.4 m (GT 21.2 m), rendered 24.9 m, rainy
30.6 m, snowy 30.2 m. Per instant, rendered is longer than real by +2.6 m median (longer on 68 %
of instants, shorter on 13 %), rainy +4.3 m (75 % / 14 %), snowy +1.8 m (59 % / 23 %). The
instants with 6-9 m paired ADE are all of this kind (e.g. t=+0.0 s: real stops at 15 m with GT at
13 m, all renders run to 40 m while their text still says "slow down due to the lead vehicle").
A symmetric seed-noise floor cannot produce a one-sided bias, so this is a real replica effect on
the model: the lead vehicle in the renders reads as further away / less constraining. Video:
`outputs/full_variations_27f93c3e.mp4` (make-video on the run).

**Seed-to-seed noise floor, run `alpamayo-run-af7ff7bc` (real arm, seed 1, same export).**
`pixi run -e dev compare-runs <run 27f93c3e> <run af7ff7bc>`, reference real@seed42, 174 instants:

| column | paired ADE med / p90 | predicted-distance diff med | longer / shorter (>1 m) | ADE vs GT med | CoT sentence differs |
|---|---|---|---|---|---|
| real@seed1 (floor) | 0.95 / 2.27 m | -0.01 m | 32 % / 36 % | 1.13 m | 72 % |
| rendered@seed42 | 1.14 / 5.78 m | +2.60 m | 68 % / 13 % | 1.33 m | 70 % |
| rendered_rainy@seed42 | 1.73 / 5.93 m | +4.34 m | 75 % / 14 % | 1.64 m | 89 % |
| rendered_snowy@seed42 | 1.34 / 6.03 m | +1.75 m | 59 % / 23 % | 1.17 m | 91 % |

Reading: (1) the median paired ADE of the rendered arm (1.14 m) is only 20 % above the seed floor
(0.95 m), i.e. on a typical instant the replica changes the trajectory about as much as the
diffusion noise does; (2) the tail is the replica effect: p90 5.8-6.0 m vs a 2.3 m floor, 2.5x;
(3) the predicted-distance bias is the cleanest discriminator: seeds are symmetric (32/36 %,
-0.01 m), renders are one-sided (+1.8 to +4.3 m, 59-75 % longer); (4) the CoT sentence differs
on 72 % of instants between two seeds of the SAME images, so sentence-level text agreement is
sampling noise and useless as a metric; meta-actions are required. Per-scene gate candidates:
paired-ADE p90 relative to the seed floor, and the signed distance bias (floor: ~0, symmetric).

## Plan

**Phase 0 — cluster hello-world.** DONE (pd-fm bench on 1xH100).

**Phase 1 — NVIDIA's own smoke on the cluster.** No pdncc work. Settles risks 1-4.
1. Repo skeleton: vendored NVIDIA code, `pixi.toml` providing Python 3.12 + `uv` (+ `cuda-nvcc`
   fallback), empty `pd_alpamayo_eval` package.
2. Submitter (~100 lines): Ray Jobs API, `--working-dir` upload, `entrypoint_num_gpus=1`,
   `entrypoint_resources={"h100": 1}`, HF token as job env for the first run only.
3. Entrypoint: `uv sync --locked`; download weights to `/mnt/pdfm-data/alpamayo2-super/` if absent;
   `python -m alpamayo2_super.inference_smoke --manifest examples/validation_samples.json
   --sample-index 0 --save-viz ... --save-json ...` into `/mnt/pdfm-data/alpamayo-eval/smoke/`;
   print the JSON into the job log.
4. Record: env build time, weight load time, peak VRAM, seconds per instant, minADE on sample 0.

**Phase 2 — export contract + loader + parity test.**
- Manifest per brainstorm §6.10: `manifest.json` (rig: per camera K/distortion/size/cam_to_ego;
  clips: instant timestamps, ego history 16x(t,xyz,R) in ego@t0, gt_future up to 64x, nav_text|null,
  per-camera frame stems), `real/<clip>/<cam>/<stem>.jpg`, `rendered/<clip>/<cam>/<stem>.jpg`.
- `load_pdncc_export()` -> same dict as `load_physical_aiavdataset()`.
- **Parity test** (the one test that matters): same PAI-AV clip + t0, real frames from our MCAP
  export vs frames streamed from HF, same seed -> trajectories should nearly coincide. Checks
  timestamps, ego-frame convention and image handling at once.

**Phase 3 — runner + scorer.**
- Runner loops instants x {real, rendered}, fixed seeds (common random numbers), N samples; writes
  JSON/npz per instant to `s3://pd-nebius-ray/alpamayo-eval/<run>/`. 3 scenes x ~60 instants x 2
  arms ≈ 360 calls ≈ 1-2 h on one H100.
- Scorer offline on CPU: ADE/FDE at 1/2/3/6.4 s (real vs render, each vs GT), CoC text diff,
  meta-action agreement; per-scene median/p90; blind test (wrong images, same history).
- Later: `nav_text` CFG on the 8-GPU node, companion probes, pdncc step registering the artifact.

## Open asks (from the user / colleagues)

- Repo name + whether to create `parallel-domain/alpamayo-eval` now or start under
  `~/workspace/alpamayo-eval` and push later.
- Nebius S3 key (and store it as a separate `nebius` AWS profile, never in `default`).
- Does the cluster's git credential cover a new repo? (Ask whoever set up Nebius Ray; or stay on
  `--working-dir`.)
- Jannik: which replicas exist beyond the three; were they rendered at every 10 Hz pose or only
  `max_eval_frames=150`; export path for all 7 cameras at original framing.
- NVIDIA outreach status (gates anything customer-facing, not internal use).
- Permission to submit the first phase-1 job (holds one H100 ~30-90 min on the first run).

## Pointers

- Brainstorm + decisions: `~/workspace/pd-neural-content-creation/ALPAMAYO_EVAL_BRAINSTORM.md`
- Conversion audit / handover: `~/workspace/pai_av_handover/{README,CONVERSION_AUDIT}.md`
- PAI-AV download CLI (own pixi env): `~/workspace/pai_av/parallel_domain_scripts/scenes.py`
- Cluster rules: `~/workspace/pd-fm/docs/clusters.md`, launcher internals
  `pd_fm/launch/{ray.py,_remote.py}`, lightweight submitter `scripts/submit_mirror_job.py`,
  Nebius launcher config `configs/launcher/ray_nebius.yaml` (resources tag, `/mnt/pdfm-data`).
- This repo: `README.md` (CLI, env, two-GPU demo), `src/alpamayo2_super/{load_physical_aiavdataset,
  input_profiles,helper,inference_smoke}.py`, `pyproject.toml` + `uv.lock` (torch 2.8.0,
  transformers 4.57.1, flash-attn 2.8.3 sdist, physical-ai-av 0.2.2, Python 3.12 only).

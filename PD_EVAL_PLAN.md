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

> **Path change 2026-10-26.** Every `/mnt/pdfm-data/pdfm-data-cache/alpamayo-eval` path below is
> historical: the weights, HF cache and token were moved to the `pd-general` shared filesystem,
> `/mnt/pd-general/data/alpamayo-eval` (entrypoint default `PERSIST`), at the OI team's request; the
> pd-fm filesystem is deleted after the move. Nothing else ever lived on a shared filesystem.

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

## TODO (2026-10-19)

Where things stand: three scenes run end to end (Italy with a seed floor, Madrid, Austria), the
improved replicas of all three re-run and compared (Phase 2, "round 2": no scene moved closer to real; seed floors for all three scenes show paired
ADE cannot separate render from seed, the signed speed bias can), the dashboard and video work, everything generated is listed in
`pd_alpamayo_eval/INVENTORY.md`.

1. **Seed floors: done for all three scenes** (Italy 0.95 / 2.27 m, Madrid 2.29 / 7.65 m, Austria
   1.39 / 3.69 m; runs `af7ff7bc`, `e6d4f80e`, `b357ae61`). Paired ADE of the renders is at or below
   the floor on Madrid and Austria and only marginally above it on Italy; the signed predicted-distance
   bias is the metric that separates render from seed (one-sided on Italy and Austria, symmetric for
   seeds). Next: make the gate proposal (item 5) around the signed bias + a multi-sample ADE.
2. **Meta-action text metric**: run the model's `meta_action` task (longitudinal / lateral / lane
   labels) per instant and arm; sentence-level text comparison is pure sampling noise (72 %
   differs between two seeds of the same images). Needs a small addition to `run_export.py`
   (`text_tasks.generate_text`).
3. **Blind test** (brainstorm hard requirement): same ego history, wrong/blank images; the model
   must change its output materially or it is not measuring the pictures.
3b. **Attention heatmaps and camera occlusion** (implemented 2026-10-20/21, `attention.py`,
   `ablation.py`): done for Madrid (attention with every layer `eedc9a2d`, occlusion `d931ddaf`): the
   red traffic light is not reproduced by the render; the model reads it mainly from the front-wide
   camera; restoring that camera removes 24 of 28 flips (see Phase 2). Next: the attention run for
   Italy and Austria (`italy_full_variations_v2`, `austria_full_variations_v2`; 35 min each) as the
   screening tool for what makes the model drive longer / shorter there, then occlusion on the
   camera the maps point at. Before more occlusion runs: a prefetching loader (the 13 forwards per
   instant are CPU-bound).
4. **More samples per instant** (`--num-traj-samples 5`): minADE over samples and a per-instant
   spread, cheaper than many seeds.
5. **Per-scene gate proposal**: paired-ADE p90 relative to the seed floor, and the signed
   predicted-distance bias (floor symmetric, renders one-sided). Write it up once 1 is done.
6. **More scenes**: any bundle with a dynamic splat works with the same three commands (render,
   export, mirror); candidates in the registry: other PAI-AV scenes with 10 Hz curated bundles.
7. **Housekeeping**: delete the leftover `tmp_alpamayo_eval/italy_3s/` in artifact `ee58e53a`
   (duplicate of `w5s_3s`); the `full` export (Italy without variations) is superseded by
   `full_variations`. Nothing else is redundant.
8. **Not started**: `nav_text` CFG on the 8-GPU node, companion probes, pdncc step that registers
   the export as an artifact, NVIDIA outreach status for anything customer-facing.

**pdncc integration (2026-10-23): the handoff for implementing this as a pipeline step
(`BehaviourEvalStep`) and a `ncc bundle visualize` tab is `pd_alpamayo_eval/PDNCC_INTEGRATION.md`;
nothing has been implemented in pdncc yet.**

**Step-by-step recipe for all of this: `pd_alpamayo_eval/HOWTO.md`. Before submitting anything,
check the experiment log in `pd_alpamayo_eval/INVENTORY.md`: every (scene, replica version, arms,
seed) combination already run is listed there with its run id.**

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

**ADE by horizon (Italy, 2026-10-15; `compare-runs` now prints this; the dashboard has a horizon
selector).** Paired ADE vs real, median / p90 in m:

| column | 1 s | 2 s | 3 s | 6.4 s | weighted exp(-t/2 s) |
|---|---|---|---|---|---|
| real@seed1 (floor) | 0.05 / 0.13 | 0.21 / 0.42 | 0.38 / 0.82 | 0.95 / 2.27 | 0.44 / 1.04 |
| rendered | 0.04 / 0.14 | 0.13 / 0.62 | 0.29 / 1.59 | 1.14 / 5.78 | 0.51 / 2.34 |
| rendered_rainy | 0.05 / 0.17 | 0.20 / 0.77 | 0.40 / 1.92 | 1.73 / 5.93 | 0.65 / 2.63 |
| rendered_snowy | 0.06 / 0.17 | 0.22 / 0.72 | 0.41 / 1.73 | 1.34 / 6.03 | 0.63 / 2.59 |

Reading: the first second is identical in every arm (5 cm), i.e. it is determined by the ego
history, not the pictures; a short-horizon metric cannot see the replica and would also pass the
blind test trivially. The render effect opens from 3 s (p90 2x the floor) and is largest at 6.4 s
(2.5x). Report per horizon; if one number is wanted, the time-weighted ADE (tau = 2 s) keeps the
tail visible (p90 2.3 m vs floor 1.0 m) without being dominated by it.

**Two more scenes (2026-10-14), same recipe, both with rainy + snowy.** Madrid intersection: bundle
`ea732026-8a79-474e-9bf4-70b2f6438256`, export `madrid_full_variations` (185 instants), run
`alpamayo-run-e368b20f`. Austria roundabout: bundle `890838c3-7aa1-4278-b8e4-e4040b0688fb`,
export `austria_full_variations` (186 instants), run `alpamayo-run-17c45f0a`. Both seed 42,
1 sample, no PNGs (~35 min each, 11 s/instant for 4 arms). `compare-runs`, reference real@seed42:

| scene / column | paired ADE med / p90 | distance diff med | longer / shorter | ADE vs GT med (real) |
|---|---|---|---|---|
| Madrid rendered | 2.38 / 7.65 m | +0.41 m | 48 % / 39 % | 3.27 m (real 2.80 m) |
| Madrid rainy | 2.44 / 7.60 m | +0.12 m | 45 % / 43 % | 3.08 m |
| Madrid snowy | 2.32 / 7.40 m | -1.04 m | 38 % / 51 % | 2.55 m |
| Austria rendered | 0.73 / 4.01 m | -1.01 m | 16 % / 50 % | 2.56 m (real 1.99 m) |
| Austria rainy | 1.04 / 3.78 m | -0.91 m | 21 % / 48 % | 2.13 m |
| Austria snowy | 1.31 / 4.39 m | -1.45 m | 20 % / 52 % | 2.71 m |

Reading across the three scenes: (1) the model itself is less competent on the intersection and
roundabout than on the tram street (real-arm ADE vs GT 2.8 m / 2.0 m vs 0.86 m), so paired
numbers must be read against a per-scene seed floor, not Italy's 0.95 m; (2) the **direction of
the speed shift is scene-dependent**: Italy renders made the model drive longer (+2.6 m, 68 %
longer), Austria renders make it drive shorter (-1.0 m, 50 % shorter vs 16 % longer), Madrid is
symmetric; so "renders bias speed" is the general statement, "faster" was Italy-specific;
(3) the tails are large everywhere (p90 4-7.7 m). Videos: `outputs/madrid_full_variations_e368b20f.mp4`,
`outputs/austria_full_variations_17c45f0a.mp4`. Next: seed-floor run (`--arms real --seed 1`) for
Madrid and Austria to size their paired numbers.

**Improved replicas, round 2 (2026-10-19).** All three replicas were re-trained with improved
quality and rendered with the same step config: Italy bundle
`f47e7a5d-bf47-4d9f-a540-07f158d28800` (artifact `281b3822`), Austria bundle
`d2fb85d9-2bd0-41bf-9db5-ec5251707da2` (artifact `3a3586e2`), Madrid bundle
`c6769a36-4bad-43f7-90d8-f49ed780992c` (artifact `7d262860`); exports `italy_full_variations_v2`,
`austria_full_variations_v2`, `madrid_full_variations_v2` (same instant grids as before); runs
`alpamayo-run-928b4727` (Italy), `alpamayo-run-9fc99df6` (Austria), `alpamayo-run-99bb841a`
(Madrid), seed 42, 1 sample, with PNGs. Two checks first:
(a) `real@new` vs `real@old` paired ADE is exactly 0.000 on every instant of all three scenes (same
images, same seed): the whole pipeline, from export to model to upload, is bit-reproducible across
jobs, so any difference below is the pictures; (b) the instant grids match, so old and new runs join
instant by instant (dashboard and `compare-runs` support this now, `--labels old new`).

How much the new renders moved the prediction, and whether they got closer to real
(`compare-runs`, seed floor from Italy: paired 0.95 / 2.27 m):

| scene / column | vs real: paired ADE med / p90 | distance diff | longer / shorter | vs rendered@old: paired ADE med / p90 | ADE vs GT med |
|---|---|---|---|---|---|
| Italy rendered old | 1.14 / 5.78 m | +2.60 m | 68 % / 13 % | (reference) | 1.33 m (real 0.86) |
| Italy rendered **new** | 1.18 / 5.55 m | +2.30 m | 70 % / 16 % | 0.38 / 1.85 m | 1.53 m |
| Italy rainy old / new | 1.73 / 5.93 → 1.71 / 5.31 m | +4.34 → +3.99 m | 75/14 → 70/14 % | 0.97 / 4.92 m | 1.64 → 1.87 m |
| Italy snowy old / new | 1.34 / 6.03 → 1.59 / 5.56 m | +1.75 → +2.16 m | 59/23 → 56/32 % | 1.37 / 6.46 m | 1.17 → 1.57 m |
| Austria rendered old | 0.73 / 4.01 m | -1.01 m | 16 % / 50 % | (reference) | 2.56 m (real 1.99) |
| Austria rendered **new** | 0.92 / 4.07 m | -1.12 m | 18 % / 53 % | 0.20 / 3.48 m | 2.80 m |
| Austria rainy old / new | 1.04 / 3.78 → 1.07 / 3.72 m | -0.91 → -0.69 m | 21/48 → 24/44 % | 0.48 / 3.58 m | 2.13 → 2.12 m |
| Austria snowy old / new | 1.31 / 4.39 → 1.27 / 4.55 m | -1.45 → -1.18 m | 20/52 → 19/54 % | 1.13 / 3.65 m | 2.71 → 3.13 m |
| Madrid rendered old | 2.38 / 7.65 m | +0.41 m | 48 % / 39 % | (reference) | 3.27 m (real 2.80) |
| Madrid rendered **new** | 2.08 / 7.08 m | +0.56 m | 49 % / 44 % | 0.48 / 6.00 m | 2.71 m |
| Madrid rainy old / new | 2.44 / 7.60 → 2.25 / 7.63 m | +0.12 → +0.08 m | 45/43 → 45/43 % | 0.83 / 6.26 m | 3.08 → 3.46 m |
| Madrid snowy old / new | 2.32 / 7.40 → 2.57 / 6.82 m | -1.04 → -0.96 m | 38/51 → 39/50 % | 1.20 / 6.62 m | 2.55 → 2.09 m |
| **Madrid seed floor** real@seed1 (run `alpamayo-run-e6d4f80e`) | 2.29 / 7.65 m | -2.90 m | 30 % / 55 % | 2.29 / 7.65 m | 1.73 m |
| **Austria seed floor** real@seed1 (run `alpamayo-run-b357ae61`; `99460be7` is a bit-identical duplicate) | 1.39 / 3.69 m | -0.17 m | 42 % / 37 % | 1.39 / 3.69 m | 1.62 m |

Reading: (1) with common random numbers the new clear-weather render changes the prediction by
only 0.38 m (Italy) / 0.20 m (Austria) / 0.48 m (Madrid) median, i.e. the model sees the two
renders as nearly the same scene on most instants; Madrid's tail is wider (p90 6.0 m vs 1.9 / 3.5 m),
so there the new render did flip a minority of instants; the weather variations changed more
(new rainy/snowy differ 0.5-1.7 m from the old ones) because the weather effects themselves were
re-rendered; (2) **the gap to real did not close on Italy and Austria**: paired ADE vs real is flat
on Italy (1.14 → 1.18 m) and slightly worse on Austria (0.73 → 0.92 m, p90 equal), ADE vs the
recorded future is 0.2 m worse on both, all within the seed-level noise of ~0.3 m on ADE-vs-GT;
**Madrid looked like a modest improvement** (paired ADE vs real 2.38 → 2.08 m median, p90 7.65 →
7.08 m, ADE vs the recorded future 3.27 → 2.71 m) **but the Madrid seed floor is 2.29 / 7.65 m**:
two seeds on the identical real images differ by as much as either render does, so on Madrid the
render effect and the 0.3 m "improvement" are both indistinguishable from sampling noise. The
Madrid floor also shows how bimodal the model is there: seed 1 drives 2.9 m shorter than seed 42 on
the median instant (55 % shorter by >1 m) and happens to land much closer to the recorded future
(1.73 m vs 2.80 m), i.e. at this intersection the seed decides stop-vs-go, and one trajectory sample
per instant cannot resolve it (TODO 4, several samples per instant, is the fix). **Austria's seed
floor is 1.39 / 3.69 m**, above every Austria render arm's paired ADE (0.7-1.3 m median, 3.7-4.6 m
p90): by ADE alone the Austria render effect is also inside the noise. (3) The signed distance test
is what separates the scenes. The seed rows are symmetric on Italy (-0.01 m, 32 % / 36 %) and
Austria (-0.17 m, 42 % / 37 %) and lopsided only on bimodal Madrid (-2.9 m), while the render rows
are one-sided on Italy (+2.3 to +2.6 m, ~70 % longer vs ~15 % shorter) and Austria (-1.0 to -1.5 m,
~50 % shorter vs ~18 % longer) in BOTH render versions. So on Italy and Austria the reconstruction
shifts the model's speed in a consistent direction that sampling noise never produces, the quality
pass did not change that, and ADE is the wrong instrument to see it (it is swamped by symmetric
sampling noise on every scene).

**Shift vs spread (2026-10-20).** Decomposing the 6.4 s predicted distance, arm minus real, per instant
(same seed): the per-instant spread is the sampling noise, the mean is the systematic render effect.

| scene, arm | mean shift | spread (std) | t | sign test p |
|---|---|---|---|---|
| Italy rendered old / new | +3.9 / +3.8 m | 5.8 / 6.0 m | +9.0 / +8.4 | 1e-14 / 1e-12 |
| Italy real seed 1 | -0.2 m | 3.6 m | -0.8 | n.s. |
| Austria rendered old / new | -1.9 / -1.9 m | 5.1 / 5.8 m | -5.0 / -4.5 | 1e-7 / 1e-7 |
| Austria real seed 1 | -0.2 m | 5.4 m | -0.6 | n.s. |
| Madrid rendered old / new | +2.6 / +2.0 m | 11.3 / 10.4 m | +3.1 / +2.7 | n.s. |
| Madrid real seed 1 | -5.8 m | 10.0 m | -7.9 | 2e-3 |

Per instant, noise dominates (spread > shift everywhere), which is why unsigned ADE reads "noise". Over
the clip the shift is unambiguous on Italy and Austria (t 5-9, identical in both render versions) and
absent for a seed re-roll (t < 1). Madrid's seed re-roll itself shifts the mean by 6 m (bimodal
stop-vs-go), so Madrid is unmeasurable with one sample per instant. A gate therefore tests the signed
shift over many instants (or many samples per instant), not the per-instant ADE.

**Attention maps, Madrid (2026-10-21, run `alpamayo-run-ed10a713`, predictions bit-identical to
`99bb841a` on all 740 records).** The diffusion expert attends straight into the VLM cache, so its
attention over the image patch tokens is where the trajectory decision looked
(`pd_alpamayo_eval/attention.py`; dashboard "model attention"). Facts from the 740 maps:

- The model sees each camera at ~504 x 280 px (10 x 18 tokens of 28 px); a traffic light is a few
  pixels. 59 % of the expert's attention is on the images, 30 % on the prompt tokens (ego history),
  3 % on its own sentence. Camera shares: cross-left 27 %, front-wide 24 %, cross-right 19 %,
  front-tele 15 %, rear 7-8 % each; the four frames of a camera get equal shares (no recency
  preference). Shares are the same for every arm, so the render does not change WHERE the model
  looks at the camera level.
- Raw maps are dominated by attention sinks: a few sky patches, identical on every instant (maps
  correlate 0.75 between instants, late layers put half the attention into 5 % of the cells). Only
  the map minus the scene average (or minus the real arm) shows content; the dashboard defaults to
  that.
- **The red traffic light is the whole Madrid story.** The real arm's sentence says stop on 54
  instants (the car waits at a red light for the first ~5 s and again at ~6.4 s), the rendered arm's on
  27: on 28 instants the render makes the model "accelerate to proceed through the intersection since
  the traffic light turns green" where the real frames give "stop due to the red traffic light", and
  the trajectory follows (these are the stop-vs-go flips behind the 7-8 m ADE tails). In the rendered
  front-tele frames the light's red lamp is not reproduced (it reads dark/greenish). The attention
  shows the mechanism: on real frames the expert's excess attention sits on the traffic-light block
  of the tele camera when it says stop (1.13 x the scene average there, 30 % of the frame's excess
  attention at the flip instants, 6 instants above 1.5 x); on rendered frames that block is never
  above 1.5 x, gets 14 % of the excess attention at the flips, and is lower than real on 75 % of
  them. So on Madrid the replica effect is not "noise": it is one small rendering defect (traffic
  light state) that removes the cue the model uses to decide, which single-sample ADE could not
  separate from the seed noise because the model is bimodal exactly where that cue matters.
- For the reconstruction team: emissive small objects (traffic lights, brake lights) must keep
  their colour state in the dynamic replica; this is a checkable, scene-independent requirement, and
  the attention map makes it a measurable one (attention on the light block, real vs rendered).
- **Limits of the heatmaps (same day).** Looked at freely across the clip, the bright cells sit on
  sky and foliage patches that move with the content, in every arm and every mode (minus scene
  average, z-score, ratio, sink-masked all tried): the deep expert layers read global summaries the
  LLM parks on low-information "register" tokens, so the maps show where the expert reads, not what
  visual evidence it uses. The traffic-light signal above is real but was found by knowing where to
  look. Two follow-ups implemented: `--attention-per-layer` (diagnostic run `alpamayo-run-bd4af965`,
  2 instants, all 64 layers, to see whether early layers are spatially localised) and **camera
  occlusion** (`--ablate blank real`: one extra forward per camera with its frames blanked or
  swapped back to real; causal, sink-free; dashboard "occlusion" views). The real-swap run on Madrid
  is the direct test of the traffic-light hypothesis: if swapping only the front-tele frames to real
  removes the stop-vs-go flips, the defect is proven to be that camera's render.
- **Per-layer result (run `alpamayo-run-eedc9a2d`, every layer stored, same day).** The sink
  problem is a depth problem: the last quarter of the expert's layers puts 51 % of a frame's attention
  into its top 5 % cells, the early and middle bands 10 %. In the middle band (layers 16-47) the
  maps sit on content: lead cars and road in the wide camera, and in the real front-tele frames the
  red traffic light is the brightest spot at every stop-vs-go flip instant, while the rendered tele
  frame of the same instant (light drawn dark/greenish) has no attention on it. Numbers, middle band:
  real arm 1.22 x the scene average on the light block when it says stop vs 0.96 x when it says go, 7
  instants above 1.5 x; every render arm ~1.0 x always, never above 1.5 x; at the 28 flip instants
  real 1.23 x vs rendered 1.05 x (rendered lower on 79 %), 31 % vs 16 % of the frame's excess
  attention. Dashboard: `--scene madrid`, mode "minus scene average", layers "mid layers" (default),
  cameras "front tele", first 7 s.

**Camera occlusion, Madrid (2026-10-21, run `alpamayo-run-d931ddaf`, real + rendered, 185 instants,
13 forwards per instant, main predictions bit-identical to the reference).** The causal readout,
no model internals:

| ADE shift when a camera is blanked (m) | cross left | front wide | cross right | rear left | rear right | front tele |
|---|---|---|---|---|---|---|
| real, median / mean | 0.19 / 0.86 | **1.15 / 2.57** | 0.12 / 0.81 | 0.06 / 0.31 | 0.10 / 0.38 | **1.18 / 2.45** |
| real, sentence changes | 17 % | **89 %** | 13 % | 4 % | 7 % | 42 % |
| rendered, median / mean | 0.20 / 1.28 | **2.32 / 3.03** | 0.14 / 1.00 | 0.07 / 0.44 | 0.07 / 0.53 | **1.37 / 2.79** |

| rendered arm, ONE camera swapped back to real | cross left | front wide | cross right | rear left | rear right | front tele |
|---|---|---|---|---|---|---|
| gap to the real prediction after the swap, median (before 2.08 m) | 2.33 | **0.67** | 1.85 | 2.02 | 2.13 | **1.11** |
| share of the mean gap removed | 1 % | **41 %** | 3 % | -1 % | -3 % | **22 %** |
| of the 28 stop-vs-go flips, now says stop | 9 | **24** | 4 | 1 | 2 | **14** |
| (blanking that camera of the REAL input makes real say go on) | 0 | **17** | 0 | 0 | 0 | 4 |

Reading: (1) the model drives by the two front cameras; the rear cameras are irrelevant (blanking
them moves the prediction < 0.1 m) and the cross cameras matter only in the tail; (2) **the Madrid
replica effect sits in the front-wide render first, the front-tele render second**: restoring only
the front-wide frames to real removes 41 % of the whole gap and turns 24 of the 28 red-light flips
back into "stop", the tele 14 of 28, every other camera nothing; (3) the attention maps pointed at
the tele camera's light because it is large there, but the causal test says the model reads the
light state mainly from the wide camera (blanking it on real frames makes the model run the red
light on 17 of the 28 instants). Both renders show the same defect (light not red), so the two
cameras agree. The occlusion run is the slow one (75 s per instant: 13 forwards each with CPU
preprocessing), so it is a per-hypothesis tool, not a per-scene default; the attention run (2 s per
record) stays the screening tool and the occlusion run the proof.

**Madrid v3 (2026-10-22, bundle `b96107b7-1ddb-4925-8c09-bb3c1b6b90b2`, artifact `3fedc926`, export
`madrid_full_v3`, run `alpamayo-run-405e3f8a`, real + rendered, no weather arms).** The replica was
re-trained with the traffic-light finding in hand. Checked first with a 10-minute `quick` run on five v2
flip instants (`alpamayo-run-f82958fd`): the rendered tele frame now shows the red lamp, the attention
sits on it as on the real frame, and the rendered sentence says "stop due to the red traffic light"
(paired ADE 0.14 m where v2 had 1.70 m). Full clip, reference = real (bit-identical to v2's real arm):

| Madrid, clear render | paired ADE vs real, median / p90 | distance bias | stop-vs-go flips (real stop, render go) | red-light instants (n=9): paired ADE median, share > 1 m |
|---|---|---|---|---|
| v2 (`eedc9a2d`) | 2.08 / 7.08 m | +0.6 m, 49 / 44 % | 28 of 185 | 1.95 m, 89 % |
| **v3** (`405e3f8a`) | **1.04 / 6.75 m** | -0.7 m, 25 / 46 % | **15 of 185** | **0.14 m, 22 %** |
| seed floor (`e6d4f80e`) | 2.29 / 7.65 m | -2.9 m | (seed) | |

Reading: (1) the median paired ADE halved and is now well under the seed floor (1.04 vs 2.29 m): on
the typical instant the v3 render is closer to the real prediction than a second seed is; (2) the
red-light instants are fixed outright (0.14 m median, one instant above 1 m); (3) the tail is not
gone (p90 6.75 m) and 15 flips remain, now at the green phase ("resume" vs "accelerate" while the
lead car moves), which is the model's own stop-vs-go bimodality at a green light, the same thing the
seed floor shows; (4) the distance bias flipped sign to slightly shorter (46 % shorter vs 25 %
longer), to be watched with weather arms when they exist for v3. First case in this project of a
replica change producing a measurable, explained behavioural improvement; the attention screening
plus occlusion found the defect, the quick run confirmed the fix in ten minutes before the full run.
 For the eye: the
sentence differs from the old render on 36-42 % of instants (vs 70 % between seeds), consistent
with (1). Videos: `outputs/italy_full_variations_v2_928b4727.mp4`,
`outputs/austria_full_variations_v2_9fc99df6.mp4`, `outputs/madrid_full_variations_v2_99bb841a.mp4`;
old and new side by side on the same instant:
`pixi run -e dev dashboard <old run> <new run> --labels '' v2`.

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

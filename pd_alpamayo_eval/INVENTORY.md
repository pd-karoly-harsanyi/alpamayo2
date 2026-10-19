# Inventory of generated data (as of 2026-10-19)

Everything below exists and can be reused without rerunning anything. Paths are exact.
Recipe for producing more: `HOWTO.md`. Findings: `../PD_EVAL_PLAN.md`.

## Scenes, bundles, artifacts (pdncc registry, AWS bucket `paralleldomain-pipeline`)

All three are PhysicalAI-AV val-split clips (20 s, 7 cameras, 10 Hz ego poses in the MCAP),
reconstructed as dynamic replicas. The bundle is what `ncc step run` produced after re-rendering
the replipair with `replipair_step_alpamayo.json` (every 30 fps frame, native f-theta, rainy+snowy).

| scene | session id | bundle (location_artifact_id) | replipair artifact (`replica_replipair_pdncc`) | original data (`replica_original_data`) |
|---|---|---|---|---|
| Italy tram street | `pai_av_italy_tram_street_c3608270` | `ceb8e94d-de22-4558-be1d-d7da397e75f0` | `5c833e25-58bc-4ab6-ae0f-1065c071c55e` | `99317319-d464-4efe-b17c-aa5f686f4f99` |
| Italy tram street (first render, no variations) | same | `ce48f271-05b1-4f04-a1d1-af1472473ed9` | `ee58e53a-765c-4684-ae80-358ca17f5265` | `99317319-d464-4efe-b17c-aa5f686f4f99` |
| Madrid intersection | `pai_av_madrid_intersection_82982ec0` | `ea732026-8a79-474e-9bf4-70b2f6438256` | `062b2b02-9ddc-4d2b-8a95-fed4fbaafc90` | `28b5b860-5609-4926-ae43-d83f0ca3a4ec` |
| Austria roundabout | `pai_av_austria_roundabout_6861e348` | `890838c3-7aa1-4278-b8e4-e4040b0688fb` | `3671dc87-0693-4b5c-a63b-1ea3f280faa7` | `8d81eb21-bd08-40d6-98c1-67e376f74fa4` |
| Italy tram street, **v2** (improved replica, 2026-10-19) | same as Italy | `f47e7a5d-bf47-4d9f-a540-07f158d28800` | `281b3822-cb83-4235-8418-a0aa7c892345` | same |
| Austria roundabout, **v2** (improved replica, 2026-10-19) | same as Austria | `d2fb85d9-2bd0-41bf-9db5-ec5251707da2` | `3a3586e2-5353-4287-b93e-78150b7231c5` | same |
| Madrid intersection, **v2** (improved replica, 2026-10-19) | same as Madrid | `c6769a36-4bad-43f7-90d8-f49ed780992c` | `7d262860-b466-4e15-8f5d-d54e4d634a9d` | same |

Artifact root: `s3://paralleldomain-pipeline/artifacts/replica_replipair_pdncc/<replipair uid>/`.
The replipair artifacts themselves hold every frame: `<camera>/original/`, `<camera>/rendered/`,
`variations/<rainy|snowy>/<camera>/rendered/` (file stems `<session>__<camera>__<t_ns>`).

Source bundles the re-renders were made from (have the trained splats; usable for other render
configs): Italy `2971f47f-918b-4b02-b9d5-f23e9da16004` (nightly), Madrid
`8dca3e15-9982-4372-ba98-d7f7c445339f` (release test, 4 variation types baked), Austria
`602e7901-4406-4773-871a-038874d09f0d`.

## Model-input exports (format: `EXPORT_FORMAT.md`)

Each export exists twice: in the artifact (AWS, `<replipair root>/tmp_alpamayo_eval/<name>/`) and
mirrored to Nebius (`s3://pd-nebius-ray/alpamayo-eval/<export name>/`). Jobs read Nebius.

| Nebius export name | AWS location | instants | arms | objects / size |
|---|---|---|---|---|
| `full_variations` | `.../5c833e25-.../tmp_alpamayo_eval/full/` | 174 | real, rendered, rendered_rainy, rendered_snowy | 4957 / 1.9 GB |
| `madrid_full_variations` | `.../062b2b02-.../tmp_alpamayo_eval/full/` | 185 | same four | 5265 / 2.2 GB |
| `austria_full_variations` | `.../3671dc87-.../tmp_alpamayo_eval/full/` | 186 | same four | 5293 / 1.8 GB |
| `italy_full_variations_v2` | `.../281b3822-.../tmp_alpamayo_eval/full/` | 174 | same four | 4957 |
| `austria_full_variations_v2` | `.../3a3586e2-.../tmp_alpamayo_eval/full/` | 186 | same four | 5293 |
| `madrid_full_variations_v2` | `.../7d262860-.../tmp_alpamayo_eval/full/` | 185 | same four | 5265 |
| `full` (Italy, no variations; superseded) | `.../ee58e53a-.../tmp_alpamayo_eval/full/` | 174 | real, rendered | 2479 / 1.0 GB |
| `w5s_3s` (Italy 5-8 s window; first test) | `.../ee58e53a-.../tmp_alpamayo_eval/w5s_3s/` | 30 | real, rendered | 463 / 0.2 GB |

`.../ee58e53a-.../tmp_alpamayo_eval/italy_3s/` is a leftover duplicate of `w5s_3s` (AWS only).
The `_v2` exports sit on exactly the same t0 grid as the first exports of their scene (same clip,
same pose log), so runs on old and new exports can be joined instant by instant (dashboard,
compare-runs).

## Model runs (Nebius, `s3://pd-nebius-ray/alpamayo-eval/<export>/runs/<run id>/`)

All: Alpamayo 2 Super, bf16, 10 diffusion steps, 1 trajectory sample, top_p 0.98, temperature 0.6,
one H100. Contents per run: `results.jsonl`, `<t0_ns>_<arm>.json` (text, timings, ADE),
`<t0_ns>_<arm>.npz` (`pred_xyz` (1,64,3), `gt_xyz` (64,3) when the full future exists), `summary.json`,
and `<t0_ns>_<arm>.png` (NVIDIA's six-camera figure) where `--save-viz` was used.

| run id | export | arms | seed | notes |
|---|---|---|---|---|
| `alpamayo-run-27f93c3e` | `full_variations` | real, rendered, rendered_rainy, rendered_snowy | 42 | main Italy run, with PNGs |
| `alpamayo-run-af7ff7bc` | `full_variations` | real | 1 | Italy seed-to-seed noise floor |
| `alpamayo-run-e368b20f` | `madrid_full_variations` | all four | 42 | no PNGs |
| `alpamayo-run-17c45f0a` | `austria_full_variations` | all four | 42 | no PNGs |
| `alpamayo-run-928b4727` | `italy_full_variations_v2` | all four | 42 | improved Italy replica, with PNGs |
| `alpamayo-run-9fc99df6` | `austria_full_variations_v2` | all four | 42 | improved Austria replica, with PNGs |
| `alpamayo-run-99bb841a` | `madrid_full_variations_v2` | all four | 42 | improved Madrid replica, with PNGs (`alpamayo-run-ca0b46ad` = failed resubmit, typo, nothing uploaded) |
| `alpamayo-run-e6d4f80e` | `madrid_full_variations_v2` | real | 1 | Madrid seed-to-seed noise floor |
| `alpamayo-run-0d74dd90` | `w5s_3s` | real, rendered | 42 | first end-to-end test, with PNGs |

Earlier infrastructure jobs (no results of interest): `alpamayo-probe-f5d8ec4f` (env + weights),
`alpamayo-smoke-16f38bee` (died on HF dataset access), `alpamayo-run-82bbdb88` (died: shared volume
full), `alpamayo-exec-*` (diagnostics). Ray dashboard:
`http://ray-nebius.internal.paralleldomain.com:8265/#/jobs/<run id>`.

## Cluster

- Weights: `/mnt/pdfm-data/pdfm-data-cache/alpamayo-eval/models/alpamayo2-super/` (67 GB, shared volume).
- HF cache + token: `/mnt/pdfm-data/pdfm-data-cache/alpamayo-eval/hf-cache/` (the pod's cluster token; model access only).
- Outputs of the two earliest runs also under `/mnt/pdfm-data/pdfm-data-cache/alpamayo-eval/runs/` (196 MB; duplicates of the bucket copies).
- Everything else a job needs lives on the pod's local SSD and is gone when the pod is recycled (by design: the shared volume is 100 % full).

## Laptop

- `~/.cache/alpamayo-eval/pd-nebius-ray/alpamayo-eval/...`: staged copies of every export and run above (dashboard, video, compare-runs). Safe to delete; re-staged on demand.
- `~/.cache/alpamayo-eval/thumbs/`: dashboard thumbnails. Safe to delete.
- `~/workspace/alpamayo2/exports/italy_3s/`: the first local export (= `w5s_3s`), 185 MB, gitignored.
- `~/workspace/alpamayo2/outputs/*.mp4`: videos (gitignored): `italy_full_variations_27f93c3e`,
  `madrid_full_variations_e368b20f`, `austria_full_variations_17c45f0a`, `italy_w5s_3s_0d74dd90`
  (current layout); `italy_full_variations_v2_928b4727`, `austria_full_variations_v2_9fc99df6`,
  `madrid_full_variations_v2_99bb841a` (improved replicas, 2026-10-19); `full_variations_27f93c3e`, `w5s_3s_0d74dd90` (older layout).
- `/tmp/replipair_ee58e53a/`: full download of the first Italy replipair artifact (6.4 GB).
- AWS profiles: `default` (SSO), `nebius` (object-store key, see `HOWTO.md` §0).

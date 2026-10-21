"""Run Alpamayo 2 Super over an export: every instant x every arm, common random numbers.

    python -m pd_alpamayo_eval.run_export --model-id <ckpt dir> --export italy_3s --out-dir <dir> \\
        [--arms real rendered] [--max-instants N] [--num-traj-samples 1] [--diffusion-steps 10] \\
        [--seed 42] [--save-viz]

``--export`` is a directory, a name resolved under ``$ALPAMAYO_EVAL_EXPORTS`` (the entrypoint sets
it to the persistent volume on the cluster), or an ``s3://`` export, which is staged there once
(same-size files are skipped on later runs) and whose outputs are uploaded to
``<export>/runs/<run id>/`` at the end. On the cluster the only store is the Nebius bucket, reached
with the credentials and endpoint the submitter forwards. The model is loaded once. For each instant the
same CUDA seed is set before sampling each arm, so the diffusion noise is identical across arms
(brainstorm "common random numbers"). Per instant and arm it writes ``<t0_ns>_<arm>.json`` (CoT,
timings, metrics) and ``<t0_ns>_<arm>.npz`` (predicted and GT xyz), appends a line to
``results.jsonl``, and at the end writes ``summary.json`` and prints ``SUMMARY {...}``.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from alpamayo2_super.inference_smoke import validate_model_id
from pd_alpamayo_eval.export_common import MANIFEST_NAME
from pd_alpamayo_eval.load_export import list_arms, list_instants, load_export
from pd_alpamayo_eval.s3util import client_from_env, download_prefix, is_s3, parse_s3, upload_dir


def _upload_new(out_dir: Path, upload_to: str, uploaded: set[Path], log=print) -> int:
    """Upload files not uploaded yet (plus results.jsonl / summary.json, which change), without HEAD checks."""
    s3 = client_from_env()
    bucket, prefix = parse_s3(upload_to)
    n = 0
    for path in sorted(p for p in out_dir.rglob("*") if p.is_file()):
        if path in uploaded and path.name not in ("results.jsonl", "summary.json"):
            continue
        s3.upload_file(str(path), bucket, f"{prefix}/{path.relative_to(out_dir).as_posix()}")
        uploaded.add(path)
        n += 1
    log(f"  checkpoint: uploaded {n} files -> {upload_to}", flush=True)
    return n


def resolve_export(export: str) -> Path:
    """A local directory, a name under ``$ALPAMAYO_EVAL_EXPORTS``, or an ``s3://`` export staged there."""
    root = os.environ.get("ALPAMAYO_EVAL_EXPORTS")
    if is_s3(export):
        if not root:
            raise RuntimeError("staging an s3:// export needs ALPAMAYO_EVAL_EXPORTS (set by the entrypoint)")
        name = export.rstrip("/").rsplit("/", 1)[-1]
        local = Path(root) / name
        t = time.perf_counter()
        n = download_prefix(client_from_env(), export, local)
        print(f"staged {export} -> {local}: {n} new files in {time.perf_counter() - t:.0f} s", flush=True)
        if not (local / MANIFEST_NAME).is_file():
            raise FileNotFoundError(f"{export} has no {MANIFEST_NAME}")
        return local
    path = Path(export).expanduser()
    if path.is_dir():
        return path
    if root and (Path(root) / export).is_dir():
        return Path(root) / export
    raise FileNotFoundError(f"export {export!r} is not a directory and not found under ALPAMAYO_EVAL_EXPORTS={root!r}")


def _ade(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mean xy distance over time between trajectories ``a`` (S,T,3) and ``b`` (T,3) or (S,T,3)."""
    return np.linalg.norm(a[..., :2] - b[..., :2], axis=-1).mean(-1)


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"n": 0, "median": None, "p90": None}
    arr = np.asarray(values, dtype=float)
    return {"n": int(arr.size), "median": round(float(np.median(arr)), 3), "p90": round(float(np.percentile(arr, 90)), 3)}


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    validate_model_id(args.model_id)
    export_dir = resolve_export(args.export)
    upload_to = args.upload_to
    if upload_to is None and is_s3(args.export):
        upload_to = f"{args.export.rstrip('/')}/runs/{args.run_id}"
    instants = list_instants(export_dir)
    if args.max_instants is not None:
        instants = instants[: args.max_instants]
    if not instants:
        raise SystemExit(f"{export_dir} has no instants")
    export_arms = list_arms(export_dir)
    arms = args.arms or export_arms
    unknown = [a for a in arms if a not in export_arms]
    if unknown:
        raise SystemExit(f"arms {unknown} are not in the export (has {export_arms})")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from alpamayo2_super import helper
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.visualization import plot_inference_result

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"

    summary: dict[str, Any] = {
        "model_id": args.model_id,
        "export": args.export,
        "export_dir": str(export_dir),
        "run_id": args.run_id,
        "upload_to": upload_to,
        "arms": arms,
        "num_instants": len(instants),
        "num_traj_samples": args.num_traj_samples,
        "diffusion_steps": args.diffusion_steps,
        "seed": args.seed,
        "device": torch.cuda.get_device_name(0),
        "records": [],
    }
    t = time.perf_counter()
    model = Alpamayo2Super.from_pretrained(args.model_id, dtype=torch.bfloat16, device_map="cuda:0")
    torch.cuda.synchronize()
    summary["model_load_s"] = round(time.perf_counter() - t, 1)
    print(f"model loaded in {summary['model_load_s']} s", flush=True)
    recorder = None
    if args.save_attention:
        from pd_alpamayo_eval.attention import ExpertAttentionRecorder

        recorder = ExpertAttentionRecorder(model)
        summary["attention"] = {"expert_attn_implementation": recorder.impl, "expert_layers": recorder.num_layers}
        print(f"recording expert attention ({recorder.impl}, {recorder.num_layers} layers)", flush=True)

    preds: dict[tuple[int, str], np.ndarray] = {}
    done: set[tuple[int, str]] = set()
    if args.resume_from:
        # Continue a run whose node died: take its records (json/npz/results.jsonl) as our own and skip them.
        t = time.perf_counter()
        n = download_prefix(client_from_env(), args.resume_from.rstrip("/"), out_dir, log=lambda *a, **k: None)
        if results_path.is_file():
            kept = []
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                old = json.loads(line)
                if old.get("error"):
                    continue
                key = (int(old["t0_ns"]), old["arm"])
                npz_old = out_dir / f"{old['t0_ns']}_{old['arm']}.npz"
                if not npz_old.is_file():
                    continue
                done.add(key)
                preds[key] = np.load(npz_old)["pred_xyz"]
                kept.append(line)
                summary["records"].append({k: old.get(k) for k in ("t0_ns", "arm", "inference_s", "min_ade_vs_gt_m", "error")})
            results_path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        (out_dir / "summary.json").unlink(missing_ok=True)
        summary["resumed_from"] = args.resume_from
        print(f"resumed {len(done)} records from {args.resume_from} ({n} files, {time.perf_counter() - t:.0f} s)", flush=True)
    uploaded: set[Path] = set()
    if args.ablate:
        from pd_alpamayo_eval.ablation import ablated, camera_names_of, summarize_record

        arms = sorted(arms, key=lambda a: a != "real")  # real first: the real swap and the vs-real metrics need it
        summary["ablate"] = list(args.ablate)
    with results_path.open("a", encoding="utf-8") as results_file:
        for i, t0_ns in enumerate(instants):
            available = list_arms(export_dir, t0_ns)
            real_data = None  # the real arm's loaded data of this instant, for the real swap
            for arm in arms:
                if arm not in available:
                    continue  # optional arm (e.g. a weather variation) missing for this instant
                if (t0_ns, arm) in done:
                    if arm == "real" and args.ablate and "real" in args.ablate:
                        real_data = select_task_input(load_export(export_dir, t0_ns, "real"), "trajectory")  # frames for the real swap
                    continue
                rec: dict[str, Any] = {"t0_ns": t0_ns, "arm": arm, "index": i}
                try:
                    t = time.perf_counter()
                    data = select_task_input(load_export(export_dir, t0_ns, arm), "trajectory")
                    model_inputs = helper.to_device(
                        helper.prepare_model_inputs(data, model.config, model.tokenizer), "cuda"
                    )
                    rec["load_s"] = round(time.perf_counter() - t, 2)
                    t = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.manual_seed_all(args.seed)
                    with torch.autocast("cuda", dtype=torch.bfloat16), (recorder.recording() if recorder else nullcontext()):
                        pred_xyz, _pred_rot, _logprob, extra = model.sample_trajectories_from_data(
                            data=model_inputs,
                            top_p=0.98,
                            temperature=0.6,
                            num_traj_samples=args.num_traj_samples,
                            diffusion_kwargs={"inference_step": args.diffusion_steps},
                            return_extra=True,
                        )
                    torch.cuda.synchronize()
                    rec["inference_s"] = round(time.perf_counter() - t, 2)
                    rec["peak_vram_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
                    pred = pred_xyz.cpu().numpy()[0, 0]  # (S, 64, 3)
                    preds[(t0_ns, arm)] = pred
                    # extra["cot"] is an object array of shape (batch, sets, samples); one string per sample.
                    rec["cot"] = [str(c) for c in np.asarray(extra["cot"], dtype=object).reshape(-1).tolist()]
                    npz: dict[str, np.ndarray] = {"pred_xyz": pred}
                    if recorder is not None:
                        try:
                            npz.update(recorder.summarize(model, model_inputs["tokenized_data"], data["camera_indices"],
                                                          int(data["image_frames"].shape[1]), per_layer=args.attention_per_layer))
                            rec["attention_mass_late"] = [round(float(v), 4) for v in npz["attn_mass"][-json.loads(str(npz["attn_layout"]))["late_layers"]:].mean(0)]
                        except Exception as exc:  # attention is best-effort; the prediction is already made
                            rec["attention_error"] = f"{type(exc).__name__}: {exc}"
                    if data.get("ego_future_available"):
                        gt = data["ego_future_xyz"].cpu().numpy()[0, 0]
                        npz["gt_xyz"] = gt
                        ade = _ade(pred, gt)
                        rec["ade_vs_gt_m"] = [round(float(v), 3) for v in ade]
                        rec["min_ade_vs_gt_m"] = round(float(ade.min()), 3)
                    if args.ablate:
                        if arm == "real":
                            real_data = data
                        t_abl = time.perf_counter()
                        cameras = camera_names_of(data)
                        npz["ablate_cameras"] = np.array(cameras)
                        real_pred = preds.get((t0_ns, "real"))
                        for mode in args.ablate:
                            if mode == "real" and (arm == "real" or real_data is None):
                                continue
                            abl_preds, abl_cots = [], []
                            for k in range(len(cameras)):
                                d = ablated(data, k, real_data["image_frames"] if mode == "real" else None)
                                inputs = helper.to_device(helper.prepare_model_inputs(d, model.config, model.tokenizer), "cuda")
                                torch.cuda.manual_seed_all(args.seed)
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    a_xyz, _r, _l, a_extra = model.sample_trajectories_from_data(
                                        data=inputs, top_p=0.98, temperature=0.6, num_traj_samples=args.num_traj_samples,
                                        diffusion_kwargs={"inference_step": args.diffusion_steps}, return_extra=True)
                                abl_preds.append(a_xyz.cpu().numpy()[0, 0, 0])
                                abl_cots.append(str(np.asarray(a_extra["cot"], dtype=object).reshape(-1)[0]))
                                del a_xyz, _r, _l, a_extra, inputs
                            npz[f"ablate_{mode}_xyz"] = np.stack(abl_preds)
                            rec[f"ablate_{mode}"] = summarize_record(pred[0], real_pred[0] if real_pred is not None else None, abl_preds, cameras, abl_cots)
                        rec["ablate_s"] = round(time.perf_counter() - t_abl, 2)
                    np.savez(out_dir / f"{t0_ns}_{arm}.npz", **npz)
                    if args.save_viz:
                        try:
                            fig, _ = plot_inference_result(
                                data=data,
                                pred_xyz=pred_xyz,
                                extra=extra,
                                output_path=str(out_dir / f"{t0_ns}_{arm}.png"),
                                json_path=None,
                                model_id=args.model_id,
                                seed=args.seed,
                            )
                            plt.close(fig)  # NVIDIA's plotter leaves the figure open; 60+ of them leak host memory
                        except Exception as exc:  # viz is best-effort; GT may be absent
                            rec["viz_error"] = f"{type(exc).__name__}: {exc}"
                    del pred_xyz, _pred_rot, _logprob, extra, model_inputs
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                    rec["traceback"] = traceback.format_exc()
                    print(f"[{i}] {t0_ns} {arm}: FAILED {rec['error']}", flush=True)
                (out_dir / f"{t0_ns}_{arm}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
                results_file.write(json.dumps({k: v for k, v in rec.items() if k != "traceback"}) + "\n")
                results_file.flush()
                summary["records"].append({k: rec.get(k) for k in ("t0_ns", "arm", "inference_s", "min_ade_vs_gt_m", "error")})
                print(
                    f"[{i + 1}/{len(instants)}] {t0_ns} {arm:8s} "
                    f"inf={rec.get('inference_s')} s minADE={rec.get('min_ade_vs_gt_m')} "
                    f"{'ERR' if 'error' in rec else ''}",
                    flush=True,
                )
            if upload_to and args.upload_every and (i + 1) % args.upload_every == 0:
                # Partial summary + everything new, so a dead node loses at most `upload_every` instants
                # (resume with --resume-from <this run's uri>) and the dashboard can open the partial run.
                summary["partial"] = True
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
                try:
                    _upload_new(out_dir, upload_to, uploaded)
                except Exception as exc:  # never let a checkpoint kill the run
                    print(f"  checkpoint upload failed: {type(exc).__name__}: {exc}", flush=True)

    # Readout. Paired ADE of every other arm against `real` uses sample s of one arm against
    # sample s of the other (same seed, so same diffusion noise), then the minimum over samples.
    for arm in arms:
        if arm == "real" or "real" not in arms:
            continue
        paired = []
        for t0_ns in instants:
            a, b = preds.get((t0_ns, "real")), preds.get((t0_ns, arm))
            if a is not None and b is not None:
                paired.append(float(_ade(a, b).min()))
        summary[f"paired_ade_real_vs_{arm}_m"] = _percentiles(paired)
    for arm in arms:
        vals = [r["min_ade_vs_gt_m"] for r in summary["records"] if r["arm"] == arm and r.get("min_ade_vs_gt_m") is not None]
        summary[f"min_ade_vs_gt_m_{arm}"] = _percentiles(vals)
        times = [r["inference_s"] for r in summary["records"] if r["arm"] == arm and r.get("inference_s") is not None]
        summary[f"inference_s_{arm}"] = _percentiles(times)
    summary["num_errors"] = sum(1 for r in summary["records"] if r.get("error"))
    summary["partial"] = False
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if upload_to:
        t = time.perf_counter()
        n = upload_dir(client_from_env(), out_dir, upload_to)
        print(f"uploaded {n} files -> {upload_to} in {time.perf_counter() - t:.0f} s", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--export",
        required=True,
        help="Export directory, a name under $ALPAMAYO_EVAL_EXPORTS, or an s3:// export (staged there once).",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-id", default=os.environ.get("ALPAMAYO_EVAL_RUN_ID", time.strftime("local-%Y%m%dT%H%M%SZ", time.gmtime())))
    parser.add_argument(
        "--upload-to",
        default=None,
        help="s3:// directory for the outputs. Default for an s3:// export: <export>/runs/<run id>; none otherwise.",
    )
    parser.add_argument("--arms", nargs="+", default=None, help="Arms to run (default: every arm in the export).")
    parser.add_argument("--max-instants", type=int, default=None)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-viz", action="store_true", help="Also write NVIDIA's six-camera PNG per instant and arm.")
    parser.add_argument("--resume-from", default=None,
                        help="s3 uri of an earlier run of the same export whose job died: its records are taken over and skipped.")
    parser.add_argument("--upload-every", type=int, default=10, help="Checkpoint-upload the results every N instants (0 = only at the end).")
    parser.add_argument("--ablate", nargs="+", choices=["blank", "real"], default=None,
                        help="Camera occlusion: per record one extra forward per camera with its frames replaced by their mean colour (blank) "
                             "and/or, for render arms, by the real frames (real). 6 extra forwards per record. See ablation.py.")
    parser.add_argument("--no-attention-per-layer", dest="attention_per_layer", action="store_false",
                        help="Do not store every expert layer's attention grid (attn_img_layers, float16, ~0.5 MB per record; stored by default "
                             "since 2026-10-21 because the informative layers are the middle ones, not the stored all/late averages).")
    parser.add_argument("--no-attention", dest="save_attention", action="store_false",
                        help="Do not record the expert's attention over the images (recorded by default into the npz: attn_img, attn_layout, attn_mass, attn_image_mass).")
    args = parser.parse_args()
    summary = run(args)
    printable = {k: v for k, v in summary.items() if k != "records"}
    print("SUMMARY " + json.dumps(printable), flush=True)


if __name__ == "__main__":
    main()

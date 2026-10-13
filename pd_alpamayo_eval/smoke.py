"""Phase 1 smoke: NVIDIA's own inference path on the cluster, with timings and peak VRAM.

Same calls as ``alpamayo2_super.inference_smoke.run_smoke`` (which this does not modify), but the
model is loaded once and several manifest samples run back to back, so the per-instant cost
is separated from the one-off weight-load cost. Writes one ``<sample>.json`` / ``<sample>.png``
pair per instant plus ``summary.json`` into ``--out-dir`` and prints the summary to stdout,
because the job log is the only retrieval channel until a Nebius S3 key exists.

    python -m pd_alpamayo_eval.smoke --model-id /mnt/pdfm-data/alpamayo-eval/models/alpamayo2-super \
        --out-dir /tmp/smoke --sample-indices 0 1
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from alpamayo2_super.inference_smoke import MANIFEST_PATH, validate_model_id


def _gib(num_bytes: float) -> float:
    return round(num_bytes / 2**30, 2)


def run(
    model_id: str,
    manifest: Path,
    sample_indices: list[int],
    out_dir: Path,
    num_traj_samples: int,
    diffusion_steps: int,
    seed: int,
) -> dict[str, Any]:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    validate_model_id(model_id)

    import numpy as np
    import torch

    from alpamayo2_super import helper
    from alpamayo2_super.input_profiles import select_task_input
    from alpamayo2_super.load_physical_aiavdataset import load_physical_aiavdataset
    from alpamayo2_super.models.alpamayo2_super import Alpamayo2Super
    from alpamayo2_super.visualization import plot_inference_result

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    samples = json.loads(manifest.read_text(encoding="utf-8"))["samples"]
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "model_id": model_id,
        "manifest": str(manifest),
        "device": torch.cuda.get_device_name(0),
        "num_traj_samples": num_traj_samples,
        "diffusion_steps": diffusion_steps,
        "seed": seed,
        "instants": [],
    }

    t = time.perf_counter()
    model = Alpamayo2Super.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda:0")
    torch.cuda.synchronize()
    summary["model_load_s"] = round(time.perf_counter() - t, 1)
    summary["vram_after_load_gib"] = _gib(torch.cuda.memory_allocated())
    print(f"model loaded in {summary['model_load_s']} s, {summary['vram_after_load_gib']} GiB", flush=True)

    for index in sample_indices:
        sample = samples[index]
        clip_id, t0_us = sample["clip_id"], int(sample["t0_us"])
        record: dict[str, Any] = {"sample_index": index, "clip_id": clip_id, "t0_us": t0_us}

        t = time.perf_counter()
        source_data = load_physical_aiavdataset(clip_id, t0_us=t0_us)
        data = select_task_input(source_data, "trajectory")
        record["data_load_s"] = round(time.perf_counter() - t, 1)

        t = time.perf_counter()
        model_inputs = helper.to_device(helper.prepare_model_inputs(data, model.config, model.tokenizer), "cuda")
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.manual_seed_all(seed)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, logprob, extra = model.sample_trajectories_from_data(
                data=model_inputs,
                top_p=0.98,
                temperature=0.6,
                num_traj_samples=num_traj_samples,
                diffusion_kwargs={"inference_step": diffusion_steps},
                return_extra=True,
            )
        torch.cuda.synchronize()
        record["inference_s"] = round(time.perf_counter() - t, 1)
        record["peak_vram_gib"] = _gib(torch.cuda.max_memory_allocated())
        del pred_rot, logprob

        gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
        pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
        ade = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
        record["min_ade_m"] = round(float(ade.min()), 3)
        record["cot"] = str(extra["cot"][0])

        stem = out_dir / f"sample{index}"
        _, metadata = plot_inference_result(
            data=data,
            pred_xyz=pred_xyz,
            extra=extra,
            output_path=str(stem.with_suffix(".png")),
            json_path=str(stem.with_suffix(".json")),
            model_id=model_id,
            seed=seed,
        )
        record["projection_available"] = bool(metadata.get("projection_available"))
        summary["instants"].append(record)
        print(json.dumps(record, indent=2), flush=True)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", required=True, help="Local checkpoint directory or HF repo id.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH, help="validation_samples.json")
    parser.add_argument("--sample-indices", type=int, nargs="+", default=[0], help="Manifest entries to run.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--num-traj-samples", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = run(
        model_id=args.model_id,
        manifest=args.manifest,
        sample_indices=args.sample_indices,
        out_dir=args.out_dir,
        num_traj_samples=args.num_traj_samples,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
    )
    print("SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()

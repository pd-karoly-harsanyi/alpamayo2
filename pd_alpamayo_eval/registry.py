"""Registry of the runs that exist, per scene, so the dashboard and compare-runs can be started by scene
name instead of S3 paths (``--scene italy``). Nothing here triggers a job: the runs are read from the
bucket (and from the laptop cache after the first use). Full list and findings: ``INVENTORY.md``.

Add a row when a new run lands; the first run of a scene is the reference (its ``real`` arm).
"""

from __future__ import annotations

BUCKET = "s3://pd-nebius-ray/alpamayo-eval"

# scene -> list of (export name, run id, label); label "" = the reference run, tagged plainly.
SCENES: dict[str, list[tuple[str, str, str]]] = {
    "italy": [
        ("full_variations", "alpamayo-run-27f93c3e", ""),  # v1 replica, seed 42, all arms
        ("italy_full_variations_v2", "alpamayo-run-928b4727", "v2"),  # v2 replica, seed 42, all arms
        ("full_variations", "alpamayo-run-af7ff7bc", "seed1"),  # real arm, seed 1 (noise floor)
    ],
    # Madrid: the attention run (v2 replica, predictions bit-identical to 99bb841a) is the reference, so the
    # dashboard's "minus real" attention view works; the first render is the "v1" column.
    "madrid": [
        ("madrid_full_variations_v2", "alpamayo-run-eedc9a2d", ""),  # v2 replica, seed 42, all arms, expert attention with every layer stored
        ("madrid_full_variations", "alpamayo-run-e368b20f", "v1"),  # v1 replica, seed 42, all arms
        ("madrid_full_variations_v2", "alpamayo-run-e6d4f80e", "seed1"),  # real arm, seed 1 (noise floor)
    ],
    "austria": [
        ("austria_full_variations", "alpamayo-run-17c45f0a", ""),
        ("austria_full_variations_v2", "alpamayo-run-9fc99df6", "v2"),
        ("austria_full_variations_v2", "alpamayo-run-b357ae61", "seed1"),
    ],
}


def scene_runs(name: str) -> tuple[list[str], list[str]]:
    """(run URIs, labels) for a scene name; raises SystemExit with the known names otherwise."""
    key = name.lower()
    if key not in SCENES:
        raise SystemExit(f"unknown scene {name!r}; known: {', '.join(SCENES)}")
    rows = SCENES[key]
    return [f"{BUCKET}/{export}/runs/{run}" for export, run, _ in rows], [label for _, _, label in rows]

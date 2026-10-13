"""PD-internal evaluation glue around the untouched ``alpamayo2_super`` package.

See PD_EVAL_PLAN.md. Nothing in here modifies NVIDIA's code; it only calls its public entry
points (``load_physical_aiavdataset``, ``select_task_input``, ``helper.prepare_model_inputs``,
``Alpamayo2Super.sample_trajectories_from_data``).
"""

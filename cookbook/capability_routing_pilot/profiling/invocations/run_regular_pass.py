import json
import os
import sys
from typing import Final

from datasets import load_dataset

from poc import run_pilot
from poc.run_profiling import (
    DATASET_NAME,
    DATASET_REVISION,
    DATASET_SPLIT,
    EVAL_DATASET_REVISION,
    MANIFEST_SHA256,
    PUBLIC_DEV_TENANT,
    ROOT,
    ProfileConfig,
    _default_disk_stats,
    _default_ops,
    _guard_sha256,
    _load_dataset,
    _profile_versions,
    _task_data,
    execute_batch,
    profile_batches,
    profile_guard,
)


def main() -> None:
    config: Final = ProfileConfig(ROOT, (ROOT / "profiling" / "profiling_manifest.json" if (ROOT / "profiling" / "profiling_manifest.json").exists() else ROOT / "profiling_manifest.json"), MANIFEST_SHA256, PUBLIC_DEV_TENANT)
    batches: Final = profile_batches(config)[1:]
    if "--execute" not in sys.argv:
        print(json.dumps([{"batch": b.index, "tasks": [[t.task_id, t.model_order] for t in b.tasks]} for b in batches]))
        return
    guard_sha256: Final = _guard_sha256(config.root_dir, config.overlays)
    print(json.dumps({"guard_sha256": guard_sha256, "first_batch": batches[0].index, "batch_count": len(batches)}))
    versions: Final = _profile_versions(config.root_dir, config.runs_path, guard_sha256)
    agent_source: Final = tuple(load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=DATASET_SPLIT))
    eval_source: Final = _load_dataset("SWE-bench/SWE-bench", EVAL_DATASET_REVISION, "dev")
    data: Final = _task_data(config, tuple(t for b in batches for t in b.tasks), agent_source, eval_source)
    env: Final = os.environ.copy()
    env.pop("LITELLM_GATEWAY_ADMIN_KEY", None)
    env["LITELLM_PROXY_API_KEY"] = run_pilot._read_pilot_key(config.root_dir / ".private" / "profiling_key")
    env["PYTHONPATH"] = str(config.root_dir)
    env["MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT"] = "1"
    base_url: Final = os.environ.get("LITELLM_GATEWAY_BASE_URL", "https://gateway.litellm-sandbox.ai").rstrip("/")
    guard: Final = profile_guard(config, free_bytes=lambda: _default_disk_stats()["free_bytes"])
    ops: Final = _default_ops(config, data, env)
    for batch in batches:
        execute_batch(config, batch, data, versions, ops, env, base_url, guard, guard_sha256)


if __name__ == "__main__":
    main()

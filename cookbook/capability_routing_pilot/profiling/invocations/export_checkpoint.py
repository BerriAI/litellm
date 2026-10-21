import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Final

FIELDS: Final = (
    "record_type", "task_id", "category", "tenant_id", "manifest_sha256", "model_alias", "gateway_model",
    "returned_model_version", "deployment_id", "attempt", "model_order", "success", "eval_status", "exit_status",
    "error_class", "is_infra", "retry_eligible", "budget_guard_refused", "cost_usd", "cost_incomplete", "n_calls",
    "tokens", "latency_s", "solve_latency_s", "eval_latency_s", "harness_sha256", "harness_version",
    "budget_version", "harness_config_hash", "gold_status", "noop_status", "status", "upstream_digest",
    "upstream_image_id", "overlay_image_id", "final_image_id", "agent_image_ref", "grader_image_ref",
    "env_correction_sha256", "corrections_file_sha256", "dockerfile_sha256", "repair_script_sha256", "cause",
    "evidence", "applies_on_top_of", "run_id", "started_at", "finished_at",
)


def main() -> None:
    source: Final = Path(sys.argv[1])
    target: Final = Path(sys.argv[2])
    target.mkdir(parents=True, exist_ok=True)
    rows: Final = tuple(json.loads(line) for line in (source / "profiling_runs.jsonl").read_text().splitlines() if line)
    sanitized: Final = "".join(
        json.dumps({key: row[key] for key in FIELDS if key in row}, sort_keys=True) + "\n" for row in rows
    )
    (target / "task_runs.jsonl").write_text(sanitized)
    shutil.copyfile(source / "capability_map_intermediate.json", target / "capability_map_intermediate.json")
    print(json.dumps({
        "rows": len(rows),
        "source_runs_sha256": hashlib.sha256((source / "profiling_runs.jsonl").read_bytes()).hexdigest(),
        "task_runs_sha256": hashlib.sha256(sanitized.encode()).hexdigest(),
    }))


if __name__ == "__main__":
    main()

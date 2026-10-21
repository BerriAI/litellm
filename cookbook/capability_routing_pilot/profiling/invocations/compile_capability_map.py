import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from poc.profiles import compile_map_from_profiling_runs
from poc.run_profiling import DEFAULT_OVERLAYS, MANIFEST_SHA256, _guard_sha256, _profile_versions

SPEC_EVALUATOR_DEFICITS: Final = {
    "pydicom__pydicom-995": "44 FAIL_TO_PASS IDs embed original-machine temp paths, gold passes 256/256 but cannot match",
    "pydicom__pydicom-1439": "20 PASS_TO_PASS IDs embed original-machine paths, corrected gold passes 80 with 0 failed",
}
GATEWAY_NOTE: Final = (
    "All three aliases share gateway model-info id 13ef3d99-7edf-4a6b-aab6-f088c8e25964, a gateway mapping snapshot, "
    "not model identity. Cohorts are separated by operational alias and returned model name"
)
MODELS: Final = ("haiku-4.5", "sonnet-5", "opus-5")


def _task_status(task_id: str, rows: tuple[Mapping[str, object], ...]) -> dict[str, object]:
    task_rows: Final = tuple(row for row in rows if row.get("task_id") == task_id)
    preflights: Final = tuple(row for row in task_rows if row.get("record_type") in {"preflight_passed", "preflight_failed"})
    corrected: Final = any(row.get("record_type") == "env_correction" for row in task_rows)
    graded_models: Final = {
        row.get("model_alias") for row in task_rows if row.get("record_type", "attempt") == "attempt" and row.get("is_infra") is False
    }
    last: Final = preflights[-1].get("record_type") if preflights else None
    return {
        "task_id": task_id,
        "corrected": corrected,
        "last_preflight": last,
        "graded_models": sorted(str(model) for model in graded_models),
    }


def main() -> None:
    root: Final = Path(sys.argv[1]).resolve()
    output: Final = Path(sys.argv[2])
    runs_path: Final = root / "profiling" / "profiling_runs.jsonl"
    rows: Final = tuple(json.loads(line) for line in runs_path.read_text().splitlines() if line)
    corrections: Final = json.loads((root / "profiling" / "env_corrections.json").read_text())
    versions: Final = _profile_versions(root, None, _guard_sha256(root, DEFAULT_OVERLAYS))
    compiled: Final = compile_map_from_profiling_runs(
        runs_path, root / "profiling_manifest.json", MANIFEST_SHA256, "public_dev_poc", versions
    )
    failed_task_ids: Final = sorted({str(row["task_id"]) for row in rows if row.get("record_type") == "preflight_failed"})
    statuses: Final = {task_id: _task_status(task_id, rows) for task_id in sorted(set(failed_task_ids) | set(corrections))}
    pending: Final = sorted(
        task_id
        for task_id, status in statuses.items()
        if task_id in corrections
        and task_id not in SPEC_EVALUATOR_DEFICITS
        and (
            not status["corrected"]
            or (status["last_preflight"] == "preflight_passed" and len(status["graded_models"]) < len(MODELS))
        )
    )
    unresolved: Final = sorted(
        task_id for task_id, status in statuses.items() if status["last_preflight"] == "preflight_failed" and task_id not in pending
    )
    attempts: Final = tuple(row for row in rows if row.get("record_type", "attempt") == "attempt")
    result: Final = compiled | {
        "final": compiled["final"] is True and not pending,
        "compiler_final_flag": compiled["final"],
        "pending_environment_repairs": pending,
        "spec_evaluator_deficits": [
            {"task_id": task_id, "reason": SPEC_EVALUATOR_DEFICITS[task_id]} for task_id in unresolved if task_id in SPEC_EVALUATOR_DEFICITS
        ],
        "environment_deficits": [task_id for task_id in unresolved if task_id not in SPEC_EVALUATOR_DEFICITS],
        "repaired_tasks": [task_id for task_id, status in statuses.items() if status["corrected"] and status["last_preflight"] == "preflight_passed"],
        "accounting": {
            "attempt_count": len(attempts),
            "graded_count": sum(1 for row in attempts if row.get("is_infra") is False),
            "infra_count": sum(1 for row in attempts if row.get("is_infra") is True),
            "successes": sum(1 for row in attempts if row.get("success") is True),
            "full_attempt_spend_usd": round(sum(float(row["cost_usd"]) for row in attempts), 6),
        },
        "gateway_identity_note": GATEWAY_NOTE,
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: result[key] for key in ("final", "compiler_final_flag", "pending_environment_repairs",
                                                    "environment_deficits", "repaired_tasks", "accounting")}))


if __name__ == "__main__":
    main()

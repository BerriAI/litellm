import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[1]
TASKS_PATH = ROOT / "heldout_tasks.json"
MAP_PATH = ROOT / "heldout_capability_map.json"
MAP_SHA_PATH = ROOT / "heldout_capability_map.sha256"
PROMPT_SHA_PATH = ROOT / "heldout_baseline_prompt.sha256"
RUNS_PATH = ROOT / "heldout_runs.jsonl"
RAW_PATH = ROOT / "heldout_raw"
LABELS_PATH = ROOT / "heldout_labels.json"
AUX_PATH = ROOT / "heldout_costs_aux.jsonl"
WORKERS = 3
SEED = "litellm-capability-routing-heldout-v1"
MAX_BUDGET_USD = 20.0


def balanced_schedule(task_ids: tuple[str, ...]) -> tuple[tuple[str, tuple[str, str]], ...]:
    rng: Final = random.Random(SEED)
    ordered: Final = tuple(sorted(task_ids, key=lambda item: hashlib.sha256(f"{SEED}{item}".encode()).hexdigest()))
    first_arms: Final = ("capability", "baseline") if rng.getrandbits(1) else ("baseline", "capability")
    return tuple(
        (
            task_id,
            (first_arms[index % 2], "baseline" if first_arms[index % 2] == "capability" else "capability"),
        )
        for index, task_id in enumerate(ordered)
    )


def classifier_plus_solver_cost(classifier_cost: float, solver_cost: float) -> float:
    return classifier_cost + solver_cost


def _sha(path: Path) -> str:
    return path.read_text().split()[0]


def _load_plan() -> tuple[tuple[str, ...], tuple[tuple[str, tuple[str, str]], ...], str, str]:
    manifest: Final = json.loads(TASKS_PATH.read_text())
    task_ids: Final = tuple(item["instance_id"] for item in manifest["tasks"])
    map_sha: Final = _sha(MAP_SHA_PATH)
    prompt_sha: Final = _sha(PROMPT_SHA_PATH)
    return task_ids, balanced_schedule(task_ids), map_sha, prompt_sha


def _print_plan() -> None:
    task_ids, schedule, map_sha, prompt_sha = _load_plan()
    print(
        json.dumps(
            {
                "workers": WORKERS,
                "seed": SEED,
                "map_sha256": map_sha,
                "baseline_prompt_sha256": prompt_sha,
                "tasks": [
                    {"task_id": task_id, "arms": list(arms), "order_index": index}
                    for index, (task_id, arms) in enumerate(schedule)
                ],
                "task_count": len(task_ids),
            },
            indent=2,
        )
    )


def _execute() -> None:
    if os.environ.get("ROUTING_POC_HELDOUT_DESIGN_APPROVED") != "approved":
        raise PermissionError("held-out paid execution requires design approval")
    from poc.heldout_live import execute_schedule

    task_ids, schedule, map_sha, prompt_sha = _load_plan()
    execute_schedule(task_ids, schedule, map_sha, prompt_sha, WORKERS, MAX_BUDGET_USD)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if arguments.dry_run == arguments.execute:
        parser.error("choose exactly one of --dry-run or --execute")
    if arguments.dry_run:
        _print_plan()
        return
    _execute()


if __name__ == "__main__":
    main()

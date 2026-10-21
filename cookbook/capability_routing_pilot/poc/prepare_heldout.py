import hashlib
import json
import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Final

from datasets import load_dataset

from poc.policy import SpendSuccessCell, build_spend_success_map
from poc.prepare import DATASET_REVISION, DATASET_SPLIT, _load_rows, agent_rows
from poc.run_pilot import read_jsonl_locked

ROOT = Path(__file__).resolve().parents[1]
SEED = "litellm-capability-routing-pilot-v1"
RULE = "sort dev ids by sha256(seed+instance_id); exclude original five; take next five with public eval image"
TASKS_PATH = ROOT / "heldout_tasks.json"
DATASET_PATH = ROOT / "heldout_agent_dataset" / "train.jsonl"
MAP_PATH = ROOT / "heldout_capability_map.json"
EVAL_DATASET_REVISION = "c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2"
EVAL_DATASET_PATH = ROOT / "eval_heldout" / "swebench_dev_c6fe717.json"
NOOP_PREDICTIONS_PATH = ROOT / "eval_heldout" / "noop_preds.json"
NOOP_PATCH = (
    "diff --git a/routing_poc_noop.txt b/routing_poc_noop.txt\n"
    "new file mode 100644\n"
    "index 0000000..d0951a7\n"
    "--- /dev/null\n"
    "+++ b/routing_poc_noop.txt\n"
    "@@ -0,0 +1 @@\n"
    "+noop\n"
)


def _selection_candidates() -> tuple[Mapping[str, object], ...]:
    payload: Final[object] = json.loads((ROOT / "selection_candidates.json").read_text())
    if not isinstance(payload, Mapping) or payload.get("seed") != SEED:
        raise ValueError("selection_candidates.json does not match the frozen seed")
    candidates: Final[object] = payload.get("candidates")
    if not isinstance(candidates, list) or not all(isinstance(row, Mapping) for row in candidates):
        raise ValueError("selection_candidates.json must contain candidate objects")
    return tuple(candidates)


def _candidate_id(candidate: Mapping[str, object]) -> str:
    value: Final = candidate.get("instance_id")
    if not isinstance(value, str):
        raise ValueError("candidate instance_id must be a string")
    return value


def _heldout_candidates() -> tuple[Mapping[str, object], ...]:
    candidates: Final = _selection_candidates()
    hashes: Final = tuple(
        hashlib.sha256(f"{SEED}{_candidate_id(candidate)}".encode()).hexdigest()
        for candidate in candidates
    )
    if hashes != tuple(sorted(hashes)):
        raise ValueError("selection candidates are not in the original seeded order")
    exclusions: Final = frozenset(_candidate_id(candidate) for candidate in candidates[:5])
    training_ids: Final = frozenset(
        str(row["task_id"])
        for row in read_jsonl_locked(ROOT / "runs.jsonl")
        if isinstance(row.get("task_id"), str)
    )
    if exclusions != training_ids:
        raise ValueError("the first five seeded candidates do not match the read-only training records")
    selected: Final = tuple(
        candidate
        for candidate in candidates
        if _candidate_id(candidate) not in exclusions and candidate.get("exists") is True
    )[:5]
    if len(selected) != 5:
        raise ValueError("fewer than five public-image candidates are frozen in selection_candidates.json")
    return selected


def _heldout_task_payload(candidates: Sequence[Mapping[str, object]]) -> dict[str, object]:
    tasks: Final = tuple(
        {
            "instance_id": _candidate_id(candidate),
            "image": candidate["image"],
            "upstream_registry_digest": candidate["digest"],
            "image_size_bytes": candidate["size"],
        }
        for candidate in candidates
    )
    return {
        "seed": SEED,
        "rule": RULE,
        "dataset": f"princeton-nlp/SWE-bench {DATASET_SPLIT}@{DATASET_REVISION}",
        "exclusion_list": [
            _candidate_id(candidate)
            for candidate in _selection_candidates()[:5]
        ],
        "tasks": tasks,
    }


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite frozen held-out artifact {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _write_tasks_and_dataset() -> None:
    candidates: Final = _heldout_candidates()
    dataset_rows: Final = _load_rows()
    task_ids: Final = tuple(_candidate_id(candidate) for candidate in candidates)
    agent_dataset_rows: Final = agent_rows(dataset_rows, task_ids)
    for task, candidate in zip(_heldout_task_payload(candidates)["tasks"], candidates, strict=True):
        row: Final = next(row for row in agent_dataset_rows if row["instance_id"] == _candidate_id(candidate))
        if row["instance_id"] != task["instance_id"]:
            raise ValueError("held-out task projection did not preserve its selected id")
    _write_new(TASKS_PATH, json.dumps(_heldout_task_payload(candidates), indent=2) + "\n")
    _write_new(DATASET_PATH, "".join(json.dumps(row) + "\n" for row in agent_dataset_rows))


def _write_capability_map() -> str:
    if MAP_PATH.exists():
        raise FileExistsError(f"refusing to overwrite frozen held-out artifact {MAP_PATH.name}")
    records: Final = read_jsonl_locked(ROOT / "runs.jsonl")
    model_payload: Final[object] = json.loads((ROOT / "poc" / "pilot_models.json").read_text())
    models: Final = tuple(model_payload["solvers"])
    scores: Final = build_spend_success_map(records, models)
    training_ids: Final = sorted({str(row["task_id"]) for row in records})
    payload: Final = {
        "training_records": "runs.jsonl, read-only",
        "training_task_ids": training_ids,
        "training_only": True,
        "score": "total spend including failures / graded successes; unsmoothed",
        "sparse_threshold_graded_tasks": 3,
        "scores": {
            category: {
                model: asdict(cell)
                for model, cell in cells.items()
            }
            for category, cells in scores.items()
        },
    }
    content: Final = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    _write_new(MAP_PATH, content)
    digest: Final = hashlib.sha256(content.encode()).hexdigest()
    _write_new(MAP_PATH.with_suffix(".sha256"), f"{digest}  {MAP_PATH.name}\n")
    return digest


def prepare_heldout() -> tuple[str, str]:
    _write_tasks_and_dataset()
    map_sha256: Final = _write_capability_map()
    task_sha256: Final = hashlib.sha256(TASKS_PATH.read_bytes()).hexdigest()
    return task_sha256, map_sha256


def prepare_eval_dataset() -> tuple[str, str]:
    task_payload: Final[object] = json.loads(TASKS_PATH.read_text())
    if not isinstance(task_payload, Mapping) or not isinstance(task_payload.get("tasks"), list):
        raise ValueError("heldout_tasks.json is not a valid frozen manifest")
    task_ids: Final = tuple(
        str(task["instance_id"])
        for task in task_payload["tasks"]
        if isinstance(task, Mapping) and isinstance(task.get("instance_id"), str)
    )
    if len(task_ids) != 5:
        raise ValueError("the held-out evaluator requires exactly five frozen tasks")
    dataset: Final = load_dataset(
        "SWE-bench/SWE-bench",
        revision=EVAL_DATASET_REVISION,
        split="dev",
    )
    rows: Final = tuple(
        row
        for row in dataset
        if isinstance(row.get("instance_id"), str) and row["instance_id"] in task_ids
    )
    if {row["instance_id"] for row in rows} != set(task_ids):
        raise ValueError("held-out task IDs are missing from the pinned SWE-bench evaluation dataset")
    _write_new(EVAL_DATASET_PATH, json.dumps(rows, indent=2) + "\n")
    predictions: Final = tuple(
        {
            "instance_id": task_id,
            "model_name_or_path": "noop-control",
            "model_patch": NOOP_PATCH,
        }
        for task_id in task_ids
    )
    _write_new(NOOP_PREDICTIONS_PATH, json.dumps(predictions, indent=2) + "\n")
    dataset_sha256: Final = hashlib.sha256(EVAL_DATASET_PATH.read_bytes()).hexdigest()
    predictions_sha256: Final = hashlib.sha256(NOOP_PREDICTIONS_PATH.read_bytes()).hexdigest()
    return dataset_sha256, predictions_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.eval_only:
        dataset_sha256, predictions_sha256 = prepare_eval_dataset()
        print(json.dumps({"eval_dataset_sha256": dataset_sha256, "noop_predictions_sha256": predictions_sha256}))
        return
    task_sha256, map_sha256 = prepare_heldout()
    print(json.dumps({"heldout_tasks_sha256": task_sha256, "heldout_capability_map_sha256": map_sha256}))


if __name__ == "__main__":
    main()

import argparse
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "princeton-nlp/SWE-bench"
DATASET_REVISION = "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6"
DATASET_SPLIT = "dev"
TRAIN_FIELDS: Final = ("instance_id", "repo", "base_commit", "problem_statement", "version", "created_at")


def _load_rows() -> tuple[Mapping[str, object], ...]:
    dataset: Final = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=DATASET_SPLIT)
    return tuple(dataset)


def _train_row(row: Mapping[str, object]) -> dict[str, object]:
    result: Final = {field: row[field] for field in TRAIN_FIELDS}
    if not all(isinstance(result[field], str) for field in TRAIN_FIELDS):
        raise RuntimeError("SWE-bench dev row contains a non-string training field")
    return result


def agent_rows(
    dataset_rows: Sequence[Mapping[str, object]],
    task_ids: Sequence[str],
) -> list[dict[str, object]]:
    dataset_ids: Final = frozenset(
        row["instance_id"]
        for row in dataset_rows
        if isinstance(row.get("instance_id"), str)
    )
    missing_ids: Final = tuple(task_id for task_id in task_ids if task_id not in dataset_ids)
    if missing_ids:
        raise ValueError(f"Frozen task IDs are missing from the SWE-bench dev dataset: {', '.join(missing_ids)}")
    frozen_ids: Final = frozenset(task_ids)
    return [
        _train_row(row)
        for row in dataset_rows
        if isinstance(row.get("instance_id"), str) and row["instance_id"] in frozen_ids
    ]


def _task_id(row: object) -> str:
    if not isinstance(row, Mapping):
        raise ValueError("Each tasks.json task must be an object")
    task_id: Final = row.get("instance_id")
    if not isinstance(task_id, str):
        raise ValueError("Each tasks.json task must have a string instance_id")
    return task_id


def _frozen_task_ids() -> tuple[str, ...]:
    manifest: Final[object] = json.loads((ROOT / "tasks.json").read_text())
    if not isinstance(manifest, Mapping):
        raise ValueError("tasks.json must contain an object")
    tasks: Final[object] = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("tasks.json must contain a tasks list")
    return tuple(_task_id(task) for task in tasks)


def _write_train(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def prepare(check: bool = False) -> None:
    task_ids: Final = _frozen_task_ids()
    rows: Final = agent_rows(_load_rows(), task_ids)
    train_path: Final = ROOT / "agent_dataset" / "train.jsonl"
    if not check:
        _write_train(train_path, rows)
        return
    with tempfile.TemporaryDirectory(prefix="routing-poc-prepare-") as temp_dir:
        generated_train: Final = Path(temp_dir) / "agent_dataset" / "train.jsonl"
        _write_train(generated_train, rows)
        if generated_train.read_bytes() != train_path.read_bytes():
            raise AssertionError("Regenerated train.jsonl is not byte-identical")
    print(f"Dataset preparation check passed for {len(task_ids)} task IDs and byte-identical train.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    prepare(check=args.check)


if __name__ == "__main__":
    main()

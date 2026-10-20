import argparse
import hashlib
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
DATASET_NAME = "princeton-nlp/SWE-bench"
DATASET_REVISION = "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6"
DATASET_SPLIT = "dev"
SEED = "litellm-capability-routing-pilot-v1"
FROZEN_AT = "2026-10-20T23:10Z"
TASK_COUNT = 5
TRAIN_FIELDS = ("instance_id", "repo", "base_commit", "problem_statement", "version", "created_at")
TASK_RULE = "sort dev ids by sha256(seed+id); take first 5 with public swebench eval image"
DOCKER_HUB_TIMEOUT_S = 30


def _image_name(instance_id: str) -> str:
    image_id = instance_id.replace("__", "_1776_").lower()
    return f"swebench/sweb.eval.x86_64.{image_id}:latest"


def _image_metadata(instance_id: str) -> dict[str, object] | None:
    image = _image_name(instance_id)
    repository, tag = image.removesuffix(":latest").split("/", 1)
    url = f"https://hub.docker.com/v2/repositories/{repository}/{tag}/tags/latest"
    request = Request(url, headers={"User-Agent": "capability-routing-pilot-prepare/1.0"})
    try:
        with urlopen(request, timeout=DOCKER_HUB_TIMEOUT_S) as response:
            payload = json.load(response)
    except HTTPError as error:
        if error.code == 404:
            return None
        raise
    size = payload.get("full_size")
    digest = payload.get("digest")
    if not isinstance(size, int) or not isinstance(digest, str):
        raise RuntimeError(f"Docker Hub returned incomplete metadata for {image}")
    return {
        "instance_id": instance_id,
        "image": image,
        "exists": True,
        "size": size,
        "digest": digest,
    }


def _load_rows() -> tuple[Mapping[str, object], ...]:
    dataset = load_dataset(DATASET_NAME, revision=DATASET_REVISION, split=DATASET_SPLIT)
    return tuple(dataset)


def _select_tasks(rows: Sequence[Mapping[str, object]]) -> tuple[dict[str, object], ...]:
    ordered = sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{SEED}{row['instance_id']}".encode("utf-8")
        ).hexdigest(),
    )
    selected: list[dict[str, object]] = []
    for row in ordered:
        instance_id = row.get("instance_id")
        if not isinstance(instance_id, str):
            raise RuntimeError("SWE-bench dev row is missing an instance_id")
        metadata = _image_metadata(instance_id)
        if metadata is None:
            continue
        selected.append(metadata)
        if len(selected) == TASK_COUNT:
            break
    if len(selected) != TASK_COUNT:
        raise RuntimeError(f"Found only {len(selected)} public eval images, expected {TASK_COUNT}")
    return tuple(selected)


def _train_row(row: Mapping[str, object]) -> dict[str, object]:
    result = {field: row[field] for field in TRAIN_FIELDS}
    if not all(isinstance(result[field], str) for field in TRAIN_FIELDS):
        raise RuntimeError("SWE-bench dev row contains a non-string training field")
    return result


def _write_outputs(
    output_dir: Path,
    rows: Sequence[Mapping[str, object]],
    tasks: Sequence[dict[str, object]],
) -> None:
    task_payload = {
        "frozen_at": FROZEN_AT,
        "dataset": DATASET_NAME,
        "split": DATASET_SPLIT,
        "seed": SEED,
        "rule": TASK_RULE,
        "tasks": list(tasks),
    }
    (output_dir / "tasks.json").write_text(json.dumps(task_payload, indent=2) + "\n")
    selected_ids = {str(task["instance_id"]) for task in tasks}
    train_rows = tuple(
        _train_row(row)
        for row in rows
        if isinstance(row.get("instance_id"), str) and row["instance_id"] in selected_ids
    )
    train_path = output_dir / "agent_dataset" / "train.jsonl"
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_text("".join(json.dumps(row) + "\n" for row in train_rows))


def prepare(check: bool = False) -> None:
    rows = _load_rows()
    tasks = _select_tasks(rows)
    if not check:
        _write_outputs(ROOT, rows, tasks)
        return
    with tempfile.TemporaryDirectory(prefix="routing-poc-prepare-") as temp_dir:
        generated_dir = Path(temp_dir)
        _write_outputs(generated_dir, rows, tasks)
        generated_tasks = json.loads((generated_dir / "tasks.json").read_text())
        existing_tasks = json.loads((ROOT / "tasks.json").read_text())
        generated_ids = [task["instance_id"] for task in generated_tasks["tasks"]]
        existing_ids = [task["instance_id"] for task in existing_tasks["tasks"]]
        if generated_ids != existing_ids:
            raise AssertionError("Regenerated task IDs do not match tasks.json")
        generated_train = (generated_dir / "agent_dataset" / "train.jsonl").read_bytes()
        existing_train = (ROOT / "agent_dataset" / "train.jsonl").read_bytes()
        if generated_train != existing_train:
            raise AssertionError("Regenerated train.jsonl is not byte-identical")
    print(f"Dataset preparation check passed for {len(tasks)} task IDs and byte-identical train.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    prepare(check=args.check)


if __name__ == "__main__":
    main()

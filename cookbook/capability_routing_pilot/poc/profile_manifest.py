import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import takewhile
from pathlib import Path
from typing import Final

from poc.profiles import PUBLIC_DEV_TENANT, SUPPORTED_CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
SEED: Final = "litellm-capability-profiling-v1"
MAX_PER_CATEGORY: Final = 30
ALL_CATEGORIES: Final = (
    "repository_debugging",
    "feature_implementation",
    "shell_build",
    "algorithms_data",
    "unknown",
)


@dataclass(frozen=True, slots=True)
class CandidateLabel:
    task_id: str
    primary_category: str
    secondary_label: str | None
    returned_model: str | None = None
    deployment_id: str | None = None
    call_id: str | None = None
    cost_usd: float | None = None


@dataclass(frozen=True, slots=True)
class ImageAvailability:
    task_id: str
    image: str
    exists: bool | None
    digest: str | None
    compressed_size_bytes: int | None
    error: str | None
    manifest_response_received: bool


def _hash_order(task_id: str) -> str:
    return hashlib.sha256(f"{SEED}{task_id}".encode()).hexdigest()


def _prefix_selected_ids(
    category: str,
    labels: Sequence[CandidateLabel],
    image_by_id: dict[str, ImageAvailability],
) -> tuple[str, ...]:
    ordered_ids: Final = tuple(
        sorted(
            (label.task_id for label in labels if label.primary_category == category),
            key=lambda task_id: (_hash_order(task_id), task_id),
        )
    )
    resolved_prefix: Final = tuple(
        takewhile(
            lambda task_id: task_id in image_by_id and image_by_id[task_id].exists is not None,
            ordered_ids,
        )
    )
    return tuple(
        task_id
        for task_id in resolved_prefix
        if image_by_id[task_id].exists is True
    )[:MAX_PER_CATEGORY]


def _candidate_record(
    task_id: str,
    label: CandidateLabel,
    image: ImageAvailability | None,
) -> dict[str, object]:
    status: Final = (
        "not_inspected"
        if image is None
        else "available"
        if image.exists is True
        else "unavailable"
        if image.exists is False
        else "check_error"
    )
    return {
        "task_id": task_id,
        "tenant_id": PUBLIC_DEV_TENANT,
        "primary_category": label.primary_category,
        "secondary_label": label.secondary_label,
        "returned_model": label.returned_model,
        "deployment_id": label.deployment_id,
        "call_id": label.call_id,
        "label_cost_usd": label.cost_usd,
        "image": (
            image.image
            if image is not None
            else f"swebench/sweb.eval.x86_64.{task_id.replace('__', '_1776_').lower()}"
        ),
        "image_available": image.exists if image is not None else None,
        "image_lookup_status": status,
        "image_digest": image.digest if image is not None else None,
        "compressed_size_bytes": image.compressed_size_bytes if image is not None else None,
        "image_error": image.error if image is not None else None,
        "manifest_response_received": image.manifest_response_received if image is not None else False,
    }


def build_draft_manifest(
    labels: Sequence[CandidateLabel],
    images: Sequence[ImageAvailability],
) -> dict[str, object]:
    label_by_id: Final = {label.task_id: label for label in labels}
    image_by_id: Final = {image.task_id: image for image in images}
    if len(label_by_id) != len(labels) or len(image_by_id) != len(images):
        raise ValueError("candidate labels and image results must have unique task IDs")
    if not set(image_by_id).issubset(label_by_id):
        raise ValueError("image results must belong to labeled candidates")
    records: Final = tuple(
        _candidate_record(task_id, label_by_id[task_id], image_by_id.get(task_id))
        for task_id in sorted(label_by_id)
    )
    category_counts: Final = {
        category: {
            "labeled": sum(label.primary_category == category for label in labels),
            "inspected": sum(
                label.primary_category == category and label.task_id in image_by_id
                for label in labels
            ),
            "with_image": sum(
                label.primary_category == category
                and label.task_id in image_by_id
                and image_by_id[label.task_id].exists is True
                for label in labels
            ),
            "image_unavailable": sum(
                label.primary_category == category
                and label.task_id in image_by_id
                and image_by_id[label.task_id].exists is False
                for label in labels
            ),
            "image_lookup_errors": sum(
                label.primary_category == category
                and label.task_id in image_by_id
                and image_by_id[label.task_id].exists is None
                for label in labels
            ),
            "manifest_responses": sum(
                label.primary_category == category
                and label.task_id in image_by_id
                and image_by_id[label.task_id].manifest_response_received
                for label in labels
            ),
            "not_inspected": sum(
                label.primary_category == category and label.task_id not in image_by_id
                for label in labels
            ),
            "selected": 0,
            "selection_status": "profiled" if category in SUPPORTED_CATEGORIES else "unsupported_category",
            "selected_ids": (),
        }
        for category in ALL_CATEGORIES
    }
    selected_ids: Final = {
        category: _prefix_selected_ids(category, labels, image_by_id)
        if category in SUPPORTED_CATEGORIES
        else ()
        for category in ALL_CATEGORIES
    }
    categories_with_selection: Final = {
        category: {
            **counts,
            "selected": len(selected_ids[category]),
            "target_shortfall": max(0, MAX_PER_CATEGORY - len(selected_ids[category]))
            if category in SUPPORTED_CATEGORIES
            else 0,
            "target_reached": len(selected_ids[category]) == MAX_PER_CATEGORY
            if category in SUPPORTED_CATEGORIES
            else False,
            "selected_ids": selected_ids[category],
        }
        for category, counts in category_counts.items()
    }
    return {
        "frozen": False,
        "tenant_id": PUBLIC_DEV_TENANT,
        "seed": SEED,
        "rule": f"Per supported category, sort by sha256(seed + task_id), stop at the first unresolved check, and select up to {MAX_PER_CATEGORY} confirmed images",
        "availability_bias": "Selection includes only tasks with confirmed public images; image lookup errors block later IDs in deterministic order",
        "profiled_categories": SUPPORTED_CATEGORIES,
        "unsupported_categories": ("shell_build", "algorithms_data"),
        "heldout_exclusion_list": (
            "pylint-dev__astroid-1164",
            "pylint-dev__astroid-2240",
            "pydicom__pydicom-793",
            "pvlib__pvlib-python-1191",
            "pylint-dev__astroid-927",
        ),
        "candidate_count": len(records),
        "error_exclusions": tuple(
            {
                "task_id": record["task_id"],
                "category": record["primary_category"],
                "status": "check_error",
            }
            for record in records
            if record["image_available"] is None and record["image_lookup_status"] == "check_error"
        ),
        "unknown_count": category_counts["unknown"]["labeled"],
        "image_unavailable_count": sum(record["image_available"] is False for record in records),
        "image_lookup_error_count": sum(record["image_lookup_status"] == "check_error" for record in records),
        "image_not_inspected_count": sum(record["image_lookup_status"] == "not_inspected" for record in records),
        "selected_count": sum(len(ids) for ids in selected_ids.values()),
        "categories": categories_with_selection,
        "candidates": records,
    }


def _load_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    rows: Final = tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name} must contain JSON objects")
    return tuple(row for row in rows if isinstance(row, dict))


def _load_labels(path: Path) -> tuple[CandidateLabel, ...]:
    rows: Final = _load_jsonl(path)
    labels: Final = tuple(
        CandidateLabel(
            task_id=row["task_id"],
            primary_category=row["primary_category"],
            secondary_label=row.get("secondary_label"),
            returned_model=row.get("returned_model"),
            deployment_id=row.get("deployment_id"),
            call_id=row.get("call_id"),
            cost_usd=row.get("cost_usd"),
        )
        for row in rows
        if isinstance(row.get("task_id"), str)
        and isinstance(row.get("primary_category"), str)
        and (row.get("secondary_label") is None or isinstance(row.get("secondary_label"), str))
        and (row.get("returned_model") is None or isinstance(row.get("returned_model"), str))
        and (row.get("deployment_id") is None or isinstance(row.get("deployment_id"), str))
        and (row.get("call_id") is None or isinstance(row.get("call_id"), str))
        and (
            row.get("cost_usd") is None
            or isinstance(row.get("cost_usd"), (int, float))
            and not isinstance(row.get("cost_usd"), bool)
        )
    )
    if len(labels) != len(rows):
        raise ValueError(f"{path.name} contains invalid candidate label fields")
    return labels


def _load_images(path: Path) -> tuple[ImageAvailability, ...]:
    values: Final = json.loads(path.read_text())
    if not isinstance(values, list) or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"{path.name} must contain a JSON array of image results")
    images: Final = tuple(
        ImageAvailability(
            task_id=value["task_id"],
            image=value["image"],
            exists=value["exists"],
            digest=value.get("digest"),
            compressed_size_bytes=value.get("compressed_size_bytes"),
            error=value.get("error"),
            manifest_response_received=value.get("manifest_response_received", False),
        )
        for value in values
        if isinstance(value.get("task_id"), str)
        and isinstance(value.get("image"), str)
        and (value.get("exists") is None or isinstance(value.get("exists"), bool))
        and (value.get("digest") is None or isinstance(value.get("digest"), str))
        and (
            value.get("compressed_size_bytes") is None
            or isinstance(value.get("compressed_size_bytes"), int)
            and not isinstance(value.get("compressed_size_bytes"), bool)
        )
        and (value.get("error") is None or isinstance(value.get("error"), str))
        and isinstance(value.get("manifest_response_received", False), bool)
    )
    if len(images) != len(values):
        raise ValueError(f"{path.name} contains invalid image result fields")
    return images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=ROOT / "profiling_labels.jsonl")
    parser.add_argument("--images", type=Path, default=ROOT / "profiling_candidate_images.json")
    parser.add_argument("--output", type=Path, default=ROOT / "profiling_manifest_draft.json")
    arguments = parser.parse_args()
    manifest: Final = build_draft_manifest(
        _load_labels(arguments.labels),
        _load_images(arguments.images),
    )
    arguments.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"manifest={arguments.output} candidates={manifest['candidate_count']} "
        f"selected={manifest['selected_count']} image_errors={manifest['image_lookup_error_count']}"
    )


if __name__ == "__main__":
    main()

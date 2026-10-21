import argparse
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
import fcntl
import hashlib
import json
import os
import stat
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import httpx

from poc import label
from poc.guarded_model import valid_cost
from poc.prepare import _load_rows

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_IDS: Final = frozenset(
    {
        "pylint-dev__astroid-1164",
        "pylint-dev__astroid-2240",
        "pydicom__pydicom-793",
        "pvlib__pvlib-python-1191",
        "pylint-dev__astroid-927",
    }
)
LABEL_BUDGET_USD = 5.0
LABEL_RESERVATION_USD = 1.0
LABEL_WORKERS = 4
IMAGE_WORKERS = 4
IMAGE_REQUEST_INTERVAL_S = 1.0
IMAGE_MAX_BACKOFF_S = 30.0
REGISTRY_AUTH_URL = "https://auth.docker.io/token"
REGISTRY_URL = "https://registry-1.docker.io"
MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


@dataclass(frozen=True, slots=True)
class Candidate:
    task_id: str
    problem_statement: str
    image: str


@dataclass(frozen=True, slots=True)
class LabelResult:
    task_id: str
    primary_category: str
    raw_response: str
    cost_usd: float | None
    call_id: str | None
    returned_model: str | None
    deployment_id: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class ImageResult:
    task_id: str
    image: str
    exists: bool | None
    digest: str | None
    compressed_size_bytes: int | None
    error: str | None
    manifest_response_received: bool


class LabelBudget:
    def __init__(self, committed_spend_usd: float) -> None:
        self._lock = threading.Lock()
        self._committed_spend_usd = committed_spend_usd
        self._reserved_usd = 0.0
        self._stopped = False

    def reserve(self) -> bool:
        with self._lock:
            if self._stopped or self._committed_spend_usd + self._reserved_usd + LABEL_RESERVATION_USD > LABEL_BUDGET_USD:
                return False
            self._reserved_usd += LABEL_RESERVATION_USD
            return True

    def settle(self, cost_usd: float | None, failed: bool) -> bool:
        with self._lock:
            self._reserved_usd -= LABEL_RESERVATION_USD
            if cost_usd is None or failed:
                self._stopped = True
                return False
            self._committed_spend_usd += cost_usd
            if self._committed_spend_usd > LABEL_BUDGET_USD:
                self._stopped = True
            return not self._stopped

    def stop(self) -> None:
        with self._lock:
            self._stopped = True

    @property
    def committed_spend_usd(self) -> float:
        with self._lock:
            return self._committed_spend_usd


class RegistryPacer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_request_at = 0.0
        self._blocked = False

    def wait(self) -> bool:
        while True:
            with self._lock:
                if self._blocked:
                    return False
                now: Final = time.monotonic()
                delay: Final = max(0.0, self._next_request_at - now)
                if delay == 0.0:
                    self._next_request_at = now + IMAGE_REQUEST_INTERVAL_S
                    return True
            time.sleep(delay)

    def backoff(self, delay_s: float) -> None:
        with self._lock:
            self._next_request_at = max(self._next_request_at, time.monotonic() + delay_s)

    def block(self) -> None:
        with self._lock:
            self._blocked = True

    @property
    def blocked(self) -> bool:
        with self._lock:
            return self._blocked


def _registry_get(
    client: httpx.Client,
    pacer: RegistryPacer,
    url: str,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    method: str = "GET",
) -> httpx.Response | None:
    if not pacer.wait():
        return None
    first: Final = client.request(method, url, params=params, headers=headers)
    if first.status_code != 429:
        return first
    delay: Final = _backoff_seconds(first)
    if delay is None:
        pacer.block()
        return first
    pacer.backoff(delay)
    if not pacer.wait():
        return None
    second: Final = client.request(method, url, params=params, headers=headers)
    if second.status_code == 429:
        pacer.block()
    return second


def _backoff_seconds(response: httpx.Response) -> float | None:
    retry_after: Final = response.headers.get("Retry-After")
    if retry_after is None:
        return 1.0
    try:
        delay = float(retry_after)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
        except (TypeError, ValueError, OverflowError):
            return 1.0
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=UTC)
        delay = (retry_at - datetime.now(UTC)).total_seconds()
    return None if delay > IMAGE_MAX_BACKOFF_S else max(1.0, delay)


def _candidates() -> tuple[Candidate, ...]:
    return tuple(
        Candidate(
            task_id=task_id,
            problem_statement=label._sanitize_label_text(problem_statement),
            image=f"swebench/sweb.eval.x86_64.{task_id.replace('__', '_1776_').lower()}",
        )
        for row in _load_rows()
        if isinstance((task_id := row.get("instance_id")), str)
        and task_id not in HELDOUT_IDS
        and isinstance((problem_statement := row.get("problem_statement")), str)
    )


def _jsonl_rows(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    rows = tuple(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name} must contain JSON objects")
    return tuple(row for row in rows if isinstance(row, dict))


def _image_candidates() -> tuple[Candidate, ...]:
    rows: Final = _jsonl_rows(ROOT / "profiling_labels.jsonl")
    category_by_id: Final = {
        task_id: category
        for row in rows
        if isinstance((task_id := row.get("task_id")), str)
        and task_id not in HELDOUT_IDS
        and isinstance((category := row.get("primary_category")), str)
    }
    if len(category_by_id) != len(rows):
        raise ValueError("profiling_labels.jsonl contains duplicate or invalid image candidates")
    return tuple(
        Candidate(
            task_id=task_id,
            problem_statement="",
            image=f"swebench/sweb.eval.x86_64.{task_id.replace('__', '_1776_').lower()}",
        )
        for task_id in category_by_id
    )


def _existing_label_state(path: Path) -> tuple[float, frozenset[str]]:
    rows: Final = _jsonl_rows(path)
    task_ids: Final = tuple(row.get("task_id") for row in rows)
    if not all(isinstance(task_id, str) for task_id in task_ids):
        raise ValueError("profiling_labels.jsonl contains a missing or invalid task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("profiling_labels.jsonl contains duplicate task IDs")
    costs: Final = tuple(row.get("cost_usd") for row in rows)
    if not all(valid_cost(cost) for cost in costs):
        raise ValueError("profiling_labels.jsonl contains incomplete cost accounting")
    return sum(float(cost) for cost in costs), frozenset(task_id for task_id in task_ids if isinstance(task_id, str))


def _label_one(client: httpx.Client, candidate: Candidate) -> LabelResult:
    result: Final = label._offline_label(
        client,
        candidate.task_id,
        candidate.problem_statement,
        record_auxiliary=False,
    )
    primary: Final = result.get("label")
    raw_response: Final = result.get("raw_label_text")
    cost: Final = result.get("cost_usd")
    call_id: Final = result.get("call_id")
    returned_model: Final = result.get("response_model")
    deployment_id: Final = result.get("deployment_id")
    error: Final = result.get("error")
    return LabelResult(
        task_id=candidate.task_id,
        primary_category=primary if isinstance(primary, str) else "unknown",
        raw_response=raw_response if isinstance(raw_response, str) else "",
        cost_usd=float(cost) if valid_cost(cost) else None,
        call_id=call_id if isinstance(call_id, str) else None,
        returned_model=returned_model if isinstance(returned_model, str) else None,
        deployment_id=deployment_id if isinstance(deployment_id, str) else None,
        error=error if isinstance(error, str) else None,
    )


def _append_label(path: Path, result: LabelResult) -> None:
    row: Final = {
        "task_id": result.task_id,
        "tenant_id": "public_dev_poc",
        "primary_category": result.primary_category,
        "secondary_label": None,
        "raw_response": result.raw_response,
        "cost_usd": result.cost_usd,
        "call_id": result.call_id,
        "returned_model": result.returned_model,
        "deployment_id": result.deployment_id,
    }
    if result.error is not None:
        row["error"] = result.error
    with path.open("a") as output:
        fcntl.flock(output, fcntl.LOCK_EX)
        output.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        output.flush()
        fcntl.flock(output, fcntl.LOCK_UN)


def label_candidates() -> None:
    candidates: Final = _candidates()
    path: Final = ROOT / "profiling_labels.jsonl"
    committed_spend, existing_ids = _existing_label_state(path)
    pending_candidates: Final = tuple(candidate for candidate in candidates if candidate.task_id not in existing_ids)
    if not pending_candidates:
        print(f"labels_complete=true labeled={len(existing_ids)} spend_usd={committed_spend:.6f}")
        return
    key_path: Final = ROOT / ".private" / "profiling_key"
    if stat.S_IMODE(key_path.stat().st_mode) != 0o600:
        raise PermissionError("profiling key must have mode 0600")
    os.environ["LITELLM_PROXY_API_KEY"] = key_path.read_text().strip()
    if not os.environ["LITELLM_PROXY_API_KEY"]:
        raise ValueError("profiling key is empty")
    budget: Final = LabelBudget(committed_spend)
    pending: dict[Future[LabelResult], Candidate] = {}
    next_index = 0
    stopped_for_error = False
    with httpx.Client(
        transport=httpx.HTTPTransport(retries=0),
        timeout=label.REQUEST_TIMEOUT_S,
    ) as client, ThreadPoolExecutor(max_workers=LABEL_WORKERS) as executor:
        while next_index < len(pending_candidates) or pending:
            while next_index < len(pending_candidates) and len(pending) < LABEL_WORKERS:
                if not budget.reserve():
                    budget.stop()
                    break
                candidate: Final = pending_candidates[next_index]
                pending[executor.submit(_label_one, client, candidate)] = candidate
                next_index += 1
            if not pending:
                break
            future: Final = next(as_completed(tuple(pending)))
            candidate = pending.pop(future)
            try:
                result: Final = future.result()
            except Exception as exception:
                result = LabelResult(
                    candidate.task_id,
                    "unknown",
                    "",
                    None,
                    None,
                    None,
                    None,
                    type(exception).__name__,
                )
            _append_label(path, result)
            budget.settle(result.cost_usd, result.error is not None)
            stopped_for_error = stopped_for_error or result.error is not None or result.cost_usd is None
    final_rows: Final = _jsonl_rows(path)
    complete = len(final_rows) == len(candidates)
    print(
        f"labels_complete={str(complete).lower()} labeled={len(final_rows)} candidates={len(candidates)} "
        f"spend_usd={budget.committed_spend_usd:.6f} "
        f"stop_reason={'error_or_cost_accounting' if stopped_for_error else 'budget_or_complete'}"
    )


def _as_mapping(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _manifest_digest(response: httpx.Response) -> str:
    header_digest: Final = response.headers.get("Docker-Content-Digest")
    if header_digest is not None:
        return header_digest
    return f"sha256:{hashlib.sha256(response.content).hexdigest()}"


def _compressed_size(payload: dict[str, object]) -> int:
    layers: Final = payload.get("layers")
    layer_sizes: Final = (
        tuple(
            int(layer["size"])
            for item in layers
            if (layer := _as_mapping(item)) is not None
            and isinstance(layer.get("size"), int)
            and not isinstance(layer.get("size"), bool)
        )
        if isinstance(layers, list)
        else ()
    )
    return sum(layer_sizes)


def _image_error(
    candidate: Candidate,
    error: str,
    manifest_response_received: bool = False,
) -> ImageResult:
    return ImageResult(candidate.task_id, candidate.image, None, None, None, error, manifest_response_received)


def _lookup_image_response(
    client: httpx.Client,
    pacer: RegistryPacer,
    candidate: Candidate,
) -> ImageResult:
    auth: Final = _registry_get(
        client,
        pacer,
        REGISTRY_AUTH_URL,
        {"service": "registry.docker.io", "scope": f"repository:{candidate.image}:pull"},
    )
    if auth is None:
        return _image_error(candidate, "registry_rate_limited")
    if auth.status_code != 200:
        return _image_error(candidate, f"authorization_http_{auth.status_code}")
    auth_payload: Final = _as_mapping(auth.json())
    token: Final = (
        auth_payload.get("token") or auth_payload.get("access_token")
        if auth_payload is not None
        else None
    )
    if not isinstance(token, str):
        return _image_error(candidate, "authorization_response_missing_token")
    registry_headers: Final = {"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT}
    head: Final = _registry_get(
        client,
        pacer,
        f"{REGISTRY_URL}/v2/{candidate.image}/manifests/latest",
        headers=registry_headers,
        method="HEAD",
    )
    if head is None:
        return _image_error(candidate, "registry_rate_limited")
    if head.status_code == 404:
        return ImageResult(candidate.task_id, candidate.image, False, None, None, None, True)
    if head.status_code != 200:
        return _image_error(candidate, f"manifest_head_http_{head.status_code}", True)
    response: Final = _registry_get(
        client,
        pacer,
        f"{REGISTRY_URL}/v2/{candidate.image}/manifests/latest",
        headers=registry_headers,
    )
    if response is None:
        return _image_error(candidate, "manifest_body_rate_limited", True)
    if response.status_code == 404:
        return ImageResult(candidate.task_id, candidate.image, False, None, None, None, True)
    if response.status_code != 200:
        return _image_error(candidate, f"manifest_body_http_{response.status_code}", True)
    payload: Final = _as_mapping(response.json())
    if payload is None:
        return _image_error(candidate, "manifest_response_not_object", True)
    media_type: Final = payload.get("mediaType")
    if isinstance(media_type, str) and ("index" in media_type or "manifest.list" in media_type):
        descriptors: Final = payload.get("manifests")
        amd64: Final = (
            next(
                (
                    descriptor
                    for item in descriptors
                    if (descriptor := _as_mapping(item)) is not None
                    and (platform := _as_mapping(descriptor.get("platform"))) is not None
                    and platform.get("os") == "linux"
                    and platform.get("architecture") in {"amd64", "x86_64"}
                ),
                None,
            )
            if isinstance(descriptors, list)
            else None
        )
        if amd64 is None or not isinstance(amd64.get("digest"), str):
            return _image_error(candidate, "linux_amd64_manifest_missing", True)
        child: Final = _registry_get(
            client,
            pacer,
            f"{REGISTRY_URL}/v2/{candidate.image}/manifests/{amd64['digest']}",
            headers={"Authorization": f"Bearer {token}", "Accept": MANIFEST_ACCEPT},
        )
        if child is None:
            return _image_error(candidate, "registry_rate_limited")
        if child.status_code == 404:
            return ImageResult(candidate.task_id, candidate.image, False, _manifest_digest(response), None, None, True)
        if child.status_code != 200:
            return _image_error(candidate, f"platform_manifest_http_{child.status_code}", True)
        child_payload: Final = _as_mapping(child.json())
        if child_payload is None:
            return _image_error(candidate, "platform_manifest_response_not_object", True)
        return ImageResult(
            candidate.task_id,
            candidate.image,
            True,
            _manifest_digest(child),
            _compressed_size(child_payload),
            None,
            True,
        )
    return ImageResult(
        candidate.task_id,
        candidate.image,
        True,
        _manifest_digest(response),
        _compressed_size(payload),
        None,
        True,
    )


def _lookup_image(client: httpx.Client, pacer: RegistryPacer, candidate: Candidate) -> ImageResult:
    try:
        return _lookup_image_response(client, pacer, candidate)
    except Exception as exception:
        return _image_error(candidate, type(exception).__name__)


def discover_images() -> None:
    candidates: Final = _image_candidates()
    category_by_id: Final = {
        task_id: category
        for row in _jsonl_rows(ROOT / "profiling_labels.jsonl")
        if isinstance((task_id := row.get("task_id")), str)
        and isinstance((category := row.get("primary_category")), str)
    }
    ordered_categories: Final = tuple(
        (
            category,
            tuple(
                sorted(
                    (
                        candidate
                        for candidate in candidates
                        if category_by_id.get(candidate.task_id) == category
                    ),
                    key=lambda candidate: (
                        hashlib.sha256(f"litellm-capability-profiling-v1{candidate.task_id}".encode()).hexdigest(),
                        candidate.task_id,
                    ),
                )
            ),
        )
        for category in ("repository_debugging", "feature_implementation")
    )
    pacer: Final = RegistryPacer()
    with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=30.0) as client:
        with ThreadPoolExecutor(max_workers=IMAGE_WORKERS) as executor:
            results: tuple[ImageResult, ...] = ()
            for _, category_candidates in ordered_categories:
                available = 0
                offset = 0
                while offset < len(category_candidates) and available < 30 and not pacer.blocked:
                    batch: Final = category_candidates[offset : offset + IMAGE_WORKERS]
                    batch_results: Final = tuple(
                        executor.map(partial(_lookup_image, client, pacer), batch)
                    )
                    results += batch_results
                    available += sum(result.exists is True for result in batch_results)
                    offset += len(batch)
    path: Final = ROOT / "profiling_candidate_images.json"
    path.write_text(
        json.dumps(
            [
                {
                    "task_id": result.task_id,
                    "image": result.image,
                    "exists": result.exists,
                    "digest": result.digest,
                    "compressed_size_bytes": result.compressed_size_bytes,
                    "error": result.error,
                    "manifest_response_received": result.manifest_response_received,
                }
                for result in results
            ],
            indent=2,
        )
        + "\n"
    )
    print(
        f"image_candidates={len(results)} "
        f"image_available={sum(result.exists is True for result in results)} "
        f"image_unavailable={sum(result.exists is False for result in results)} "
        f"image_errors={sum(result.exists is None for result in results)} "
        f"manifest_responses={sum(result.manifest_response_received for result in results)} "
        f"candidate_records={len(results)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("images", "labels"))
    operation: Final = parser.parse_args().operation
    if operation == "images":
        discover_images()
    else:
        label_candidates()


if __name__ == "__main__":
    main()

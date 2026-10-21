import asyncio
import json
import os
import re
import stat
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx

from poc.heldout_baseline import (
    GatewayJevClassifierClient,
    active_tiers,
    choose_baseline_route,
    source_config,
    source_router,
)
from poc.label import JEV_CRITERIA, JEV_MODEL, _json_response, _response_metadata, parse_jev_response
from poc.policy import SpendSuccessCell, choose_capability_model
from poc.run_pilot import (
    AttemptPaths,
    DispatchGuard,
    _attempt,
    append_jsonl,
    committed_spend_usd,
)

ROOT = Path(__file__).resolve().parents[1]
HELDOUT_KEY = ROOT / ".private" / "heldout_key"
TASK_DATA = ROOT / "heldout_agent_dataset" / "train.jsonl"
TASKS = ROOT / "heldout_tasks.json"
MAP = ROOT / "heldout_capability_map.json"
RUNS = ROOT / "heldout_runs.jsonl"
RAW = ROOT / "heldout_raw"
LABELS = ROOT / "heldout_labels.json"
AUX = ROOT / "heldout_costs_aux.jsonl"
IMAGE_PROVENANCE = ROOT / "heldout_image_provenance.json"
MODEL_PATH = ROOT / "poc" / "pilot_models.json"
EVAL_DATASET = str(ROOT / "eval_heldout" / "swebench_dev_c6fe717.json")
LABEL_TIMEOUT_S = 120.0
MODELS = ("haiku-4.5", "sonnet-5", "opus-5")


def _jsonl(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text().splitlines())


def _sanitize(statement: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", statement).strip()


def _spend_map(payload: Mapping[str, object]) -> dict[str, dict[str, SpendSuccessCell]]:
    scores = payload["scores"]
    if not isinstance(scores, dict):
        raise ValueError("invalid held-out capability map")
    return {
        str(category): {
            str(model): SpendSuccessCell(
                total_spend_usd=float(cell["total_spend_usd"]),
                successes=int(cell["successes"]),
                graded_tasks=int(cell["graded_tasks"]),
                score_usd_per_success=(
                    float(cell["score_usd_per_success"])
                    if cell["score_usd_per_success"] is not None
                    else None
                ),
                sparse=bool(cell["sparse"]),
            )
            for model, cell in models.items()
        }
        for category, models in scores.items()
        if isinstance(models, dict)
    }


def _image_overlay_ids() -> dict[str, str]:
    payload: Final = json.loads(IMAGE_PROVENANCE.read_text())
    images: Final = payload.get("images")
    if not isinstance(images, list):
        raise ValueError("invalid held-out image provenance")
    return {
        image["image"]: image["local_overlay_image_id"]
        for image in images
        if isinstance(image, dict)
        and isinstance(image.get("image"), str)
        and isinstance(image.get("local_overlay_image_id"), str)
    }


def _key() -> str:
    if stat.S_IMODE(HELDOUT_KEY.stat().st_mode) != 0o600:
        raise PermissionError("held-out key must have mode 0600")
    return HELDOUT_KEY.read_text().strip()


def _classifier_record(
    task_id: str,
    arm: str,
    call_id: str | None,
    cost: float | None,
    latency: float,
    label: str | None,
    error: str | None,
) -> dict[str, object]:
    row: Final = {
        "kind": "jev_classifier",
        "task_id": task_id,
        "arm": arm,
        "model": JEV_MODEL,
        "call_id": call_id,
        "cost_usd": cost,
        "latency_s": latency,
        "label": label,
        "error": error,
    }
    append_jsonl(AUX, row)
    return row


def _capability_call(task_id: str, statement: str, api_key: str, base_url: str) -> dict[str, object]:
    body: Final = {
        "model": JEV_MODEL,
        "state": _sanitize(statement),
        "questions": {
            "capability": {
                "type": "choice",
                "instructions": "Which capability does solving this software task mainly require?",
                "criteria": JEV_CRITERIA,
            }
        },
    }
    started: Final = time.perf_counter()
    response: httpx.Response | None = None
    payload: dict[str, object] = {}
    error: str | None = None
    try:
        with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=LABEL_TIMEOUT_S) as client:
            response = client.post(
                f"{base_url}/v1/systemone",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=body,
            )
        response.raise_for_status()
        payload = _json_response(response)
    except (httpx.HTTPError, ValueError) as exception:
        error = type(exception).__name__
    latency: Final = time.perf_counter() - started
    metadata: Final = (
        _response_metadata(response, payload, latency)
        if response is not None
        else {"call_id": None, "cost_usd": None}
    )
    label, confidence, probabilities = parse_jev_response(payload)
    if error is not None:
        label = None
    return {
        **_classifier_record(
            task_id,
            "capability",
            metadata.get("call_id") if isinstance(metadata.get("call_id"), str) else None,
            metadata.get("cost_usd") if isinstance(metadata.get("cost_usd"), (int, float)) else None,
            latency,
            label,
            error,
        ),
        "confidence": confidence,
        "probabilities": probabilities,
    }


def _baseline_call(task_id: str, statement: str, api_key: str, base_url: str) -> dict[str, object]:
    async def classify() -> tuple[object, GatewayJevClassifierClient]:
        async with httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(retries=0),
            timeout=LABEL_TIMEOUT_S,
        ) as client:
            gateway_client = GatewayJevClassifierClient(client, base_url, api_key)
            router, _ = source_router(gateway_client)
            outcome = await router.aclassify(prompt=_sanitize(statement))
            return outcome, gateway_client

    outcome, gateway_client = asyncio.run(classify())
    call: Final = gateway_client.last_call
    tier: Final = getattr(outcome.tier, "value", str(outcome.tier))
    failed: Final = outcome.cause != "jev_classifier" or call is None or call.error is not None
    route: Final = choose_baseline_route(tier, active_tiers(source_config()), failed)
    return {
        **_classifier_record(
            task_id,
            "baseline",
            call.call_id if call is not None else None,
            call.cost_usd if call is not None else None,
            call.latency_s if call is not None else 0.0,
            tier,
            call.error if call is not None else "missing_classifier_call",
        ),
        "tier": tier,
        "model": route.model,
        "reason": route.reason,
        "classifier_failed": failed,
    }


def execute_schedule(
    task_ids: tuple[str, ...],
    schedule: tuple[tuple[str, tuple[str, str]], ...],
    map_sha: str,
    prompt_sha: str,
    workers: int,
    budget: float,
) -> None:
    if RUNS.exists() or AUX.exists() or LABELS.exists():
        raise FileExistsError("held-out run outputs already exist")
    api_key: Final = _key()
    base_url: Final = os.environ.get("LITELLM_GATEWAY_BASE_URL", "https://gateway.litellm-sandbox.ai").rstrip("/")
    manifest: Final = json.loads(TASKS.read_text())
    task_manifest: Final = {task["instance_id"]: task for task in manifest["tasks"]}
    statements: Final = {row["instance_id"]: row for row in _jsonl(TASK_DATA)}
    score_map: Final = _spend_map(json.loads(MAP.read_text()))
    model_payload: Final = json.loads(MODEL_PATH.read_text())
    solvers: Final = model_payload["solvers"]
    commits: Final = {task_id: statements[task_id]["base_commit"] for task_id in task_ids}
    images: Final = {task["image"]: task["upstream_registry_digest"] for task in task_manifest.values()}
    image_overlay_ids: Final = _image_overlay_ids()
    attempt_paths: Final = AttemptPaths(
        run_root=RAW,
        dataset_root=ROOT / "heldout_agent_dataset",
        eval_dataset=EVAL_DATASET,
        base_commits=commits,
        image_digests=images,
        image_overlay_ids=image_overlay_ids,
        dataset_metadata={"agent_dataset": str(TASK_DATA), "eval_revision": "c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2"},
    )
    RAW.mkdir(exist_ok=True)
    base_env: Final = {
        **os.environ,
        "LITELLM_PROXY_API_KEY": api_key,
        "PYTHONPATH": str(ROOT),
        "MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT": "1",
    }
    guard: Final = DispatchGuard(
        RAW / "dispatch_stops.jsonl",
        budget=budget,
        committed_spend=lambda: committed_spend_usd(RUNS, AUX),
    )
    output_lock: Final = threading.Lock()
    labels: list[dict[str, object]] = []

    def run_task(task_id: str, arms: tuple[str, str], task_index: int) -> None:
        task: Final = task_manifest[task_id]
        data: Final = statements[task_id]
        for order_index, arm in enumerate(arms):
            if not guard.acquire(task_id):
                return
            try:
                classification: Final = (
                    _capability_call(task_id, data["problem_statement"], api_key, base_url)
                    if arm == "capability"
                    else _baseline_call(task_id, data["problem_statement"], api_key, base_url)
                )
            finally:
                guard.release()
            classifier_cost: Final = classification.get("cost_usd")
            if not isinstance(classifier_cost, (int, float)) or classifier_cost < 0:
                guard.stop_for_accounting(task_id)
                classification["error"] = "invalid_or_missing_classifier_cost"
                with output_lock:
                    labels.append({"task_id": task_id, "arm": arm, **classification})
                append_jsonl(
                    RUNS,
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "order_index": order_index,
                        "task_order": task_index,
                        "classifier": classification,
                        "decision": "unproven",
                        "reason": "cost_accounting_incomplete",
                        "map_sha256": map_sha,
                        "baseline_prompt_sha256": prompt_sha,
                        "solver_executed": False,
                    },
                )
                return
            decision: Final = (
                choose_capability_model(
                    classification.get("label") if isinstance(classification.get("label"), str) else None,
                    score_map,
                    MODELS,
                    bool(classification.get("error")),
                )
                if arm == "capability"
                else None
            )
            model: Final = (
                decision.model
                if decision is not None
                else str(classification["model"])
            )
            reason: Final = decision.reason if decision is not None else str(classification["reason"])
            with output_lock:
                labels.append({"task_id": task_id, "arm": arm, **classification})
            if model is None:
                append_jsonl(
                    RUNS,
                    {
                        "task_id": task_id,
                        "arm": arm,
                        "order_index": order_index,
                        "task_order": task_index,
                        "classifier": classification,
                        "decision": "unproven",
                        "reason": reason,
                        "map_sha256": map_sha,
                        "baseline_prompt_sha256": prompt_sha,
                        "classifier_cost_usd": classifier_cost,
                        "arm_cost_usd": classifier_cost,
                        "solver_executed": False,
                    },
                )
                continue
            def solve(attempt_number: int) -> dict[str, object] | None:
                if not guard.acquire(task_id):
                    return None
                try:
                    return _attempt(
                        task_id,
                        model,
                        attempt_number,
                        {**task, **data},
                        solvers[model],
                        None,
                        classification.get("label"),
                        base_env,
                        base_url,
                        attempt_paths,
                    )
                finally:
                    guard.release()

            first_result: Final = solve(1)
            if first_result is None:
                return
            retry_result: Final = solve(2) if first_result.get("retry_eligible") is True else None
            attempts: Final = (first_result,) if retry_result is None else (first_result, retry_result)
            result: Final = attempts[-1]
            solver_cost: Final = sum(float(attempt.get("cost_usd") or 0.0) for attempt in attempts)
            row: Final = {
                **result,
                "cost_usd": solver_cost,
                "solver_attempts": attempts,
                "arm": arm,
                "order_index": order_index,
                "task_order": task_index,
                "classifier": classification,
                "decision": model,
                "reason": reason,
                "map_sha256": map_sha,
                "baseline_prompt_sha256": prompt_sha,
                "classifier_cost_usd": classifier_cost,
                "arm_cost_usd": float(classifier_cost) + solver_cost,
                "solver_executed": True,
            }
            append_jsonl(RUNS, row)
            if result.get("error_class") == "accounting" or result.get("cost_incomplete"):
                guard.stop_for_accounting(task_id)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = tuple(
            executor.submit(run_task, task_id, arms, index)
            for index, (task_id, arms) in enumerate(schedule)
        )
        for future in futures:
            future.result()
    LABELS.write_text(json.dumps(labels, indent=2) + "\n")

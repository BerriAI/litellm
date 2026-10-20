import argparse
import fcntl
import json
import math
import os
import re
import stat
import time
from datetime import UTC, datetime
from json import JSONDecodeError, JSONDecoder
from pathlib import Path

import httpx

from poc.policy import CATEGORIES, parse_category

ROOT = Path(__file__).resolve().parents[1]
BASE_URL = os.environ.get("LITELLM_GATEWAY_BASE_URL", "https://gateway.litellm-sandbox.ai").rstrip("/")
LABEL_MODEL = "openai/gpt-5.4-mini-2026-03-17"
JEV_MODEL = "typesafe/jev-latest"
REQUEST_TIMEOUT_S = 120.0
SYSTEM_PROMPT = (
    "You label software engineering tasks by the main capability needed to solve them. Use only the problem "
    "statement. Answer with JSON only: {\"label\": <one of repository_debugging, feature_implementation, "
    "shell_build, algorithms_data, unknown>, \"rationale\": <one sentence>}. repository_debugging: diagnose and "
    "fix incorrect behavior in an existing codebase. feature_implementation: add new functionality, options or "
    "APIs. shell_build: build, packaging, environment, CI or shell tooling. algorithms_data: algorithmic, numerical "
    "or data-processing logic is the core difficulty. unknown: cannot tell from the statement."
)
JEV_CRITERIA = {
    "repository_debugging": "diagnose and fix incorrect behavior in an existing codebase",
    "feature_implementation": "add new functionality, options or APIs",
    "shell_build": "build, packaging, environment, CI or shell tooling",
    "algorithms_data": "algorithmic, numerical or data-processing logic is the core difficulty",
    "unknown": "cannot tell from the statement",
}


def _first_json_object(text: str) -> dict[str, object] | None:
    decoder = JSONDecoder()
    for index, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_label_response(text: str) -> tuple[str, str | None, bool]:
    payload = _first_json_object(text)
    if payload is None or not isinstance(payload.get("label"), str):
        return "unknown", None, False
    raw_label = payload["label"]
    label = parse_category(raw_label)
    if label == "unknown" and raw_label.strip().lower() != "unknown":
        return "unknown", None, False
    rationale = payload.get("rationale")
    if rationale is not None and not isinstance(rationale, str):
        return "unknown", None, False
    return label, rationale, True


def _as_mapping(value: object) -> dict[str, object] | None:
    return value if isinstance(value, dict) else None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def parse_jev_response(
    payload: object,
) -> tuple[str | None, float | None, dict[str, float]]:
    response = _as_mapping(payload)
    answers = None if response is None else _as_mapping(response.get("answers"))
    answer = None if answers is None else _as_mapping(answers.get("capability"))
    if answer is None and response is not None:
        answer = _as_mapping(response.get("capability")) or _as_mapping(response.get("answer"))
    if answer is None:
        return None, None, {}
    choice = answer.get("choice", answer.get("answer", answer.get("value")))
    if not isinstance(choice, str):
        return None, None, {}
    label = parse_category(choice)
    if label == "unknown" and choice.strip().lower() != "unknown":
        return None, None, {}
    confidence = _finite_number(answer.get("confidence"))
    raw_probabilities = _as_mapping(answer.get("probabilities"))
    probabilities = {
        category: number
        for key, raw_value in (raw_probabilities or {}).items()
        if isinstance(key, str)
        and (category := parse_category(key)) in CATEGORIES
        and (category != "unknown" or key.strip().lower() == "unknown")
        and (number := _finite_number(raw_value)) is not None
    }
    return label, confidence, probabilities


def _read_api_key() -> str:
    path = ROOT / ".private" / "pilot_key"
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise PermissionError("pilot key must have mode 0600")
    key = path.read_text().strip()
    if not key:
        raise ValueError("pilot key is empty")
    return key


def _headers() -> dict[str, str]:
    api_key = os.environ.get("LITELLM_PROXY_API_KEY")
    if not api_key:
        raise RuntimeError("LITELLM_PROXY_API_KEY is not set")
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def _json_response(response: httpx.Response) -> dict[str, object]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _response_metadata(response: httpx.Response, payload: dict[str, object], latency_s: float) -> dict[str, object]:
    headers = response.headers
    usage = payload.get("usage")
    raw_cost = headers.get("x-litellm-response-cost") or headers.get("llm_provider-x-litellm-response-cost")
    cost = _finite_number(raw_cost) if isinstance(raw_cost, (int, float)) else None
    if isinstance(raw_cost, str):
        try:
            cost = _finite_number(float(raw_cost))
        except ValueError:
            cost = None
    return {
        "call_id": headers.get("x-litellm-call-id"),
        "response_model": payload.get("model") or headers.get("x-litellm-response-model"),
        "reported_version": payload.get("version")
        or headers.get("x-litellm-model-version")
        or headers.get("x-litellm-version"),
        "cost_usd": cost,
        "latency_s": latency_s,
        "usage": usage if isinstance(usage, dict) else {},
    }


def _append_auxiliary_call(kind: str, task_id: str, model: str, metadata: dict[str, object]) -> None:
    path = ROOT / "costs_aux.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "kind": kind,
        "task_id": task_id,
        "model": model,
        "cost_usd": metadata["cost_usd"],
        "call_id": metadata["call_id"],
        "ts": datetime.now(UTC).isoformat(),
    }
    with path.open("a") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write(json.dumps(row, separators=(",", ":")) + "\n")
        file.flush()
        fcntl.flock(file, fcntl.LOCK_UN)


def _sanitize_label_text(text: str) -> str:
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()[:1200]


def _problem_statement_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            fragment.get("text", "")
            for fragment in content
            if isinstance(fragment, dict) and isinstance(fragment.get("text"), str)
        )
    return ""


def _offline_label(client: httpx.Client, task_id: str, problem_statement: str) -> dict[str, object]:
    body = {
        "model": LABEL_MODEL,
        "max_completion_tokens": 400,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Problem statement:\n<<<\n{problem_statement}\n>>>"},
        ],
    }
    started = time.perf_counter()
    response: httpx.Response | None = None
    payload: dict[str, object] = {}
    error: str | None = None
    try:
        response = client.post(f"{BASE_URL}/v1/chat/completions", headers=_headers(), json=body)
        payload = _json_response(response)
        if response.is_error:
            error = f"http_{response.status_code}"
    except httpx.HTTPError as exception:
        error = type(exception).__name__
    latency_s = time.perf_counter() - started
    metadata = (
        _response_metadata(response, payload, latency_s)
        if response is not None
        else {"call_id": None, "response_model": None, "reported_version": None, "cost_usd": None,
              "latency_s": latency_s, "usage": {}}
    )
    choices = payload.get("choices")
    choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice, dict) else {}
    raw_text = _problem_statement_text(message.get("content")) if isinstance(message, dict) else ""
    label, rationale, parse_ok = parse_label_response(raw_text)
    if error is not None:
        label, rationale, parse_ok = "unknown", None, False
    _append_auxiliary_call("offline_label", task_id, LABEL_MODEL, metadata)
    return {
        **metadata,
        "label": label,
        "rationale": rationale,
        "parse_ok": parse_ok,
        "raw_label_text": _sanitize_label_text(raw_text),
        "error": error,
    }


def _jev_label(client: httpx.Client, task_id: str, problem_statement: str) -> dict[str, object]:
    body = {
        "model": JEV_MODEL,
        "state": problem_statement,
        "questions": {
            "capability": {
                "type": "choice",
                "instructions": "Which capability does solving this software task mainly require?",
                "criteria": JEV_CRITERIA,
            }
        },
    }
    started = time.perf_counter()
    response: httpx.Response | None = None
    payload: dict[str, object] = {}
    error: str | None = None
    try:
        response = client.post(f"{BASE_URL}/v1/systemone", headers=_headers(), json=body)
        payload = _json_response(response)
        if response.is_error:
            error = f"http_{response.status_code}"
    except httpx.HTTPError as exception:
        error = type(exception).__name__
    latency_s = time.perf_counter() - started
    metadata = (
        _response_metadata(response, payload, latency_s)
        if response is not None
        else {"call_id": None, "response_model": None, "reported_version": None, "cost_usd": None,
              "latency_s": latency_s, "usage": {}}
    )
    label, confidence, probabilities = parse_jev_response(payload)
    response_mapping = _as_mapping(payload)
    answers = None if response_mapping is None else _as_mapping(response_mapping.get("answers"))
    raw_answer = None if answers is None else _as_mapping(answers.get("capability"))
    if raw_answer is None and response_mapping is not None:
        raw_answer = _as_mapping(response_mapping.get("capability")) or _as_mapping(response_mapping.get("answer"))
    raw_label_text = json.dumps(
        {
            key: value
            for key, value in (raw_answer or {}).items()
            if key in {"choice", "answer", "value", "confidence", "probabilities"}
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if error is not None:
        label, confidence, probabilities = None, None, {}
    _append_auxiliary_call("jev_classifier", task_id, JEV_MODEL, metadata)
    return {
        **metadata,
        "label": label,
        "confidence": confidence,
        "probabilities": probabilities,
        "raw_label_text": _sanitize_label_text(raw_label_text),
        "error": error,
    }


def _load_tasks() -> tuple[dict[str, str], list[str]]:
    frozen = json.loads((ROOT / "tasks.json").read_text())
    task_ids = [task["instance_id"] for task in frozen["tasks"]]
    statements = {
        row["instance_id"]: row["problem_statement"]
        for row in (json.loads(line) for line in (ROOT / "agent_dataset" / "train.jsonl").read_text().splitlines())
        if row.get("instance_id") in task_ids and isinstance(row.get("problem_statement"), str)
    }
    missing = set(task_ids) - statements.keys()
    if missing:
        raise ValueError(f"missing sanitized training statements for {sorted(missing)}")
    return statements, task_ids


def label_tasks() -> list[dict[str, object]]:
    if (ROOT / "labels.json").exists():
        raise FileExistsError("labels.json already exists; remove it only before intentionally relabeling")
    os.environ["LITELLM_PROXY_API_KEY"] = _read_api_key()
    statements, task_ids = _load_tasks()
    with httpx.Client(transport=httpx.HTTPTransport(retries=0), timeout=REQUEST_TIMEOUT_S) as client:
        rows = [
            {
                "task_id": task_id,
                "offline": _offline_label(client, task_id, statements[task_id]),
                "jev": _jev_label(client, task_id, statements[task_id]),
            }
            for task_id in task_ids
        ]
    (ROOT / "labels.json").write_text(json.dumps(rows, indent=2) + "\n")
    return rows


def main() -> None:
    argparse.ArgumentParser(description="Label the frozen POC task statements").parse_args()
    rows = label_tasks()
    summary = [
        {
            "task_id": row["task_id"],
            "offline": row["offline"]["label"],
            "jev": row["jev"]["label"],
            "cost_usd": (row["offline"]["cost_usd"] or 0.0) + (row["jev"]["cost_usd"] or 0.0),
        }
        for row in rows
    ]
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

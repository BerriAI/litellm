from __future__ import annotations

import asyncio
import json
import os
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal, cast

import httpx
import pytest
import yaml
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    object_value,
    string_value,
)
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy_process
from tests.integration._support.upstream import (
    JsonResponse,
    SseResponse,
    delete_scenario,
    register_scenario,
)
from tests.integration.pricing.test_service_tier_pricing import (
    BUNDLED_COST_MAP,
    CUSTOM_STANDARD_INPUT_RATE,
    CUSTOM_STANDARD_OUTPUT_RATE,
    _bundled_rate,
)

MATRIX_BACKEND_MODEL: Final = "gpt-6-astra"
MATRIX_CATALOG_ENTRY: Final = object_value(
    JSON_OBJECT.validate_json(BUNDLED_COST_MAP.read_bytes())[MATRIX_BACKEND_MODEL]
)
MATRIX_AZURE_CATALOG_MODEL: Final = "azure/eu/gpt-5-2025-08-07"
MATRIX_AZURE_ALIAS: Final = "azure/service-tier-matrix-alias"
MATRIX_PROMPT_TOKENS: Final = 20
MATRIX_COMPLETION_TOKENS: Final = 20
MATRIX_CHAT_PATH: Final = "/v1/chat/completions"
MATRIX_RESPONSES_PATH: Final = "/v1/responses"
MATRIX_MESSAGES_PATH: Final = "/v1/messages"
SERVICE_TIER_CASES: Final = (
    ("priority", "priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS, 0),
    ("flex", "flex", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS, 0),
    ("ultrafast", "ultrafast", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS, 0),
    ("fast", "priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS, 0),
)
LONG_CONTEXT_CASES: Final = (
    ("ultrafast", "ultrafast"),
    ("priority", "priority"),
)
INVALID_TIER_VALUES: Final = (
    ("integer", 5),
    ("sequence", ["priority"]),
    ("empty", ""),
    ("oversized", "x" * 5_000),
)


@dataclass(frozen=True, slots=True)
class MatrixOutcome:
    status: int
    request_id: str | None
    cost_header: float | None
    body: dict[str, JsonValue]
    text: str


def _service_tier_rates(
    service_tier: str | None,
    *,
    custom_standard: bool = True,
    long_context: bool = False,
) -> tuple[float, float]:
    if service_tier is None or service_tier == "balanced":
        if custom_standard:
            return CUSTOM_STANDARD_INPUT_RATE, CUSTOM_STANDARD_OUTPUT_RATE
        return (
            _bundled_rate(MATRIX_BACKEND_MODEL, "input_cost_per_token"),
            _bundled_rate(MATRIX_BACKEND_MODEL, "output_cost_per_token"),
        )

    tier: Final = "priority" if service_tier == "fast" else service_tier
    input_field: Final = (
        f"input_cost_per_token_above_272k_tokens_{tier}"
        if long_context and f"input_cost_per_token_above_272k_tokens_{tier}" in MATRIX_CATALOG_ENTRY
        else f"input_cost_per_token_{tier}"
    )
    output_field: Final = (
        f"output_cost_per_token_above_272k_tokens_{tier}"
        if long_context and f"output_cost_per_token_above_272k_tokens_{tier}" in MATRIX_CATALOG_ENTRY
        else f"output_cost_per_token_{tier}"
    )
    return (
        _bundled_rate(MATRIX_BACKEND_MODEL, input_field)
        if input_field in MATRIX_CATALOG_ENTRY
        else CUSTOM_STANDARD_INPUT_RATE
        if custom_standard
        else _bundled_rate(MATRIX_BACKEND_MODEL, "input_cost_per_token"),
        _bundled_rate(MATRIX_BACKEND_MODEL, output_field)
        if output_field in MATRIX_CATALOG_ENTRY
        else CUSTOM_STANDARD_OUTPUT_RATE
        if custom_standard
        else _bundled_rate(MATRIX_BACKEND_MODEL, "output_cost_per_token"),
    )


def _chat_payload(
    service_tier: JsonValue | None,
    *,
    stream: bool = False,
    no_cache: bool = True,
) -> dict[str, JsonValue]:
    return {
        "messages": [{"role": "user", "content": "service tier matrix request"}],
        "stream": stream,
        "max_tokens": MATRIX_COMPLETION_TOKENS,
        **({"service_tier": service_tier} if service_tier is not None else {}),
        **({"stream_options": {"include_usage": True}} if stream else {}),
        **({"cache": {"no-cache": True}} if no_cache else {}),
    }


def _chat_json_response(
    service_tier: JsonValue | None,
    *,
    prompt_tokens: int = MATRIX_PROMPT_TOKENS,
    completion_tokens: int = MATRIX_COMPLETION_TOKENS,
    cached_tokens: int = 0,
    response_service_tier: str | None = None,
) -> JsonResponse:
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        **({"prompt_tokens_details": {"cached_tokens": cached_tokens}} if cached_tokens else {}),
    }
    body: Final[dict[str, JsonValue]] = {
        "id": "chatcmpl-$UNIQUE_ID",
        "object": "chat.completion",
        "created": 1_700_000_000,
        "model": MATRIX_BACKEND_MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "matrix response"},
                "finish_reason": "stop",
            }
        ],
        "usage": usage,
        **(
            {"service_tier": response_service_tier}
            if response_service_tier is not None
            else {"service_tier": service_tier}
            if isinstance(service_tier, str) and service_tier
            else {}
        ),
    }
    return JsonResponse(content_type="application/json", body=body)


def _responses_json_response(
    service_tier: str | None,
    *,
    prompt_tokens: int = MATRIX_PROMPT_TOKENS,
    completion_tokens: int = MATRIX_COMPLETION_TOKENS,
) -> JsonResponse:
    body: dict[str, JsonValue] = {
        "id": "resp_$UNIQUE_ID",
        "object": "response",
        "created_at": 1_700_000_000,
        "status": "completed",
        "model": MATRIX_BACKEND_MODEL,
        "output": [
            {
                "id": "msg_$UNIQUE_ID",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "matrix response", "annotations": []}],
                "status": "completed",
            }
        ],
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    return JsonResponse(
        content_type="application/json",
        body={
            **body,
            **({"service_tier": service_tier} if service_tier is not None else {}),
        },
    )


def _chat_sse_response(
    *,
    prompt_tokens: int = MATRIX_PROMPT_TOKENS,
    completion_tokens: int = MATRIX_COMPLETION_TOKENS,
    frame_delay_ms: int = 0,
) -> SseResponse:
    usage_frame: Final = json.dumps(
        {
            "id": "chatcmpl-$UNIQUE_ID",
            "object": "chat.completion.chunk",
            "created": 1_700_000_000,
            "model": MATRIX_BACKEND_MODEL,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        },
        separators=(",", ":"),
    )
    frames: Final = (
        f'data: {{"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1700000000,"model":"{MATRIX_BACKEND_MODEL}","choices":[{{"index":0,"delta":{{"role":"assistant"}},"finish_reason":null}}]}}',
        f'data: {{"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1700000000,"model":"{MATRIX_BACKEND_MODEL}","choices":[{{"index":0,"delta":{{"content":"matrix response"}},"finish_reason":null}}]}}',
        f"data: {usage_frame}",
        "data: [DONE]",
    )
    return SseResponse(content_type="text/event-stream", frames=frames, frame_delay_ms=frame_delay_ms)


def _responses_sse_response(*, service_tier: str | None = "priority") -> SseResponse:
    response_id: Final = f"resp_{uuid.uuid4().hex}"
    message_id: Final = f"msg_{response_id}"
    completed_response: Final = {
        "id": response_id,
        "object": "response",
        "created_at": 1_700_000_000,
        "status": "completed",
        "model": MATRIX_BACKEND_MODEL,
        **({"service_tier": service_tier} if service_tier is not None else {}),
        "output": [
            {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "matrix response",
                        "annotations": [],
                    }
                ],
            }
        ],
        "usage": {
            "input_tokens": MATRIX_PROMPT_TOKENS,
            "output_tokens": MATRIX_COMPLETION_TOKENS,
            "total_tokens": MATRIX_PROMPT_TOKENS + MATRIX_COMPLETION_TOKENS,
        },
    }
    events: Final = (
        {
            "type": "response.created",
            "sequence_number": 0,
            "response": {**completed_response, "status": "in_progress", "output": []},
        },
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": message_id,
            "output_index": 0,
            "content_index": 0,
            "delta": "matrix response",
        },
        {
            "type": "response.completed",
            "sequence_number": 2,
            "response": completed_response,
        },
    )
    frames: Final = tuple(
        f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}" for event in events
    )
    return SseResponse(content_type="text/event-stream", frames=frames)


def _outcome(
    *,
    status: int,
    headers: Mapping[str, str],
    body: dict[str, JsonValue],
    text: str,
) -> MatrixOutcome:
    raw_cost: Final = headers.get("x-litellm-response-cost")
    cost: Final = float(raw_cost) if raw_cost is not None else None
    raw_id: Final = body.get("id") or headers.get("x-litellm-call-id")
    response_id: Final = raw_id if isinstance(raw_id, str) else None
    return MatrixOutcome(status, response_id, cost, body, text)


def _body_from_text(text: str) -> dict[str, JsonValue]:
    if not text.strip():
        return {}
    try:
        body: Final = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return object_value(body)


def _stream_outcome(
    *,
    status: int,
    headers: Mapping[str, str],
    text: str,
    surface: Literal["chat", "responses", "messages"],
) -> MatrixOutcome:
    events: Final = tuple(
        object_value(json.loads(line.removeprefix("data: ").strip()))
        for line in text.splitlines()
        if line.startswith("data: ") and line.removeprefix("data: ").strip() != "[DONE]"
    )
    response_id: Final = next(
        (candidate for event in events if (candidate := _stream_event_id(event, surface)) is not None),
        None,
    )
    completed_response: Final = next(
        (
            object_value(event["response"])
            for event in events
            if surface == "responses"
            and event.get("type") == "response.completed"
            and isinstance(event.get("response"), dict)
        ),
        {},
    )
    started_message: Final = next(
        (
            object_value(event["message"])
            for event in events
            if surface == "messages" and isinstance(event.get("message"), dict)
        ),
        {},
    )
    body: Final = {
        **(completed_response if surface == "responses" else started_message if surface == "messages" else {}),
        **({"id": response_id} if response_id is not None else {}),
    }
    return _outcome(status=status, headers=headers, body=body, text=text)


def _stream_event_id(
    event: Mapping[str, JsonValue],
    surface: Literal["chat", "responses", "messages"],
) -> str | None:
    candidate: JsonValue | None = event.get("id")
    if surface == "responses" and isinstance(event.get("response"), dict):
        candidate = object_value(event["response"]).get("id")
    if surface == "messages" and isinstance(event.get("message"), dict):
        candidate = object_value(event["message"]).get("id")
    return candidate if isinstance(candidate, str) else None


def _invoke_http(
    gateway: Gateway,
    *,
    path: str,
    payload: Mapping[str, JsonValue],
    key: str,
    stream: bool = False,
    surface: Literal["chat", "responses", "messages"] = "chat",
) -> MatrixOutcome:
    headers: Final = {"Authorization": f"Bearer {key}"}
    if not stream:
        response: Final = gateway.request("POST", path, payload, key=key)
        return _outcome(
            status=response.status_code,
            headers=response.headers,
            body=_body_from_text(response.text),
            text=response.text,
        )
    with gateway.client.stream("POST", path, json=payload, headers=headers) as response:
        text: Final = response.read().decode("utf-8", errors="replace")
        return _stream_outcome(
            status=response.status_code,
            headers=response.headers,
            text=text,
            surface=surface,
        )


def _invoke_sync_sdk(
    base_url: str,
    *,
    key: str,
    model: str,
    service_tier: str | None,
    stream: bool,
) -> MatrixOutcome:
    with OpenAI(api_key=key, base_url=f"{base_url.rstrip('/')}/v1", max_retries=0) as client:
        if stream:
            with client.chat.completions.with_streaming_response.create(
                model=model,
                messages=[{"role": "user", "content": "service tier matrix request"}],
                service_tier=service_tier,
                max_tokens=MATRIX_COMPLETION_TOKENS,
                stream=True,
                stream_options={"include_usage": True},
                extra_body={"cache": {"no-cache": True}},
            ) as response:
                text: Final = "\n".join(response.iter_lines())
                return _stream_outcome(
                    status=response.status_code,
                    headers=response.headers,
                    text=text,
                    surface="chat",
                )
        response: Final = client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "service tier matrix request"}],
            service_tier=service_tier,
            max_tokens=MATRIX_COMPLETION_TOKENS,
            extra_body={"cache": {"no-cache": True}},
        )
        parsed: Final = response.parse()
        return _outcome(
            status=response.http_response.status_code,
            headers=response.headers,
            body=cast(dict[str, JsonValue], parsed.model_dump(mode="json")),
            text=response.http_response.text,
        )


async def _invoke_async_sdk(
    base_url: str,
    *,
    key: str,
    model: str,
    service_tier: str,
) -> MatrixOutcome:
    async with AsyncOpenAI(api_key=key, base_url=f"{base_url.rstrip('/')}/v1", max_retries=0) as client:
        response: Final = await client.chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "user", "content": "service tier matrix request"}],
            service_tier=service_tier,
            max_tokens=MATRIX_COMPLETION_TOKENS,
            extra_body={"cache": {"no-cache": True}},
        )
        parsed: Final = response.parse()
        return _outcome(
            status=response.http_response.status_code,
            headers=response.headers,
            body=cast(dict[str, JsonValue], parsed.model_dump(mode="json")),
            text=response.http_response.text,
        )


def _scenario_model(
    scenario: Scenario,
    *,
    scenario_id: str,
    api_base: str,
    model: str = f"openai/{MATRIX_BACKEND_MODEL}",
    model_info: Mapping[str, JsonValue] | None = None,
    custom_rates: bool = True,
    **parameters: JsonValue,
) -> str:
    return scenario.model(
        model_info=model_info,
        model=model,
        api_key=scenario_id,
        api_base=api_base,
        **(
            {
                "input_cost_per_token": CUSTOM_STANDARD_INPUT_RATE,
                "output_cost_per_token": CUSTOM_STANDARD_OUTPUT_RATE,
            }
            if custom_rates
            else {}
        ),
        **parameters,
    )


def _register_upstream(
    gateway: Gateway,
    scenario: Scenario,
    *,
    response: JsonResponse | SseResponse,
    identifier: str | None = None,
) -> tuple[str, str]:
    _observations(gateway)
    scenario_id: Final = identifier or f"tier-matrix-{uuid.uuid4().hex}"
    handle: Final = register_scenario(scenario_id, response)
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario_id, handle.api_base()


def _observations_at(upstream_url: str) -> list[dict[str, JsonValue]]:
    with httpx.Client(base_url=upstream_url, trust_env=False) as upstream:
        body: Final = object_value(JSON_OBJECT.validate_json(upstream.get("/__observations").content))
    return [object_value(value) for value in body["requests"]]


def _observations(gateway: Gateway) -> list[dict[str, JsonValue]]:
    return _observations_at(gateway.upstream_url)


def _assert_forwarded_body(
    body: Mapping[str, JsonValue],
    *,
    expected_tier: JsonValue | None,
    tier_present: bool,
) -> None:
    assert not {key for key in body if "cost_per_token" in key or key.startswith("cache_read_input_token_cost")}, body
    if tier_present:
        assert body.get("service_tier") == expected_tier, body
    else:
        assert "service_tier" not in body, body


def _assert_upstream_request(
    gateway: Gateway,
    *,
    expected_count: int = 1,
    expected_tier: JsonValue | None = None,
    tier_present: bool = False,
) -> dict[str, JsonValue]:
    requests: Final = _observations(gateway)
    assert len(requests) == expected_count, requests
    latest: Final = requests[-1]
    _assert_forwarded_body(
        object_value(latest["body"]),
        expected_tier=expected_tier,
        tier_present=tier_present,
    )
    return latest


def _assert_upstream(
    gateway: Gateway,
    *,
    expected_count: int = 1,
    expected_tier: JsonValue | None = None,
    tier_present: bool = False,
) -> dict[str, JsonValue]:
    request: Final = _assert_upstream_request(
        gateway,
        expected_count=expected_count,
        expected_tier=expected_tier,
        tier_present=tier_present,
    )
    return object_value(request["body"])


def _assert_azure_upstream(gateway: Gateway, *, scenario_id: str) -> None:
    observations: Final = _observations(gateway)
    requests: Final = tuple(
        request
        for request in observations
        if string_value(request["path"]).startswith(f"/{scenario_id}/openai/deployments/")
    )
    assert len(requests) == 1, observations
    path: Final = string_value(requests[0]["path"]).split("?", maxsplit=1)[0]
    assert path.endswith("/chat/completions"), requests
    _assert_forwarded_body(
        object_value(requests[0]["body"]),
        expected_tier="priority",
        tier_present=True,
    )


def _spend_rows(request_id: str) -> list[dict[str, object]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (request_id,),
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )


def _cache_hit_spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    cache_hit_prefix: Final = f"{request_id}_cache_hit"
    return eventually(
        lambda: read_rows(
            'SELECT request_id, spend, cache_hit FROM "LiteLLM_SpendLogs" WHERE LEFT(request_id, %s) = %s',
            (len(cache_hit_prefix), cache_hit_prefix),
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )


def _spend_rows_for_key(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, status, spend, prompt_tokens, completion_tokens, cache_hit FROM "LiteLLM_SpendLogs" '
        'WHERE api_key=%s ORDER BY "startTime"',
        (sha256(key.encode()).hexdigest(),),
    )


def _register_db_model(
    gateway: Gateway,
    scenario: Scenario,
    *,
    model: str,
    scenario_id: str,
    api_base: str,
    custom_rates: bool = True,
    model_info: Mapping[str, JsonValue] | None = None,
    **parameters: JsonValue,
) -> tuple[str, str]:
    model_name: Final = f"tier-priced-{uuid.uuid4().hex}"
    created: Final = gateway.post(
        "/model/new",
        {
            "model_name": model_name,
            "litellm_params": {
                "model": model,
                "api_key": scenario_id,
                "api_base": api_base,
                **(
                    {
                        "input_cost_per_token": CUSTOM_STANDARD_INPUT_RATE,
                        "output_cost_per_token": CUSTOM_STANDARD_OUTPUT_RATE,
                    }
                    if custom_rates
                    else {}
                ),
                **parameters,
            },
            "model_info": dict(model_info) if model_info is not None else {},
        },
    )
    model_id: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, model_id)
    return model_name, model_id


def _write_proxy_config(
    tmp_path: Path,
    *,
    models: tuple[Mapping[str, JsonValue], ...],
    filename: str,
) -> Path:
    source: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
    config: Final = JSON_OBJECT.validate_python(yaml.safe_load(source.read_text()))
    current_models: Final = config.get("model_list", [])
    assert isinstance(current_models, list), current_models
    router_settings: Final = object_value(config.get("router_settings", {}))
    path: Final = tmp_path / filename
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "model_list": [*current_models, *(dict(model) for model in models)],
                "router_settings": {**router_settings, "num_retries": 0},
            }
        )
    )
    return path


def _assert_cost_header(
    outcome: MatrixOutcome,
    *,
    expected_cost: float,
    allow_missing_header: bool = False,
) -> None:
    assert outcome.status == 200, outcome.text
    assert outcome.request_id is not None, outcome
    if outcome.cost_header is None:
        assert allow_missing_header, outcome
    else:
        assert outcome.cost_header == pytest.approx(expected_cost, rel=1e-6), outcome


def _assert_billing(
    outcome: MatrixOutcome,
    *,
    expected_cost: float,
    prompt_tokens: int,
    completion_tokens: int,
    allow_missing_header: bool = False,
) -> dict[str, object]:
    _assert_cost_header(outcome, expected_cost=expected_cost, allow_missing_header=allow_missing_header)
    request_id: Final = outcome.request_id
    assert request_id is not None, outcome
    rows: Final = _spend_rows(request_id)
    row: Final = rows[0]
    assert float(str(row["spend"])) == pytest.approx(expected_cost, rel=1e-6), row
    assert int(str(row["prompt_tokens"])) == prompt_tokens, row
    assert int(str(row["completion_tokens"])) == completion_tokens, row
    return row


def _expected_cost(
    service_tier: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    custom_standard: bool = True,
    long_context: bool = False,
    cached_tokens: int = 0,
) -> float:
    input_rate, output_rate = _service_tier_rates(
        service_tier,
        custom_standard=custom_standard,
        long_context=long_context,
    )
    if cached_tokens:
        cache_field: Final = f"cache_read_input_token_cost_{service_tier}"
        if service_tier is not None and cache_field in MATRIX_CATALOG_ENTRY:
            cached_rate: Final = _bundled_rate(MATRIX_BACKEND_MODEL, cache_field)
            return (
                (prompt_tokens - cached_tokens) * input_rate
                + cached_tokens * cached_rate
                + completion_tokens * output_rate
            )
    return prompt_tokens * input_rate + completion_tokens * output_rate


def _assert_success(
    gateway: Gateway,
    outcome: MatrixOutcome,
    *,
    service_tier: JsonValue | None,
    prompt_tokens: int = MATRIX_PROMPT_TOKENS,
    completion_tokens: int = MATRIX_COMPLETION_TOKENS,
    expected_cost: float,
    tier_present: bool | None = None,
) -> None:
    _assert_billing(
        outcome,
        expected_cost=expected_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    _assert_upstream(
        gateway,
        expected_tier=service_tier,
        tier_present=service_tier is not None if tier_present is None else tier_present,
    )


@pytest.mark.parametrize(
    ("service_tier", "rate_tier", "prompt_tokens", "completion_tokens", "cached_tokens"),
    SERVICE_TIER_CASES,
    ids=("priority", "flex", "ultrafast", "fast-alias-to-priority"),
)
def test_raw_chat_tier_rates_follow_the_catalog_for_custom_deployments(
    gateway: Gateway,
    service_tier: str,
    rate_tier: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response(
                service_tier, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            ),
            identifier=f"tier-matrix-chat-{service_tier}-billed-as-{rate_tier}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        payload: Final = _chat_payload(service_tier)
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **payload},
            key=key,
        )
        _assert_success(
            gateway,
            outcome,
            service_tier=service_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            expected_cost=_expected_cost(rate_tier, prompt_tokens, completion_tokens),
        )


@pytest.mark.parametrize(
    "service_tier",
    ("balanced", None),
    ids=("balanced", "no-tier"),
)
def test_balanced_and_no_tier_keep_custom_standard_rates(
    gateway: Gateway,
    service_tier: str | None,
) -> None:
    scenario_name: Final = "balanced" if service_tier is not None else "no-tier"
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response(service_tier),
            identifier=f"tier-matrix-chat-{scenario_name}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload(service_tier)},
            key=key,
        )
        _assert_success(
            gateway,
            outcome,
            service_tier=service_tier,
            expected_cost=_expected_cost(service_tier, MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
        )


def test_openai_sdk_sync_priority_chat_uses_inherited_catalog_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-sdk-sync-chat-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_sync_sdk(
            os.environ["INTEGRATION_PROXY_URL"],
            key=key,
            model=model,
            service_tier="priority",
            stream=False,
        )
        _assert_billing(
            outcome,
            expected_cost=_expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )
        _assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_openai_sdk_async_priority_chat_uses_inherited_catalog_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-sdk-async-chat-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = asyncio.run(
            _invoke_async_sdk(
                os.environ["INTEGRATION_PROXY_URL"],
                key=key,
                model=model,
                service_tier="priority",
            )
        )
        assert outcome.status == 200, outcome.text
        assert outcome.request_id is not None, outcome
        _assert_billing(
            outcome,
            expected_cost=_expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )
        _assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_openai_sdk_streaming_priority_chat_consumes_usage_and_logs_spend(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_sse_response(),
            identifier=f"tier-matrix-sdk-stream-chat-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_sync_sdk(
            os.environ["INTEGRATION_PROXY_URL"],
            key=key,
            model=model,
            service_tier="priority",
            stream=True,
        )
        _assert_billing(
            outcome,
            expected_cost=_expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
            allow_missing_header=True,
        )
        _assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_raw_responses_priority_uses_inherited_catalog_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_responses_json_response("priority"),
            identifier=f"tier-matrix-raw-responses-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_RESPONSES_PATH,
            payload={
                "model": model,
                "input": "service tier matrix request",
                "service_tier": "priority",
                "cache": {"no-cache": True},
            },
            key=key,
            surface="responses",
        )
        _assert_billing(
            outcome,
            expected_cost=_expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )
        _assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_openai_sdk_streaming_responses_priority_consumes_full_stream(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_responses_sse_response(),
            identifier=f"tier-matrix-sdk-stream-responses-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        with OpenAI(
            api_key=key,
            base_url=f"{os.environ['INTEGRATION_PROXY_URL'].rstrip('/')}/v1",
        ) as client:
            with client.responses.with_streaming_response.create(
                model=model,
                input="service tier matrix request",
                service_tier="priority",
                stream=True,
                extra_body={"cache": {"no-cache": True}},
            ) as response:
                stream_text: Final = "\n".join(response.iter_lines())
                outcome: Final = _stream_outcome(
                    status=response.status_code,
                    headers=response.headers,
                    text=stream_text,
                    surface="responses",
                )
        expected_cost: Final = _expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS)
        _assert_cost_header(
            outcome,
            expected_cost=expected_cost,
            allow_missing_header=True,
        )
        rows: Final = eventually(
            lambda: _spend_rows_for_key(key),
            lambda current: len(current) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert row["status"] == "success", row
        assert float(str(row["spend"])) == pytest.approx(expected_cost, rel=1e-6), row
        assert int(str(row["prompt_tokens"])) == MATRIX_PROMPT_TOKENS, row
        assert int(str(row["completion_tokens"])) == MATRIX_COMPLETION_TOKENS, row
        _assert_upstream(gateway, expected_tier="priority", tier_present=True)


@pytest.mark.parametrize("stream", (False, True), ids=("non-streaming", "streaming"))
def test_messages_endpoint_routes_to_responses_upstream_and_bills_custom_standard(
    gateway: Gateway,
    stream: bool,
) -> None:
    mode: Final = "streaming" if stream else "non-streaming"
    response: Final = _responses_sse_response(service_tier=None) if stream else _responses_json_response(None)
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=response,
            identifier=f"tier-matrix-messages-{mode}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        payload: dict[str, JsonValue] = {
            "model": model,
            "max_tokens": MATRIX_COMPLETION_TOKENS,
            "messages": [{"role": "user", "content": f"service tier matrix request {scenario_id}"}],
            "cache": {"no-cache": True},
        }
        if stream:
            payload["stream"] = True
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_MESSAGES_PATH,
            payload=payload,
            key=key,
            stream=stream,
            surface="messages",
        )
        _assert_billing(
            outcome,
            expected_cost=_expected_cost(None, MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
            allow_missing_header=stream,
        )
        observed_request: Final = _assert_upstream_request(gateway)
        observed_path: Final = string_value(observed_request["path"])
        assert observed_path == f"/{scenario_id}/responses", observed_request
        observed: Final = object_value(observed_request["body"])
        assert "service_tier" not in observed


def test_cached_prompt_tokens_use_priority_cache_rate(gateway: Gateway) -> None:
    cached_tokens: Final = 5
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority", cached_tokens=cached_tokens),
            identifier=f"tier-matrix-chat-priority-cache-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload("priority")},
            key=key,
        )
        _assert_billing(
            outcome,
            expected_cost=_expected_cost(
                "priority",
                MATRIX_PROMPT_TOKENS,
                MATRIX_COMPLETION_TOKENS,
                cached_tokens=cached_tokens,
            ),
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )
        _assert_upstream(gateway, expected_tier="priority", tier_present=True)


@pytest.mark.parametrize(
    ("service_tier", "rate_tier"),
    LONG_CONTEXT_CASES,
    ids=("ultrafast", "priority"),
)
def test_long_context_tier_rate_is_inherited_for_custom_deployments(
    gateway: Gateway,
    service_tier: str,
    rate_tier: str,
) -> None:
    prompt_tokens: Final = 300_000
    completion_tokens: Final = 100
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response(
                service_tier,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            identifier=f"tier-matrix-long-context-{service_tier}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload(service_tier)},
            key=key,
        )
        _assert_success(
            gateway,
            outcome,
            service_tier=service_tier,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            expected_cost=_expected_cost(
                rate_tier,
                prompt_tokens,
                completion_tokens,
                long_context=True,
            ),
        )


def test_long_context_no_tier_keeps_custom_standard_rates(gateway: Gateway) -> None:
    prompt_tokens: Final = 300_000
    completion_tokens: Final = 100
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response(
                None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            identifier=f"tier-matrix-long-context-no-tier-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload(None)},
            key=key,
        )
        _assert_success(
            gateway,
            outcome,
            service_tier=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            expected_cost=_expected_cost(None, prompt_tokens, completion_tokens),
        )


def test_explicit_priority_input_and_output_prices_win(gateway: Gateway) -> None:
    explicit_input_rate: Final = 0.00091
    explicit_output_rate: Final = 0.00173
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-explicit-priority-rates-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            input_cost_per_token_priority=explicit_input_rate,
            output_cost_per_token_priority=explicit_output_rate,
        )
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload("priority")},
            key=key,
        )
        _assert_success(
            gateway,
            outcome,
            service_tier="priority",
            expected_cost=MATRIX_PROMPT_TOKENS * explicit_input_rate + MATRIX_COMPLETION_TOKENS * explicit_output_rate,
        )


def test_explicit_priority_input_wins_while_missing_output_is_inherited(gateway: Gateway) -> None:
    explicit_input_rate: Final = 0.00091
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-explicit-priority-input-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            input_cost_per_token_priority=explicit_input_rate,
        )
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload("priority")},
            key=key,
        )
        expected_cost: Final = MATRIX_PROMPT_TOKENS * explicit_input_rate + MATRIX_COMPLETION_TOKENS * _bundled_rate(
            MATRIX_BACKEND_MODEL, "output_cost_per_token_priority"
        )
        _assert_success(
            gateway,
            outcome,
            service_tier="priority",
            expected_cost=expected_cost,
        )


def test_standard_input_only_deployment_inherits_catalog_priority_rates(gateway: Gateway) -> None:
    explicit_standard_input_rate: Final = 0.00071
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-standard-input-only-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            custom_rates=False,
            input_cost_per_token=explicit_standard_input_rate,
        )
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload("priority")},
            key=key,
        )
        expected_cost: Final = _expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS)
        _assert_success(
            gateway,
            outcome,
            service_tier="priority",
            expected_cost=expected_cost,
        )


@pytest.mark.parametrize(
    "service_tier",
    ("priority", None),
    ids=("priority", "no-tier"),
)
def test_catalog_priced_deployment_keeps_its_catalog_rates(
    gateway: Gateway,
    service_tier: str | None,
) -> None:
    scenario_name: Final = "priority" if service_tier is not None else "no-tier"
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response(service_tier),
            identifier=f"tier-matrix-catalog-priced-{scenario_name}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            custom_rates=False,
        )
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload(service_tier)},
            key=key,
        )
        _assert_success(
            gateway,
            outcome,
            service_tier=service_tier,
            expected_cost=_expected_cost(
                service_tier,
                MATRIX_PROMPT_TOKENS,
                MATRIX_COMPLETION_TOKENS,
                custom_standard=False,
            ),
        )


def test_unknown_custom_model_uses_custom_standard_rates_without_proxy_error_log(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    unknown_model: Final = f"openai/unknown-tier-priced-model-{uuid.uuid4().hex}"
    with owned_proxy_process(gateway, tmp_path, {}) as owned:
        candidate: Final = owned.gateway
        prior_log: Final = owned.log.read_text() if owned.log.exists() else ""
        with candidate.scenario() as scenario:
            scenario_id, api_base = _register_upstream(
                gateway,
                scenario,
                response=_chat_json_response(None),
                identifier=f"tier-matrix-unknown-model-standard-pricing-{uuid.uuid4().hex}",
            )
            key: Final = scenario.key()
            model: Final = _scenario_model(
                scenario,
                scenario_id=scenario_id,
                api_base=api_base,
                model=unknown_model,
            )
            outcome: Final = _invoke_http(
                candidate,
                path=MATRIX_CHAT_PATH,
                payload={"model": model, **_chat_payload(None)},
                key=key,
            )
            _assert_success(
                gateway,
                outcome,
                service_tier=None,
                expected_cost=_expected_cost(None, MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
            )
        current_log: Final = owned.log.read_text() if owned.log.exists() else ""
        added_log: Final = current_log.removeprefix(prior_log)
        assert "ERROR" not in added_log, added_log


def test_azure_alias_inherits_priority_rates_from_model_info_base_model(gateway: Gateway) -> None:
    input_rate: Final = _bundled_rate(MATRIX_AZURE_CATALOG_MODEL, "input_cost_per_token_priority")
    output_rate: Final = _bundled_rate(MATRIX_AZURE_CATALOG_MODEL, "output_cost_per_token_priority")
    identifier: Final = f"tier-matrix-azure-alias-{uuid.uuid4().hex}"
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=identifier,
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            model=MATRIX_AZURE_ALIAS,
            model_info={"base_model": MATRIX_AZURE_CATALOG_MODEL},
            api_version="2024-10-21",
        )
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload("priority")},
            key=key,
        )
        _assert_azure_upstream(gateway, scenario_id=identifier)
        _assert_billing(
            outcome,
            expected_cost=MATRIX_PROMPT_TOKENS * input_rate + MATRIX_COMPLETION_TOKENS * output_rate,
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )


def test_ptu_deployment_keeps_zero_per_request_cost_with_attribution_enabled(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-ptu-zero-cost-{uuid.uuid4().hex}",
        )
        team_id: Final = scenario.team()
        model: Final = f"tier-priced-ptu-{uuid.uuid4().hex}"
        config: Final = _write_proxy_config(
            tmp_path,
            models=(
                {
                    "model_name": model,
                    "litellm_params": {
                        "model": f"openai/{MATRIX_BACKEND_MODEL}",
                        "api_key": scenario_id,
                        "api_base": api_base,
                    },
                    "model_info": {
                        "id": uuid.uuid4().hex,
                        "team_id": team_id,
                        "ptu_count": 4,
                        "cost_per_ptu_per_hour": 0.0,
                        "ptu_effective_from": datetime.now(UTC).isoformat(),
                    },
                },
            ),
            filename="service-tier-ptu.yaml",
        )
        with owned_proxy_process(
            gateway,
            tmp_path,
            {"LITELLM_ENABLE_PTU_COST_ATTRIBUTION": "true"},
            config=config,
        ) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as candidate_scenario:
                key: Final = candidate_scenario.key(team_id=team_id)
                outcome: Final = _invoke_http(
                    candidate,
                    path=MATRIX_CHAT_PATH,
                    payload={"model": model, **_chat_payload("priority")},
                    key=key,
                )
                _assert_billing(
                    outcome,
                    expected_cost=0.0,
                    prompt_tokens=MATRIX_PROMPT_TOKENS,
                    completion_tokens=MATRIX_COMPLETION_TOKENS,
                    allow_missing_header=True,
                )
                _assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_yaml_configured_deployment_inherits_priority_rates(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-yaml-config-priority-{uuid.uuid4().hex}",
        )
        model_name: Final = f"custom-priced-tier-yaml-{uuid.uuid4().hex}"
        model: Final = {
            "model_name": model_name,
            "litellm_params": {
                "model": f"openai/{MATRIX_BACKEND_MODEL}",
                "api_key": scenario_id,
                "api_base": api_base,
                "input_cost_per_token": CUSTOM_STANDARD_INPUT_RATE,
                "output_cost_per_token": CUSTOM_STANDARD_OUTPUT_RATE,
            },
            "model_info": {},
        }
        config: Final = _write_proxy_config(
            tmp_path,
            models=(model,),
            filename="service-tier-yaml.yaml",
        )
        with owned_proxy_process(gateway, tmp_path, {}, config=config, workers=2) as owned:
            candidate: Final = owned.gateway
            with candidate.scenario() as candidate_scenario:
                key: Final = candidate_scenario.key()
                outcome: Final = _invoke_http(
                    candidate,
                    path=MATRIX_CHAT_PATH,
                    payload={"model": model_name, **_chat_payload("priority")},
                    key=key,
                )
                _assert_billing(
                    outcome,
                    expected_cost=_expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
                    prompt_tokens=MATRIX_PROMPT_TOKENS,
                    completion_tokens=MATRIX_COMPLETION_TOKENS,
                )
                _assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_priority_behavior_survives_owned_proxy_restart_after_database_registration(
    gateway: Gateway,
    tmp_path: Path,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-database-restart-priority-{uuid.uuid4().hex}",
        )
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as first:
            key_record: Final = first.gateway.post("/key/generate", {})
            key: Final = string_value(key_record["key"])
            model_name: Final = f"custom-priced-tier-restart-{uuid.uuid4().hex}"
            created: Final = first.gateway.post(
                "/model/new",
                {
                    "model_name": model_name,
                    "litellm_params": {
                        "model": f"openai/{MATRIX_BACKEND_MODEL}",
                        "api_key": scenario_id,
                        "api_base": api_base,
                        "input_cost_per_token": CUSTOM_STANDARD_INPUT_RATE,
                        "output_cost_per_token": CUSTOM_STANDARD_OUTPUT_RATE,
                    },
                    "model_info": {},
                },
            )
            model_id: Final = string_value(object_value(created["model_info"])["id"])
            scenario.cleanups.callback(scenario.delete_model, model_id)
            scenario.cleanups.callback(scenario.delete_key, key)
        with owned_proxy_process(gateway, tmp_path, {}, workers=2) as restarted:
            outcome: Final = _invoke_http(
                restarted.gateway,
                path=MATRIX_CHAT_PATH,
                payload={"model": model_name, **_chat_payload("priority")},
                key=key,
            )
            _assert_billing(
                outcome,
                expected_cost=_expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
                prompt_tokens=MATRIX_PROMPT_TOKENS,
                completion_tokens=MATRIX_COMPLETION_TOKENS,
            )
            _assert_upstream(gateway, expected_tier="priority", tier_present=True)


def test_model_update_changes_no_tier_standard_and_keeps_priority_inheritance(gateway: Gateway) -> None:
    updated_input_rate: Final = 0.00037
    updated_output_rate: Final = 0.00082
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-model-update-priority-{uuid.uuid4().hex}",
        )
        no_tier_scenario_id, no_tier_api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response(None),
            identifier=f"tier-matrix-model-update-no-tier-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model, model_id = _register_db_model(
            gateway,
            scenario,
            model=f"openai/{MATRIX_BACKEND_MODEL}",
            scenario_id=scenario_id,
            api_base=api_base,
        )
        updated: Final = gateway.request(
            "POST",
            "/model/update",
            {
                "model_info": {"id": model_id},
                "litellm_params": {
                    "input_cost_per_token": updated_input_rate,
                    "output_cost_per_token": updated_output_rate,
                },
            },
        )
        assert updated.status_code == 200, updated.text
        priority: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload("priority")},
            key=key,
        )
        _assert_upstream(gateway, expected_tier="priority", tier_present=True)
        no_tier_update: Final = gateway.request(
            "POST",
            "/model/update",
            {
                "model_info": {"id": model_id},
                "litellm_params": {
                    "api_base": no_tier_api_base,
                    "api_key": no_tier_scenario_id,
                },
            },
        )
        assert no_tier_update.status_code == 200, no_tier_update.text
        no_tier: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload(None)},
            key=key,
        )
        _assert_billing(
            no_tier,
            expected_cost=MATRIX_PROMPT_TOKENS * updated_input_rate + MATRIX_COMPLETION_TOKENS * updated_output_rate,
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )
        _assert_upstream(gateway, expected_tier=None, tier_present=False)
        _assert_billing(
            priority,
            expected_cost=_expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )


@pytest.mark.parametrize(
    ("label", "service_tier"),
    INVALID_TIER_VALUES,
    ids=tuple(value[0] for value in INVALID_TIER_VALUES),
)
def test_malformed_or_unrecognized_tier_inputs_keep_the_observed_result(
    gateway: Gateway,
    label: str,
    service_tier: JsonValue,
) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response(service_tier),
            identifier=f"tier-matrix-malformed-tier-{label}-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload(service_tier)},
            key=key,
        )
        assert outcome.status == 200, outcome.text
        _assert_billing(
            outcome,
            expected_cost=_expected_cost(
                service_tier if isinstance(service_tier, str) else None,
                MATRIX_PROMPT_TOKENS,
                MATRIX_COMPLETION_TOKENS,
            ),
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )
        _assert_upstream(
            gateway,
            expected_tier=service_tier,
            tier_present=True,
        )


def test_uppercase_priority_tier_uses_inherited_catalog_rates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("PRIORITY"),
            identifier=f"tier-matrix-uppercase-tier-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload("PRIORITY")},
            key=key,
        )
        _assert_upstream(gateway, expected_tier="PRIORITY", tier_present=True)
        _assert_billing(
            outcome,
            expected_cost=_expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )


def test_scripted_upstream_failure_records_one_failure_spend_row(gateway: Gateway) -> None:
    expected_attempt_count: Final = 1
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=JsonResponse(
                content_type="application/json",
                body={"error": {"message": "scripted upstream failure", "type": "server_error"}},
                status=500,
            ),
            identifier=f"tier-matrix-upstream-failure-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload("priority")},
            key=key,
        )
        assert outcome.status == 500, outcome
        observed: Final = _observations(gateway)
        assert len(observed) == expected_attempt_count, observed
        for request in observed:
            _assert_forwarded_body(
                object_value(request["body"]),
                expected_tier="priority",
                tier_present=True,
            )
        rows: Final = eventually(
            lambda: _spend_rows_for_key(key),
            lambda values: len(values) == 1,
            seconds=70,
        )
        assert rows[0]["status"] == "failure", rows
        assert float(str(rows[0]["spend"])) == 0, rows


def test_null_explicit_priority_input_is_treated_as_missing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-null-tier-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            model_info={"input_cost_per_token_priority": None},
        )
        outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": model, **_chat_payload("priority")},
            key=key,
        )
        _assert_success(
            gateway,
            outcome,
            service_tier="priority",
            expected_cost=_expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS),
        )


def test_unauthenticated_priority_request_is_rejected_without_spend(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-unauthenticated-request-{uuid.uuid4().hex}",
        )
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        invalid_key: Final = f"sk-invalid-tier-matrix-{uuid.uuid4().hex}"
        response: Final = gateway.request(
            "POST",
            MATRIX_CHAT_PATH,
            {"model": model, **_chat_payload("priority")},
            key=invalid_key,
        )
        assert response.status_code == 401, response.text
        assert (
            read_rows(
                'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
                (sha256(invalid_key.encode()).hexdigest(),),
            )
            == []
        )
        assert _observations(gateway) == []


def test_three_repeated_priority_requests_create_three_identical_cost_rows(gateway: Gateway) -> None:
    expected_cost: Final = _expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS)
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-repeated-priority-requests-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        outcomes: Final = tuple(
            _invoke_http(
                gateway,
                path=MATRIX_CHAT_PATH,
                payload={"model": model, **_chat_payload("priority")},
                key=key,
            )
            for _ in range(3)
        )
        assert all(outcome.status == 200 for outcome in outcomes), outcomes
        ids: Final = tuple(outcome.request_id for outcome in outcomes)
        assert None not in ids, outcomes
        assert len(set(ids)) == 3, ids
        observed: Final = _observations(gateway)
        assert len(observed) == 3, observed
        for request in observed:
            _assert_forwarded_body(
                object_value(request["body"]),
                expected_tier="priority",
                tier_present=True,
            )
        rows: Final = tuple(_spend_rows(cast(str, request_id))[0] for request_id in ids)
        assert len(rows) == 3, rows
        for outcome in outcomes:
            _assert_cost_header(outcome, expected_cost=expected_cost)
        assert all(float(str(row["spend"])) == pytest.approx(expected_cost, rel=1e-6) for row in rows), rows
        assert all(int(str(row["prompt_tokens"])) == MATRIX_PROMPT_TOKENS for row in rows), rows
        assert all(int(str(row["completion_tokens"])) == MATRIX_COMPLETION_TOKENS for row in rows), rows


def test_priority_burst_uses_catalog_rates_during_standard_price_update(gateway: Gateway) -> None:
    updated_input_rate: Final = 0.00037
    updated_output_rate: Final = 0.00082
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_sse_response(frame_delay_ms=80),
            identifier=f"tier-matrix-standard-update-priority-burst-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model, model_id = _register_db_model(
            gateway,
            scenario,
            model=f"openai/{MATRIX_BACKEND_MODEL}",
            scenario_id=scenario_id,
            api_base=api_base,
        )
        requests: Final = tuple(
            {
                "model": model,
                **_chat_payload("priority", stream=True),
            }
            for _ in range(20)
        )
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures: Final = tuple(
                executor.submit(
                    _invoke_http,
                    gateway,
                    path=MATRIX_CHAT_PATH,
                    payload=request,
                    key=key,
                    stream=True,
                    surface="chat",
                )
                for request in requests
            )
            updated: Final = gateway.request(
                "POST",
                "/model/update",
                {
                    "model_info": {"id": model_id},
                    "litellm_params": {
                        "input_cost_per_token": updated_input_rate,
                        "output_cost_per_token": updated_output_rate,
                    },
                },
            )
            assert updated.status_code == 200, updated.text
            outcomes: Final = tuple(future.result(timeout=90) for future in futures)
        expected_cost: Final = _expected_cost("priority", MATRIX_PROMPT_TOKENS, MATRIX_COMPLETION_TOKENS)
        assert all(outcome.status == 200 for outcome in outcomes), outcomes
        for outcome in outcomes:
            _assert_cost_header(outcome, expected_cost=expected_cost, allow_missing_header=True)
        ids: Final = tuple(outcome.request_id for outcome in outcomes)
        assert None not in ids, outcomes
        rows: Final = tuple(_spend_rows(cast(str, request_id))[0] for request_id in ids)
        assert len(rows) == 20, rows
        assert all(float(str(row["spend"])) == pytest.approx(expected_cost, rel=1e-6) for row in rows), rows
        observed: Final = _observations(gateway)
        assert len(observed) == 20, observed
        assert all(object_value(request["body"]).get("service_tier") == "priority" for request in observed), observed
        assert all(
            not {
                field
                for field in object_value(request["body"])
                if "cost_per_token" in field or field.startswith("cache_read_input_token_cost")
            }
            for request in observed
        ), observed


def test_cache_hit_keeps_the_observed_second_request_billing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response("priority"),
            identifier=f"tier-matrix-cache-hit-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        model: Final = _scenario_model(scenario, scenario_id=scenario_id, api_base=api_base)
        control_payload: Final = {
            "model": model,
            **_chat_payload("priority", no_cache=False),
            "messages": [{"role": "user", "content": "cache billing control"}],
        }
        control: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload=control_payload,
            key=key,
        )
        assert control.cost_header is not None, control
        control_cost: Final = control.cost_header
        _assert_billing(
            control,
            expected_cost=control_cost,
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )
        _assert_upstream(gateway, expected_tier="priority", tier_present=True)
        payload: Final = {
            "model": model,
            **_chat_payload("priority", no_cache=False),
            "messages": [{"role": "user", "content": "cache hit billing"}],
        }
        first: Final = _invoke_http(gateway, path=MATRIX_CHAT_PATH, payload=payload, key=key)
        second: Final = _invoke_http(gateway, path=MATRIX_CHAT_PATH, payload=payload, key=key)
        assert first.status == second.status == 200, (first, second)
        assert first.request_id is not None and second.request_id is not None, (first, second)
        assert first.request_id == second.request_id, (first, second)
        expected_first_cost: Final = control_cost
        _assert_cost_header(first, expected_cost=expected_first_cost)
        _assert_cost_header(second, expected_cost=expected_first_cost)
        rows: Final = _spend_rows(first.request_id)
        assert float(str(rows[0]["spend"])) == pytest.approx(expected_first_cost, rel=1e-6), rows
        cache_hit_rows: Final = _cache_hit_spend_rows(second.request_id)
        assert cache_hit_rows[0]["cache_hit"] in ("True", True), cache_hit_rows
        assert float(str(cache_hit_rows[0]["spend"])) == 0, cache_hit_rows
        observed: Final = _observations(gateway)
        assert len(observed) == 1, observed
        _assert_forwarded_body(
            object_value(observed[0]["body"]),
            expected_tier="priority",
            tier_present=True,
        )


def test_response_priority_matches_catalog_priced_rule(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        scenario_id, api_base = _register_upstream(
            gateway,
            scenario,
            response=_chat_json_response(None, response_service_tier="priority"),
            identifier=f"tier-matrix-response-priority-{uuid.uuid4().hex}",
        )
        key: Final = scenario.key()
        custom_model: Final = _scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
        )
        catalog_model: Final = _scenario_model(
            scenario,
            scenario_id=scenario_id,
            api_base=api_base,
            custom_rates=False,
        )
        custom_outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": custom_model, **_chat_payload(None)},
            key=key,
        )
        custom_observed: Final = _assert_upstream(gateway)
        assert "service_tier" not in custom_observed
        catalog_outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": catalog_model, **_chat_payload(None)},
            key=key,
        )
        catalog_observed: Final = _assert_upstream(gateway)
        assert "service_tier" not in catalog_observed
        priority_outcome: Final = _invoke_http(
            gateway,
            path=MATRIX_CHAT_PATH,
            payload={"model": custom_model, **_chat_payload("priority")},
            key=key,
        )
        priority_observed: Final = _assert_upstream(gateway, expected_tier="priority", tier_present=True)
        catalog_cost: Final = _expected_cost(
            "priority",
            MATRIX_PROMPT_TOKENS,
            MATRIX_COMPLETION_TOKENS,
            custom_standard=False,
        )
        assert custom_outcome.body.get("service_tier") == "priority", custom_outcome.body
        assert catalog_outcome.body.get("service_tier") == "priority", catalog_outcome.body
        assert priority_outcome.body.get("service_tier") == "priority", priority_outcome.body
        assert all(
            not any("cost_per_token" in field or field.startswith("cache_read_input_token_cost") for field in request)
            for request in (custom_observed, catalog_observed, priority_observed)
        ), (custom_observed, catalog_observed, priority_observed)
        _assert_billing(
            custom_outcome,
            expected_cost=catalog_cost,
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )
        _assert_billing(
            catalog_outcome,
            expected_cost=catalog_cost,
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )
        _assert_billing(
            priority_outcome,
            expected_cost=catalog_cost,
            prompt_tokens=MATRIX_PROMPT_TOKENS,
            completion_tokens=MATRIX_COMPLETION_TOKENS,
        )

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final, Literal, cast

import httpx
import pytest
import yaml
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.upstream import JsonResponse, SseResponse, delete_scenario, register_scenario
from tests.integration.pricing.test_service_tier_pricing import (
    BUNDLED_COST_MAP,
    CUSTOM_STANDARD_INPUT_RATE,
    CUSTOM_STANDARD_OUTPUT_RATE,
)
from tests.integration.pricing.test_service_tier_pricing import (
    _bundled_rate as bundled_rate,
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


def service_tier_rates(
    service_tier: str | None,
    *,
    custom_standard: bool = True,
    long_context: bool = False,
) -> tuple[float, float]:
    if service_tier is None or service_tier == "balanced":
        if custom_standard:
            return CUSTOM_STANDARD_INPUT_RATE, CUSTOM_STANDARD_OUTPUT_RATE
        return (
            bundled_rate(MATRIX_BACKEND_MODEL, "input_cost_per_token"),
            bundled_rate(MATRIX_BACKEND_MODEL, "output_cost_per_token"),
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
        bundled_rate(MATRIX_BACKEND_MODEL, input_field)
        if input_field in MATRIX_CATALOG_ENTRY
        else CUSTOM_STANDARD_INPUT_RATE
        if custom_standard
        else bundled_rate(MATRIX_BACKEND_MODEL, "input_cost_per_token"),
        bundled_rate(MATRIX_BACKEND_MODEL, output_field)
        if output_field in MATRIX_CATALOG_ENTRY
        else CUSTOM_STANDARD_OUTPUT_RATE
        if custom_standard
        else bundled_rate(MATRIX_BACKEND_MODEL, "output_cost_per_token"),
    )


def chat_payload(
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


def chat_json_response(
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


def responses_json_response(
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


def chat_sse_response(
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


def responses_sse_response(*, service_tier: str | None = "priority") -> SseResponse:
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


def outcome(
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


def body_from_text(text: str) -> dict[str, JsonValue]:
    if not text.strip():
        return {}
    try:
        body: Final = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return object_value(body)


def stream_outcome(
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
        (candidate for event in events if (candidate := stream_event_id(event, surface)) is not None),
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
    return outcome(status=status, headers=headers, body=body, text=text)


def stream_event_id(
    event: Mapping[str, JsonValue],
    surface: Literal["chat", "responses", "messages"],
) -> str | None:
    candidate: JsonValue | None = event.get("id")
    if surface == "responses" and isinstance(event.get("response"), dict):
        candidate = object_value(event["response"]).get("id")
    if surface == "messages" and isinstance(event.get("message"), dict):
        candidate = object_value(event["message"]).get("id")
    return candidate if isinstance(candidate, str) else None


def invoke_http(
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
        return outcome(
            status=response.status_code,
            headers=response.headers,
            body=body_from_text(response.text),
            text=response.text,
        )
    with gateway.client.stream("POST", path, json=payload, headers=headers) as response:
        text: Final = response.read().decode("utf-8", errors="replace")
        return stream_outcome(
            status=response.status_code,
            headers=response.headers,
            text=text,
            surface=surface,
        )


def invoke_sync_sdk(
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
                return stream_outcome(
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
        return outcome(
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
        return outcome(
            status=response.http_response.status_code,
            headers=response.headers,
            body=cast(dict[str, JsonValue], parsed.model_dump(mode="json")),
            text=response.http_response.text,
        )


def scenario_model(
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


def register_upstream(
    gateway: Gateway,
    scenario: Scenario,
    *,
    response: JsonResponse | SseResponse,
    identifier: str | None = None,
) -> tuple[str, str]:
    observations(gateway)
    scenario_id: Final = identifier or f"tier-matrix-{uuid.uuid4().hex}"
    handle: Final = register_scenario(scenario_id, response)
    scenario.cleanups.callback(delete_scenario, handle)
    return scenario_id, handle.api_base()


def observations_at(upstream_url: str) -> list[dict[str, JsonValue]]:
    with httpx.Client(base_url=upstream_url, trust_env=False) as upstream:
        body: Final = object_value(JSON_OBJECT.validate_json(upstream.get("/__observations").content))
    return [object_value(value) for value in body["requests"]]


def observations(gateway: Gateway) -> list[dict[str, JsonValue]]:
    return observations_at(gateway.upstream_url)


def assert_forwarded_body(
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


def assert_upstream_request(
    gateway: Gateway,
    *,
    expected_count: int = 1,
    expected_tier: JsonValue | None = None,
    tier_present: bool = False,
) -> dict[str, JsonValue]:
    requests: Final = observations(gateway)
    assert len(requests) == expected_count, requests
    latest: Final = requests[-1]
    assert_forwarded_body(
        object_value(latest["body"]),
        expected_tier=expected_tier,
        tier_present=tier_present,
    )
    return latest


def assert_upstream(
    gateway: Gateway,
    *,
    expected_count: int = 1,
    expected_tier: JsonValue | None = None,
    tier_present: bool = False,
) -> dict[str, JsonValue]:
    request: Final = assert_upstream_request(
        gateway,
        expected_count=expected_count,
        expected_tier=expected_tier,
        tier_present=tier_present,
    )
    return object_value(request["body"])


def assert_azure_upstream(gateway: Gateway, *, scenario_id: str) -> None:
    upstream_observations: Final = observations(gateway)
    requests: Final = tuple(
        request
        for request in upstream_observations
        if string_value(request["path"]).startswith(f"/{scenario_id}/openai/deployments/")
    )
    assert len(requests) == 1, upstream_observations
    path: Final = string_value(requests[0]["path"]).split("?", maxsplit=1)[0]
    assert path.endswith("/chat/completions"), requests
    assert_forwarded_body(
        object_value(requests[0]["body"]),
        expected_tier="priority",
        tier_present=True,
    )


def spend_rows(request_id: str) -> list[dict[str, object]]:
    return eventually(
        lambda: read_rows(
            'SELECT request_id, spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (request_id,),
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )


def cache_hit_spend_rows(request_id: str) -> list[dict[str, JsonValue]]:
    cache_hit_prefix: Final = f"{request_id}_cache_hit"
    return eventually(
        lambda: read_rows(
            'SELECT request_id, spend, cache_hit FROM "LiteLLM_SpendLogs" WHERE LEFT(request_id, %s) = %s',
            (len(cache_hit_prefix), cache_hit_prefix),
        ),
        lambda rows: len(rows) == 1,
        seconds=70,
    )


def spend_rows_for_key(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT request_id, status, spend, prompt_tokens, completion_tokens, cache_hit FROM "LiteLLM_SpendLogs" '
        'WHERE api_key=%s ORDER BY "startTime"',
        (sha256(key.encode()).hexdigest(),),
    )


def register_db_model(
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


def write_proxy_config(
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


def assert_cost_header(
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


def assert_billing(
    outcome: MatrixOutcome,
    *,
    expected_cost: float,
    prompt_tokens: int,
    completion_tokens: int,
    allow_missing_header: bool = False,
) -> dict[str, object]:
    assert_cost_header(outcome, expected_cost=expected_cost, allow_missing_header=allow_missing_header)
    request_id: Final = outcome.request_id
    assert request_id is not None, outcome
    rows: Final = spend_rows(request_id)
    row: Final = rows[0]
    assert float(str(row["spend"])) == pytest.approx(expected_cost, rel=1e-6), row
    assert int(str(row["prompt_tokens"])) == prompt_tokens, row
    assert int(str(row["completion_tokens"])) == completion_tokens, row
    return row


def expected_cost(
    service_tier: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    custom_standard: bool = True,
    long_context: bool = False,
    cached_tokens: int = 0,
) -> float:
    input_rate, output_rate = service_tier_rates(
        service_tier,
        custom_standard=custom_standard,
        long_context=long_context,
    )
    if cached_tokens:
        cache_field: Final = f"cache_read_input_token_cost_{service_tier}"
        if service_tier is not None and cache_field in MATRIX_CATALOG_ENTRY:
            cached_rate: Final = bundled_rate(MATRIX_BACKEND_MODEL, cache_field)
            return (
                (prompt_tokens - cached_tokens) * input_rate
                + cached_tokens * cached_rate
                + completion_tokens * output_rate
            )
    return prompt_tokens * input_rate + completion_tokens * output_rate


def assert_success(
    gateway: Gateway,
    outcome: MatrixOutcome,
    *,
    service_tier: JsonValue | None,
    prompt_tokens: int = MATRIX_PROMPT_TOKENS,
    completion_tokens: int = MATRIX_COMPLETION_TOKENS,
    expected_cost: float,
    tier_present: bool | None = None,
) -> None:
    assert_billing(
        outcome,
        expected_cost=expected_cost,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )
    assert_upstream(
        gateway,
        expected_tier=service_tier,
        tier_present=service_tier is not None if tier_present is None else tier_present,
    )

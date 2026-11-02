import json
import uuid
from collections.abc import Mapping
from typing import Final

import anthropic
from integration._support.bedrock_runtime_peer import NATIVE_CHAT, answer, body_of, marker_of, respond, target_of
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.wire import Request, Wire, wire_server
from pydantic import JsonValue

BEDROCK_MODEL: Final = "us.openai.gpt-5.6-sol"
TOKEN: Final = "synthetic-bedrock-bearer"
NO_CACHE: Final[Mapping[str, JsonValue]] = {"cache": {"no-cache": True}}
ANTHROPIC_VERSION: Final[Mapping[str, str]] = {"anthropic-version": "2023-06-01"}


def _question(marker: str) -> str:
    return f"Question marker-{marker}"


def _deployment(scenario: Scenario, wire: Wire) -> str:
    return scenario.model(
        model=f"bedrock/{BEDROCK_MODEL}",
        api_key=TOKEN,
        aws_region_name="us-east-1",
        aws_bedrock_runtime_endpoint=wire.url,
    )


def _carrying(wire: Wire, marker: str) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if marker_of(request) == marker)


def _native_body(wire: Wire, marker: str) -> Mapping[str, JsonValue]:
    received: Final = _carrying(wire, marker)
    assert [(request.method, target_of(request)) for request in received] == [("POST", NATIVE_CHAT)]
    assert received[0].headers["authorization"] == f"Bearer {TOKEN}", received[0].headers
    return body_of(received[0])


def _native_request(marker: str, max_tokens: int, effort: str) -> Mapping[str, JsonValue]:
    return {
        "model": BEDROCK_MODEL,
        "messages": [{"role": "user", "content": _question(marker)}],
        "max_completion_tokens": max_tokens,
        "reasoning_effort": effort,
    }


def _spend_rows(identity: str, expected: int) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(
            "SELECT request_id, call_type, status, model_group, prompt_tokens, completion_tokens, cache_hit"
            ' FROM "LiteLLM_SpendLogs" WHERE starts_with(request_id, %s) ORDER BY "startTime"',
            (identity,),
        ),
        lambda found: len(found) == expected,
        seconds=70,
    )


def _success_row(identity: str, model: str, cache_hit: str = "None") -> dict[str, JsonValue]:
    return {
        "request_id": identity,
        "call_type": "anthropic_messages",
        "status": "success",
        "model_group": model,
        "prompt_tokens": 9,
        "completion_tokens": 5,
        "cache_hit": cache_hit,
    }


def test_anthropic_sdk_thinking_budget_reaches_native_chat_completions_as_reasoning_effort(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        message: Final = client.messages.create(
            model=model,
            max_tokens=4096,
            thinking={"type": "enabled", "budget_tokens": 2048},
            messages=[{"role": "user", "content": _question(marker)}],
            extra_body=NO_CACHE,
        )
        assert _native_body(wire, marker) == _native_request(marker, 4096, "medium")
        assert message.id == f"chatcmpl-{marker}", message
        assert [(block.type, getattr(block, "text", None)) for block in message.content] == [("text", answer(marker))]
        assert (message.usage.input_tokens, message.usage.output_tokens) == (9, 5), message
        assert _spend_rows(message.id, 1) == [_success_row(message.id, model)]


def test_anthropic_sdk_stream_with_thinking_budget_is_served_by_native_chat_completions(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        client: Final = anthropic.Anthropic(base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0)
        stream: Final = client.messages.create(
            model=model,
            max_tokens=4096,
            thinking={"type": "enabled", "budget_tokens": 2048},
            messages=[{"role": "user", "content": _question(marker)}],
            extra_body=NO_CACHE,
            stream=True,
        )
        events: Final = list(stream)
        assert _native_body(wire, marker) == {
            **_native_request(marker, 4096, "medium"),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        assert events[0].type == "message_start" and events[-1].type == "message_stop", events
        identity: Final = events[0].message.id
        assert identity.startswith("msg_"), events
        assert "".join(
            event.delta.text
            for event in events
            if event.type == "content_block_delta" and event.delta.type == "text_delta"
        ) == answer(marker)
        assert _spend_rows(identity, 1) == [_success_row(identity, model, cache_hit="False")]


def test_raw_thinking_summary_reaches_native_chat_completions_as_the_plain_effort(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {
                "model": model,
                "max_tokens": 4096,
                "thinking": {"type": "enabled", "budget_tokens": 2048, "summary": "detailed"},
                "messages": [{"role": "user", "content": _question(marker)}],
                **NO_CACHE,
            },
            headers=ANTHROPIC_VERSION,
        )
        body: Final = _native_body(wire, marker)
        assert body == _native_request(marker, 4096, "medium")
        assert "summary" not in json.dumps(body), body
        assert response.status_code == 200, response.text
        assert response.json()["id"] == f"chatcmpl-{marker}", response.text
        assert response.json()["content"] == [{"type": "text", "text": answer(marker)}], response.text
        assert _spend_rows(f"chatcmpl-{marker}", 1) == [_success_row(f"chatcmpl-{marker}", model)]


async def test_async_anthropic_sdk_disabled_thinking_reaches_native_chat_completions_as_effort_none(
    gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        client: Final = anthropic.AsyncAnthropic(
            base_url=str(gateway.client.base_url), api_key=gateway.key, max_retries=0
        )
        message: Final = await client.messages.create(
            model=model,
            max_tokens=64,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": _question(marker)}],
            extra_body=NO_CACHE,
        )
        assert _native_body(wire, marker) == _native_request(marker, 64, "none")
        assert message.id == f"chatcmpl-{marker}", message
        assert [(block.type, getattr(block, "text", None)) for block in message.content] == [("text", answer(marker))]
        assert _spend_rows(message.id, 1) == [_success_row(message.id, model)]


def test_identical_messages_requests_reach_the_peer_once_and_log_a_cache_hit_row(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = _deployment(scenario, wire)
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "max_tokens": 64,
            "messages": [{"role": "user", "content": _question(marker)}],
        }
        first: Final = gateway.request("POST", "/v1/messages", body, headers=ANTHROPIC_VERSION)
        assert first.status_code == 200, first.text
        identity: Final = str(first.json()["id"])
        assert first.json()["content"] == [{"type": "text", "text": answer(marker)}], first.text
        second: Final = gateway.request("POST", "/v1/messages", body, headers=ANTHROPIC_VERSION)
        assert second.status_code == 200, second.text
        assert second.json()["id"] == identity, (first.text, second.text)
        assert second.json()["content"] == [{"type": "text", "text": answer(marker)}], second.text
        received: Final = _carrying(wire, marker)
        assert [(request.method, marker_of(request)) for request in received] == [("POST", marker)], received
        rows: Final = _spend_rows(identity, 2)
        assert rows[0] == _success_row(identity, model), rows
        assert str(rows[1]["request_id"]).startswith(identity + "_cache_hit"), rows
        assert {**rows[1], "request_id": identity, "cache_hit": "None"} == _success_row(identity, model), rows

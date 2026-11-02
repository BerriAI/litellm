"""Auto-mode safeguards on the chat and responses bridges behind /v1/messages: the proxy classifies with the
configured model, stamps ``safeguard_results`` on the reply, and keeps the request's safeguards away from the backend
exactly as the bridges did before."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Final

import anthropic
import pytest
from integration._support.process import graceful_stop_seconds
from integration._support.safeguards_rig import (
    BASH_TOOL,
    CHAT_MODEL,
    DANGEROUS_TOOL_USE_BETA,
    RESPONSES_MODEL,
    SAFEGUARDS,
    Responder,
    Rig,
    auto_mode_body,
    available,
    chat_backend,
    classifier_prompt,
    classifier_rows,
    flagged,
    ids_under_review,
    json_body,
    key_hash,
    messages_body,
    nonce,
    one_classifier_call,
    request_json,
    requested_max_tokens,
    responses_backend,
    spend_rows_for,
    stream_results,
    stream_tool_use_ids,
    tool_id,
    tool_use_ids,
    unavailable,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(graceful_stop_seconds() + 240)

BRIDGES: Final = (CHAT_MODEL, RESPONSES_MODEL)


def backend(model: str, identity: str, *, stream: bool, text: str | None = None) -> Responder:
    if model == CHAT_MODEL:
        return chat_backend(identity, stream=stream, text=text, finish_reason="stop" if text else "tool_calls")
    return responses_backend(identity, stream=stream, text=text)


def sdk_client(rig: Rig, key: str | None = None) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        base_url=str(rig.gateway.client.base_url),
        api_key=key or rig.gateway.key,
        default_headers={"anthropic-beta": DANGEROUS_TOOL_USE_BETA},
        max_retries=0,
    )


def sdk_async_client(rig: Rig) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(rig.gateway.client.base_url),
        api_key=rig.gateway.key,
        default_headers={"anthropic-beta": DANGEROUS_TOOL_USE_BETA},
        max_retries=0,
    )


def assert_classified_once(rig: Rig, request_nonce: str, ids: tuple[str, ...]) -> None:
    classifier_call, backend_calls = one_classifier_call(rig, request_nonce)
    assert len(backend_calls) == 1, [request.target for request in backend_calls]
    assert ids_under_review(classifier_call) == ids, classifier_call.body
    classifier_body: Final = request_json(classifier_call)
    assert classifier_body["model"] == "gpt-5.4-mini", classifier_body
    assert requested_max_tokens(classifier_body) == 4096, classifier_body
    prompt: Final = classifier_prompt(classifier_call)
    assert "never-forwarded" not in prompt and "remote_url" not in prompt, prompt
    assert "/work/repo" in prompt and "permission_mode" in prompt, prompt


@pytest.mark.parametrize("stream", (False, True), ids=("non_stream", "stream"))
@pytest.mark.parametrize("model", BRIDGES)
def test_bridge_backend_body_carries_no_safeguards(rig: Rig, model: str, stream: bool) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, backend(model, tool_id(request_nonce), stream=stream))
    body: Final = auto_mode_body(model, request_nonce)
    status: Final = rig.stream(body)[0] if stream else rig.post(body).status_code
    assert status == 200, status
    backend_calls: Final = rig.backend_calls(rig.received(request_nonce))
    assert len(backend_calls) == 1, [request.target for request in backend_calls]
    assert "safeguards" not in request_json(backend_calls[0]), backend_calls[0].body


@pytest.mark.parametrize("model", BRIDGES)
def test_non_stream_reply_carries_the_classifier_verdict(rig: Rig, model: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, backend(model, tool_id(request_nonce), stream=False))
    body: Final = json_body(rig.post(auto_mode_body(model, request_nonce)))
    ids: Final = tool_use_ids(body)
    assert len(ids) == 1, body
    assert body["safeguard_results"] == flagged(ids), body
    assert_classified_once(rig, request_nonce, ids)


@pytest.mark.parametrize("model", BRIDGES)
def test_stream_final_delta_carries_the_classifier_verdict(rig: Rig, model: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, backend(model, tool_id(request_nonce), stream=True))
    status, raw = rig.stream(auto_mode_body(model, request_nonce))
    assert status == 200, raw
    ids: Final = stream_tool_use_ids(raw)
    assert len(ids) == 1, raw
    assert stream_results(raw) == flagged(ids), raw
    assert_classified_once(rig, request_nonce, ids)


@pytest.mark.parametrize("model", BRIDGES)
def test_anthropic_sdk_sync_client_reads_the_verdict(rig: Rig, model: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, backend(model, tool_id(request_nonce), stream=False))
    raw: Final = sdk_client(rig).messages.with_raw_response.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": f"clean the build directory {request_nonce}"}],
        tools=[dict(BASH_TOOL)],  # pyright: ignore[reportArgumentType]  # the SDK's ToolParam is a TypedDict of this shape
        extra_body={"safeguards": list(SAFEGUARDS)},
    )
    parsed: Final = raw.parse().model_dump(mode="json")
    ids: Final = tool_use_ids(parsed)
    assert len(ids) == 1, parsed
    assert parsed["safeguard_results"] == flagged(ids), parsed
    assert raw.http_response.json()["safeguard_results"] == flagged(ids), raw.http_response.text
    assert_classified_once(rig, request_nonce, ids)


@pytest.mark.parametrize("model", BRIDGES)
def test_anthropic_sdk_async_stream_reads_the_verdict(rig: Rig, model: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, backend(model, tool_id(request_nonce), stream=True))

    async def consume() -> tuple[tuple[str, ...], list[JsonValue]]:
        async with sdk_async_client(rig).messages.stream(
            model=model,
            max_tokens=256,
            messages=[{"role": "user", "content": f"clean the build directory {request_nonce}"}],
            tools=[dict(BASH_TOOL)],  # pyright: ignore[reportArgumentType]  # the SDK's ToolParam is a TypedDict of this shape
            extra_body={"safeguards": list(SAFEGUARDS)},
        ) as stream:
            events: Final = [event.model_dump(mode="json") async for event in stream]
        starts: Final = [event["content_block"] for event in events if event["type"] == "content_block_start"]
        ids: Final = tuple(block["id"] for block in starts if block["type"] == "tool_use")
        deltas: Final = [
            event["delta"].get("safeguard_results") for event in events if event["type"] == "message_delta"
        ]
        return ids, deltas

    ids, deltas = asyncio.run(consume())
    assert len(ids) == 1, ids
    assert deltas == [flagged(ids)], deltas
    assert_classified_once(rig, request_nonce, ids)


def test_classifier_spend_lands_on_the_calling_key(rig: Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key()
        request_nonce: Final = nonce()
        rig.register(request_nonce, backend(CHAT_MODEL, tool_id(request_nonce), stream=False))
        body: Final = json_body(rig.post(auto_mode_body(CHAT_MODEL, request_nonce), key=key))
        assert body["safeguard_results"] == flagged(tool_use_ids(body)), body
        rows: Final = spend_rows_for(key, 2)
        assert len(rows) == 2, rows
        assert {str(row["api_key"]) for row in rows} == {key_hash(key)}, rows
        classifier: Final = classifier_rows(rows)
        assert len(classifier) == 1, rows
        assert float(str(classifier[0]["spend"])) > 0, classifier
        assert all(float(str(row["spend"])) > 0 for row in rows), rows


@pytest.mark.parametrize("model", BRIDGES)
def test_text_only_reply_is_available_with_no_verdicts_and_no_classifier_call(rig: Rig, model: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, backend(model, tool_id(request_nonce), stream=False, text="all clean"))
    body: Final = json_body(rig.post(auto_mode_body(model, request_nonce)))
    assert tool_use_ids(body) == (), body
    assert body["safeguard_results"] == available({}), body
    received: Final = rig.received(request_nonce)
    assert rig.classifier_calls(received) == (), [request.target for request in received]
    assert len(rig.backend_calls(received)) == 1, [request.target for request in received]


def test_truncated_chat_tool_call_is_unavailable_without_a_classifier_call(rig: Rig) -> None:
    request_nonce: Final = nonce()
    identity: Final = tool_id(request_nonce)
    rig.register(request_nonce, chat_backend(identity, stream=False, finish_reason="length"))
    body: Final = json_body(rig.post(auto_mode_body(CHAT_MODEL, request_nonce)))
    ids: Final = tool_use_ids(body)
    assert body["stop_reason"] == "max_tokens", body
    assert body["safeguard_results"] == unavailable(ids, "truncated"), body
    assert rig.classifier_calls(rig.received(request_nonce)) == ()


def test_truncated_responses_tool_call_is_unavailable_without_a_classifier_call(rig: Rig) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, responses_backend(tool_id(request_nonce), stream=False, incomplete=True))
    body: Final = json_body(rig.post(auto_mode_body(RESPONSES_MODEL, request_nonce)))
    ids: Final = tool_use_ids(body)
    assert body["stop_reason"] == "max_tokens", body
    assert body["safeguard_results"] == unavailable(ids, "truncated"), body
    assert rig.classifier_calls(rig.received(request_nonce)) == ()


@pytest.mark.parametrize("model", BRIDGES)
def test_identical_twin_carries_the_verdict_its_cached_classifier_call_returned(rig: Rig, model: str) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, backend(model, tool_id(request_nonce), stream=False))
    request: Final[Mapping[str, JsonValue]] = auto_mode_body(model, request_nonce)
    first: Final = json_body(rig.post(request))
    second: Final = json_body(rig.post(request))
    ids: Final = tool_use_ids(first)
    assert tool_use_ids(second) == ids, second
    assert first["safeguard_results"] == flagged(ids) == second["safeguard_results"], (first, second)
    received: Final = rig.received(request_nonce)
    assert len(rig.classifier_calls(received)) == 1, [request.target for request in received]
    assert len(rig.backend_calls(received)) == 1, [request.target for request in received]


def test_chat_bridge_cache_hit_after_a_request_without_safeguards_is_still_classified(rig: Rig) -> None:
    request_nonce: Final = nonce()
    rig.register(request_nonce, backend(CHAT_MODEL, tool_id(request_nonce), stream=False))
    plain: Final = json_body(rig.post(messages_body(CHAT_MODEL, request_nonce), beta=None))
    assert "safeguard_results" not in plain, plain
    with_safeguards: Final = json_body(rig.post(auto_mode_body(CHAT_MODEL, request_nonce)))
    ids: Final = tool_use_ids(with_safeguards)
    assert ids == tool_use_ids(plain), (plain, with_safeguards)
    assert with_safeguards["safeguard_results"] == flagged(ids), with_safeguards
    received: Final = rig.received(request_nonce)
    assert len(rig.classifier_calls(received)) == 1, [request.target for request in received]
    assert len(rig.backend_calls(received)) == 1, [request.target for request in received]

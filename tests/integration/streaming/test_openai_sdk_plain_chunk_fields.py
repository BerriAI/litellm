import json
import uuid
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import pytest
from integration._support.client import Gateway, eventually, list_value, object_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
PROVIDER_FIELDS: Final = MappingProxyType({"tier": "gold"})


def _frame(identity: str, choice: Mapping[str, JsonValue] | None, **chunk_fields: JsonValue) -> bytes:
    value: Final = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [] if choice is None else [{"index": 0, "finish_reason": None, **choice}],
        **chunk_fields,
    }
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def _upstream_stream(identity: str) -> tuple[bytes, ...]:
    return (
        _frame(identity, {"delta": {"role": "assistant", "content": ""}}, system_fingerprint="fp_audit"),
        _frame(
            identity,
            {
                "delta": {"content": "Hel", "reasoning_content": "plan"},
                "content_filter_results": {"hate": {"filtered": False, "severity": "safe"}},
            },
        ),
        _frame(identity, {"delta": {"role": "assistant", "content": "lo"}}),
        _frame(identity, {"delta": {"content": " there"}}),
        _frame(identity, {"delta": {"content": "!", "provider_specific_fields": dict(PROVIDER_FIELDS)}}),
        _frame(identity, {"delta": {}, "finish_reason": "stop"}),
        _frame(identity, None, usage={"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}),
        b"data: [DONE]\n\n",
    )


def _deployment(provider: str, api_base: str) -> dict[str, JsonValue]:
    if provider == "azure":
        return {
            "model": "azure/gpt-4.1-mini",
            "api_base": api_base,
            "api_key": "synthetic-azure-key",
            "api_version": "2024-10-21",
        }
    return {"model": "openai/gpt-4o-mini", "api_base": f"{api_base}/v1"}


def _stream_events(gateway: Gateway, model: str, prompt: str) -> tuple[dict[str, JsonValue], ...]:
    with gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        headers={"Authorization": f"Bearer {gateway.key}"},
    ) as response:
        assert response.status_code == 200, response.read().decode()
        return tuple(
            JSON_OBJECT.validate_json(line.removeprefix("data: "))
            for line in response.iter_lines()
            if line.startswith("data: ") and line != "data: [DONE]"
        )


def _deltas(events: tuple[dict[str, JsonValue], ...]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        object_value(object_value(list_value(event["choices"])[0])["delta"]) for event in events if event["choices"]
    )


def _event_with_content(events: tuple[dict[str, JsonValue], ...], content: str) -> dict[str, JsonValue]:
    matches: Final = tuple(
        event
        for event in events
        if event["choices"]
        and object_value(object_value(list_value(event["choices"])[0])["delta"]).get("content") == content
    )
    assert len(matches) == 1, f"{content!r} arrived in {len(matches)} events"
    return matches[0]


@pytest.mark.parametrize("provider", ["openai", "azure"])
def test_plain_chunks_keep_text_role_and_provider_fields(gateway: Gateway, provider: str) -> None:
    identity: Final = f"plain-chunks-{uuid.uuid4().hex}"

    def upstream(request: Request) -> Reply:
        return Reply(content_type="text/event-stream", chunks=_upstream_stream(identity))

    with wire_server(upstream) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            **_deployment(provider, wire.url), input_cost_per_token=0.001, output_cost_per_token=0.002
        )
        events: Final = _stream_events(gateway, model, identity)
        received: Final = wire.drain()

    assert len(received) == 1
    assert json.loads(received[0].body)["stream"] is True
    deltas: Final = _deltas(events)
    assert "".join(str(delta.get("content") or "") for delta in deltas) == "Hello there!"
    assert tuple(delta.get("role") for delta in deltas if delta.get("role") is not None) == ("assistant",)
    assert deltas[0].get("role") == "assistant"
    assert {event["id"] for event in events} == {identity}
    assert events[0].get("system_fingerprint") == "fp_audit"
    hel_choice: Final = object_value(list_value(_event_with_content(events, "Hel")["choices"])[0])
    assert object_value(hel_choice["delta"]).get("reasoning_content") == "plan"
    assert hel_choice.get("content_filter_results") == {"hate": {"filtered": False, "severity": "safe"}}
    bang_delta: Final = object_value(object_value(list_value(_event_with_content(events, "!")["choices"])[0])["delta"])
    assert bang_delta.get("provider_specific_fields") == dict(PROVIDER_FIELDS)
    assert tuple(
        object_value(list_value(event["choices"])[0]).get("finish_reason")
        for event in events
        if event["choices"] and object_value(list_value(event["choices"])[0]).get("finish_reason") is not None
    ) == ("stop",)
    usages: Final = tuple(object_value(event["usage"]) for event in events if event.get("usage"))
    assert tuple((usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"]) for usage in usages) == (
        (11, 4, 15),
    )
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT spend, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
            (identity,),
            database_url=gateway.database_url,
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert (rows[0]["prompt_tokens"], rows[0]["completion_tokens"]) == (11, 4)
    assert float(str(rows[0]["spend"])) == pytest.approx(11 * 0.001 + 4 * 0.002)

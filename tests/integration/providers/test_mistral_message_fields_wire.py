import uuid
from collections.abc import Mapping, Sequence
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, object_value

_MODEL: Final = "mistral/mistral-small-latest"
_API_KEY: Final = "synthetic-mistral-key"
_TOOL_CALL_ID: Final = "D681PevKs"
_WEATHER_CALL: Final[JsonValue] = {
    "id": _TOOL_CALL_ID,
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
}
_WEATHER_TOOL: Final[JsonValue] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
    },
}


def _tool_conversation(marker: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        {"role": "system", "content": f"You are a terse assistant. {marker}"},
        {"role": "user", "content": "What is the weather in Paris?"},
        {"role": "assistant", "content": "Let me check.", "tool_calls": [_WEATHER_CALL]},
        {"role": "tool", "tool_call_id": _TOOL_CALL_ID, "name": "get_weather", "content": "22 degrees and sunny"},
        {"role": "assistant", "content": "It is 22 degrees and sunny in Paris."},
        {"role": "user", "content": "Thanks. And should I bring a jacket?"},
    )


def _with_fields(
    conversation: Sequence[dict[str, JsonValue]], extra: Mapping[int, dict[str, JsonValue]]
) -> tuple[dict[str, JsonValue], ...]:
    return tuple({**message, **extra.get(position, {})} for position, message in enumerate(conversation))


def _copilot_conversation(marker: str) -> tuple[dict[str, JsonValue], ...]:
    return _with_fields(
        _tool_conversation(marker),
        {
            2: {"cot_id": "cot-1f6b2c", "cot_summary": "Need the weather tool for Paris."},
            4: {"cot_id": "cot-9a04de", "cot_summary": "Answer from the tool result."},
        },
    )


def _send(
    gateway: Gateway,
    upstream: httpx.Client,
    model: str,
    marker: str,
    messages: Sequence[dict[str, JsonValue]],
    extra_body: Mapping[str, JsonValue] | None = None,
) -> list[JsonValue]:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": list(messages), "tools": [_WEATHER_TOOL], **(extra_body or {})},
    )
    assert response.status_code == 200, response.text
    observations: Final = JSON_OBJECT.validate_json(upstream.get("/__observations").content)["requests"]
    assert isinstance(observations, list)
    bodies: Final = tuple(object_value(object_value(observation)["body"]) for observation in observations)
    matching: Final = tuple(body for body in bodies if marker in str(body.get("messages")))
    assert len(matching) == 1, f"expected one upstream call carrying {marker}, got {len(matching)}: {response.text}"
    sent: Final = matching[0]["messages"]
    assert isinstance(sent, list), sent
    return sent


def _marker() -> str:
    return f"marker-{uuid.uuid4().hex}"


def test_deployment_drop_params_strips_copilot_cot_fields_from_assistant_messages(gateway: Gateway) -> None:
    marker: Final = _marker()
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        model: Final = scenario.model(model=_MODEL, api_key=_API_KEY, drop_params=True)
        sent: Final = _send(gateway, upstream, model, marker, _copilot_conversation(marker))
    assert sent == list(_tool_conversation(marker))


def test_request_drop_params_strips_metadata_from_user_messages(gateway: Gateway) -> None:
    marker: Final = _marker()
    with_metadata: Final = _with_fields(
        _tool_conversation(marker),
        {1: {"metadata": {"source": "chat-panel"}}, 5: {"metadata": {"source": "chat-panel", "turn": 2}}},
    )
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        model: Final = scenario.model(model=_MODEL, api_key=_API_KEY)
        sent: Final = _send(gateway, upstream, model, marker, with_metadata, {"drop_params": True})
    assert sent == list(_tool_conversation(marker))


def test_drop_params_strips_unknown_fields_from_system_and_tool_messages(gateway: Gateway) -> None:
    marker: Final = _marker()
    with_client_fields: Final = _with_fields(
        _tool_conversation(marker),
        {0: {"metadata": {"pinned": True}}, 3: {"metadata": {"elapsed_ms": 41}, "is_error": False}},
    )
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        model: Final = scenario.model(model=_MODEL, api_key=_API_KEY, drop_params=True)
        sent: Final = _send(gateway, upstream, model, marker, with_client_fields)
    assert sent == list(_tool_conversation(marker))


def test_without_drop_params_a_custom_api_base_receives_client_fields_unchanged(gateway: Gateway) -> None:
    marker: Final = _marker()
    with_client_fields: Final = _with_fields(_copilot_conversation(marker), {1: {"metadata": {"source": "chat-panel"}}})
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        model: Final = scenario.model(model=_MODEL, api_key=_API_KEY)
        sent: Final = _send(gateway, upstream, model, marker, with_client_fields)
    assert sent == list(with_client_fields)


@pytest.mark.parametrize("drop_params", [True, False], ids=["drop_params_on", "drop_params_off"])
def test_clean_tool_conversation_reaches_mistral_unchanged(gateway: Gateway, drop_params: bool) -> None:
    marker: Final = _marker()
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        model: Final = scenario.model(model=_MODEL, api_key=_API_KEY, drop_params=drop_params)
        sent: Final = _send(gateway, upstream, model, marker, _tool_conversation(marker))
    assert sent == list(_tool_conversation(marker))


def test_drop_params_keeps_prefix_on_the_final_assistant_turn(gateway: Gateway) -> None:
    marker: Final = _marker()
    conversation: Final[tuple[dict[str, JsonValue], ...]] = (
        {"role": "system", "content": f"Answer in one sentence. {marker}"},
        {"role": "user", "content": "Name the capital of France."},
        {"role": "assistant", "content": "The capital of France is", "prefix": True},
    )
    with gateway.scenario() as scenario, httpx.Client(base_url=gateway.upstream_url, trust_env=False) as upstream:
        model: Final = scenario.model(model=_MODEL, api_key=_API_KEY, drop_params=True)
        sent: Final = _send(gateway, upstream, model, marker, conversation)
    assert sent == list(conversation)

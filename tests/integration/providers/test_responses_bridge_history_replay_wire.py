import json
import uuid
from collections.abc import Sequence
from functools import reduce
from itertools import chain
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.responses_replay import (
    CALL_ONLY_TURN,
    MODEL,
    PATCH_NOTE,
    PATCH_TOOL,
    PATCH_TURN,
    PICK,
    PLAN,
    TWO_THOUGHTS_REPLAY,
    TWO_THOUGHTS_TURN,
    WEATHER_REPLAY,
    WEATHER_TOOL,
    WEATHER_TURN,
    posted_inputs,
    responses_upstream,
)
from integration._support.wire import wire_server
from pydantic import JsonValue

_API_KEY: Final = "synthetic-openai-key"


def _chunks(response: httpx.Response) -> list[dict[str, JsonValue]]:
    return [
        object_value(json.loads(line[6:]))
        for line in response.iter_lines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


def _tool_call(index: JsonValue, calls: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    parts: Final = [call for call in calls if call["index"] == index]
    functions: Final = [object_value(part["function"]) for part in parts]
    return {
        "id": next(part["id"] for part in parts if part.get("id")),
        "type": "function",
        "function": {
            "name": next(function["name"] for function in functions if function.get("name")),
            "arguments": "".join(str(function.get("arguments") or "") for function in functions),
        },
    }


def _deltas(chunks: Sequence[dict[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    return [
        object_value(object_value(choice)["delta"])
        for choice in chain.from_iterable(list_value(chunk["choices"]) for chunk in chunks)
    ]


def _assembled(chunks: Sequence[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    deltas: Final = _deltas(chunks)
    calls: Final = [object_value(call) for call in chain.from_iterable(d.get("tool_calls") or [] for d in deltas)]
    fields: Final = reduce(
        lambda merged, delta: {**merged, **object_value(delta.get("provider_specific_fields") or {})}, deltas, {}
    )
    return {
        "role": "assistant",
        "content": "".join(str(delta["content"]) for delta in deltas if delta.get("content")) or None,
        "tool_calls": [_tool_call(index, calls) for index in sorted({int(str(call["index"])) for call in calls})],
        "reasoning_items": list(chain.from_iterable(list_value(d.get("reasoning_items") or []) for d in deltas)),
        "provider_specific_fields": fields,
    }


def _first_turn(
    gateway: Gateway, model: str, prompt: str, tools: Sequence[dict[str, JsonValue]], *, stream: bool
) -> dict[str, JsonValue]:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "tools": list(tools),
            "stream": stream,
            "cache": {"no-cache": True},
        },
    )
    assert response.status_code == 200, response.text
    if stream:
        return _assembled(_chunks(response))
    return object_value(object_value(list_value(response.json()["choices"])[0])["message"])


def _next_turn(
    gateway: Gateway,
    model: str,
    prompt: str,
    tools: Sequence[dict[str, JsonValue]],
    message: dict[str, JsonValue],
    **extra: JsonValue,
) -> httpx.Response:
    call_id: Final = object_value(list_value(message["tool_calls"])[0])["id"]
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
                message,
                {"role": "tool", "tool_call_id": call_id, "content": "Rain, 18C"},
            ],
            "tools": list(tools),
            "cache": {"no-cache": True},
            **extra,
        },
    )


def _with_phase(message: dict[str, JsonValue], phase: str) -> dict[str, JsonValue]:
    fields: Final = object_value(message.get("provider_specific_fields") or {})
    return {**message, "provider_specific_fields": {**fields, "phase": phase}}


def _assistant_messages(items: Sequence[tuple[JsonValue, ...]]) -> list[tuple[JsonValue, ...]]:
    return [item for item in items if item[:2] == ("message", "assistant")]


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_chat_next_turn_replays_responses_items_in_the_order_openai_produced(gateway: Gateway, stream: bool) -> None:
    prompt: Final = f"Pick a city and check whether it is raining there {uuid.uuid4().hex}"
    with wire_server(responses_upstream(WEATHER_TURN)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/responses/{MODEL}", api_base=wire.url, api_key=_API_KEY)
        message: Final = _first_turn(gateway, model, prompt, [WEATHER_TOOL], stream=stream)
        response: Final = _next_turn(gateway, model, prompt, [WEATHER_TOOL], message)
        inputs: Final = posted_inputs(wire)
    assert response.status_code == 200, response.text
    assert message["content"] == PLAN + PICK, message
    assert object_value(message["provider_specific_fields"])["phase"] == "commentary", message
    assert inputs[1] == (("message", "user", None, prompt), *WEATHER_REPLAY), inputs


def test_chat_next_turn_sends_the_phase_the_app_set_on_the_assistant_message(gateway: Gateway) -> None:
    prompt: Final = f"Pick a city and check whether it is raining there {uuid.uuid4().hex}"
    with wire_server(responses_upstream(WEATHER_TURN)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/responses/{MODEL}", api_base=wire.url, api_key=_API_KEY)
        message: Final = _first_turn(gateway, model, prompt, [WEATHER_TOOL], stream=False)
        response: Final = _next_turn(gateway, model, prompt, [WEATHER_TOOL], _with_phase(message, "final_answer"))
        inputs: Final = posted_inputs(wire)
    assert response.status_code == 200, response.text
    assert [item[2] for item in _assistant_messages(inputs[1])] == ["final_answer"], inputs


def test_chat_next_turn_rejects_an_undefined_phase_unless_drop_params_is_set(gateway: Gateway) -> None:
    prompt: Final = f"Pick a city and check whether it is raining there {uuid.uuid4().hex}"
    with wire_server(responses_upstream(WEATHER_TURN)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/responses/{MODEL}", api_base=wire.url, api_key=_API_KEY)
        message: Final = _with_phase(_first_turn(gateway, model, prompt, [WEATHER_TOOL], stream=False), "analysis")
        rejected: Final = _next_turn(gateway, model, prompt, [WEATHER_TOOL], message)
        dropped: Final = _next_turn(gateway, model, prompt, [WEATHER_TOOL], message, drop_params=True)
        inputs: Final = posted_inputs(wire)
    assert rejected.status_code == 400, rejected.text
    assert "'analysis'" in rejected.text, rejected.text
    assert dropped.status_code == 200, dropped.text
    assert len(inputs) == 2, inputs
    assert [item[2] for item in _assistant_messages(inputs[1])] == [None], inputs


def test_chat_next_turn_sends_the_text_the_app_edited_instead_of_the_recorded_items(gateway: Gateway) -> None:
    prompt: Final = f"Pick a city and check whether it is raining there {uuid.uuid4().hex}"
    edited_text: Final = "I picked Tokyo."
    with wire_server(responses_upstream(WEATHER_TURN)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/responses/{MODEL}", api_base=wire.url, api_key=_API_KEY)
        message: Final = _first_turn(gateway, model, prompt, [WEATHER_TOOL], stream=False)
        response: Final = _next_turn(gateway, model, prompt, [WEATHER_TOOL], {**message, "content": edited_text})
        inputs: Final = posted_inputs(wire)
    assert response.status_code == 200, response.text
    assert _assistant_messages(inputs[1]) == [("message", "assistant", "commentary", edited_text)], inputs


def test_chat_next_turn_replays_a_custom_tool_call_as_a_custom_tool_call(gateway: Gateway) -> None:
    prompt: Final = f"Create hello.txt containing hi {uuid.uuid4().hex}"
    with wire_server(responses_upstream(PATCH_TURN)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/responses/{MODEL}", api_base=wire.url, api_key=_API_KEY)
        message: Final = _first_turn(gateway, model, prompt, [PATCH_TOOL], stream=False)
        response: Final = _next_turn(gateway, model, prompt, [PATCH_TOOL], message)
        inputs: Final = posted_inputs(wire)
    assert response.status_code == 200, response.text
    assert inputs[1] == (
        ("message", "user", None, prompt),
        ("message", "assistant", "commentary", PATCH_NOTE),
        ("custom_tool_call", "call_patch", "apply_patch"),
        ("custom_tool_call_output", "call_patch"),
    ), inputs


@pytest.mark.parametrize("stream", [False, True], ids=["json", "stream"])
def test_chat_reply_keeps_every_reasoning_item_and_the_next_turn_replays_each_one(
    gateway: Gateway, stream: bool
) -> None:
    prompt: Final = f"Weigh two cities, then check whether it is raining in your pick {uuid.uuid4().hex}"
    with wire_server(responses_upstream(TWO_THOUGHTS_TURN)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/responses/{MODEL}", api_base=wire.url, api_key=_API_KEY)
        message: Final = _first_turn(gateway, model, prompt, [WEATHER_TOOL], stream=stream)
        response: Final = _next_turn(gateway, model, prompt, [WEATHER_TOOL], message)
        inputs: Final = posted_inputs(wire)
    assert response.status_code == 200, response.text
    assert [object_value(item)["id"] for item in list_value(message["reasoning_items"])] == ["rs_scan", "rs_pick"]
    assert inputs[1] == (("message", "user", None, prompt), *TWO_THOUGHTS_REPLAY), inputs


def test_chat_stream_of_a_tool_call_only_turn_carries_null_content_like_the_json_reply(gateway: Gateway) -> None:
    prompt: Final = f"Check whether it is raining in Tokyo {uuid.uuid4().hex}"
    with wire_server(responses_upstream(CALL_ONLY_TURN)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"openai/responses/{MODEL}", api_base=wire.url, api_key=_API_KEY)
        message: Final = _first_turn(gateway, model, prompt, [WEATHER_TOOL], stream=False)
        streamed: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": [WEATHER_TOOL],
                "stream": True,
                "cache": {"no-cache": True},
            },
        )
    assert streamed.status_code == 200, streamed.text
    deltas: Final = _deltas(_chunks(streamed))
    assert message["content"] is None, message
    assert [delta["content"] for delta in deltas if delta.get("content") is not None] == [], deltas

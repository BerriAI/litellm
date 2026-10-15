import json
from collections.abc import Callable
from typing import Final, Literal

import pytest
from integration._support.client import Gateway
from integration._support.wire import wire_server
from integration.providers._cache_control_marks_support import (
    ASK,
    CITIES,
    EPHEMERAL,
    SYSTEM,
    TOOL,
    Mark,
    call_id,
    chat_body,
    final_text,
    new_marker,
    post_chat,
    tool_call,
)
from integration.providers._claude_prompt_cache_breakpoint_support import (
    ANTHROPIC_BODY_PROVIDERS,
    PROVIDERS,
    Provider,
    deployment,
    only_request,
    respond,
    sent_marks,
)
from pydantic import JsonValue

Role = Literal["system", "user", "assistant"]
Shape = Literal["list_content", "developer"]

_EXPLICIT: Final[dict[str, JsonValue]] = {"mode": "explicit"}
_ONE_HOUR_BREAKPOINT: Final[dict[str, JsonValue]] = {"mode": "explicit", "ttl": "1h"}
_THIRTY_MINUTES: Final[dict[str, JsonValue]] = {"mode": "explicit", "ttl": "30m"}
_ONE_HOUR: Final[dict[str, JsonValue]] = {"type": "ephemeral", "ttl": "1h"}
_LONDON: Final = CITIES[0]
_PARIS: Final = CITIES[1]
_FIRST_PART: Final = "first part"
_LAST_PART: Final = "last part"
_CLOUDY: Final = "cloudy"
_SUNNY: Final = "sunny"
_CHECKING: Final = "Checking."
_NOTES: Final = ("note 0", "note 1", "note 2")
_MALFORMED: Final[tuple[JsonValue, ...]] = (5, [], "", "x" * 5000)
_MALFORMED_IDS: Final = ("int", "list", "empty_string", "5kb_string")
_SEARCH_TOOLS: Final[list[JsonValue]] = [
    {"type": "tool_search", "execution": "server"},
    {**TOOL, "defer_loading": True},
]

Body = Callable[[str], dict[str, JsonValue]]


def _text(text: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {"type": "text", "text": text, **fields}


def _label(text: str) -> str:
    return f"text:{text}"


def _result_label(city: str) -> str:
    return f"result:{call_id(city)}"


def _call_label(city: str) -> str:
    return f"call:{call_id(city)}"


def _note_labels() -> list[str]:
    return [_label(note) for note in _NOTES]


def _final(marker: str, **fields: JsonValue) -> dict[str, JsonValue]:
    if not fields:
        return {"role": "user", "content": final_text(marker)}
    return {"role": "user", "content": [_text(final_text(marker), **fields)]}


def _marks_sent(gateway: Gateway, provider: Provider, body: Body, **deployment_fields: JsonValue) -> tuple[Mark, ...]:
    with wire_server(lambda request: respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = deployment(gateway, scenario, wire, provider, **deployment_fields)
        status, _, text = post_chat(gateway, body(model))
        assert status == 200, text
        request: Final = only_request(wire)
    assert b"prompt_cache_breakpoint" not in request.body, request.body
    return sent_marks(provider, request)


def _rejected(gateway: Gateway, provider: Provider, body: Body) -> str:
    with wire_server(lambda request: respond(provider, request)) as wire, gateway.scenario() as scenario:
        model: Final = deployment(gateway, scenario, wire, provider)
        status, _, text = post_chat(gateway, body(model))
        received: Final = wire.drain()
    assert status == 400, text
    assert "prompt_cache_breakpoint" in text and "drop_params" in text, text
    assert received == (), [request.target for request in received]
    return text


def _list_content_turns(role: Role, marker: str, mark: JsonValue, last: dict[str, JsonValue]) -> list[JsonValue]:
    marked: Final[dict[str, JsonValue]] = {
        "role": role,
        "content": [_text(_FIRST_PART), {**_text(_LAST_PART), **last}],
        "prompt_cache_breakpoint": mark,
    }
    match role:
        case "system":
            return [marked, _final(marker)]
        case "user":
            return [
                {"role": "system", "content": SYSTEM},
                marked,
                {"role": "assistant", "content": "noted"},
                _final(marker),
            ]
        case "assistant":
            return [{"role": "user", "content": ASK}, marked, _final(marker)]


def _developer_turns(marker: str, mark: JsonValue) -> list[JsonValue]:
    return [{"role": "developer", "content": SYSTEM, "prompt_cache_breakpoint": mark}, _final(marker)]


def _shape_turns(shape: Shape, marker: str, mark: JsonValue) -> list[JsonValue]:
    if shape == "developer":
        return _developer_turns(marker, mark)
    return _list_content_turns("user", marker, mark, {})


def _tool_result_turns(marker: str, result: dict[str, JsonValue], *, leading: bool = False) -> list[JsonValue]:
    lead: Final[list[JsonValue]] = [
        {"role": "system", "content": [_text(SYSTEM, prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_text(ASK, prompt_cache_breakpoint=_EXPLICIT)]},
    ]
    return [
        *(lead if leading else [{"role": "user", "content": ASK}]),
        {"role": "assistant", "content": None, "tool_calls": [tool_call(_LONDON)]},
        {"role": "tool", "tool_call_id": call_id(_LONDON), **result},
        _final(marker, prompt_cache_breakpoint=_EXPLICIT) if leading else _final(marker),
    ]


def _tool_call_turns(marker: str, **assistant_fields: JsonValue) -> list[JsonValue]:
    return [
        {"role": "user", "content": ASK},
        {"role": "assistant", "tool_calls": [tool_call(_LONDON), tool_call(_PARIS)], **assistant_fields},
        {"role": "tool", "tool_call_id": call_id(_LONDON), "content": "sunny."},
        {"role": "tool", "tool_call_id": call_id(_PARIS), "content": "sunny."},
        _final(marker),
    ]


def _noted_turns(marker: str, *history: JsonValue) -> list[JsonValue]:
    return [
        {"role": "system", "content": [_text(SYSTEM, prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "user", "content": [_text(note, prompt_cache_breakpoint=_EXPLICIT) for note in _NOTES]},
        *history,
        _final(marker),
    ]


@pytest.mark.parametrize("role", ("system", "user", "assistant"))
@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_message_breakpoint_on_list_content_marks_its_last_part(
    gateway: Gateway, provider: Provider, role: Role
) -> None:
    marker: Final = new_marker()
    marks: Final = _marks_sent(
        gateway, provider, lambda model: chat_body(model, _list_content_turns(role, marker, _EXPLICIT, {}))
    )
    assert marks == (Mark(_label(_LAST_PART), None),)


@pytest.mark.parametrize("content_fields", ({"content": None}, {}), ids=("null_content", "missing_content"))
@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_message_breakpoint_on_a_tool_call_turn_without_content_marks_its_last_call(
    gateway: Gateway, provider: Provider, content_fields: dict[str, JsonValue]
) -> None:
    marker: Final = new_marker()
    turns: Final = _tool_call_turns(marker, **content_fields, prompt_cache_breakpoint=_EXPLICIT)
    marks: Final = _marks_sent(gateway, provider, lambda model: chat_body(model, turns))
    assert marks == (Mark(_call_label(_PARIS), None),)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_message_breakpoint_on_an_empty_string_tool_call_turn_lands_where_a_client_cache_control_would(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    breakpoint_turns: Final = _tool_call_turns(marker, content="", prompt_cache_breakpoint=_EXPLICIT)
    cache_control_turns: Final = _tool_call_turns(marker, content="", cache_control=EPHEMERAL)
    breakpoint_marks: Final = _marks_sent(gateway, provider, lambda model: chat_body(model, breakpoint_turns))
    cache_control_marks: Final = _marks_sent(gateway, provider, lambda model: chat_body(model, cache_control_turns))
    assert breakpoint_marks == cache_control_marks


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_developer_message_breakpoint_marks_its_text(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    marks: Final = _marks_sent(gateway, provider, lambda model: chat_body(model, _developer_turns(marker, _EXPLICIT)))
    assert marks == (Mark(_label(SYSTEM), None),)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_tool_results_own_breakpoint_replaces_its_part_marks_and_joins_their_one_hour_ttl(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    result: Final[dict[str, JsonValue]] = {
        "content": [_text(_CLOUDY), _text(_SUNNY, prompt_cache_breakpoint=_ONE_HOUR_BREAKPOINT)],
        "prompt_cache_breakpoint": _EXPLICIT,
    }
    marks: Final = _marks_sent(gateway, provider, lambda model: chat_body(model, _tool_result_turns(marker, result)))
    assert marks == (Mark(_result_label(_LONDON), "1h"),)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_client_cache_control_on_a_tool_result_wins_over_its_part_breakpoints(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    result: Final[dict[str, JsonValue]] = {
        "content": [_text(_CLOUDY), _text(_SUNNY, prompt_cache_breakpoint=_ONE_HOUR_BREAKPOINT)],
        "cache_control": EPHEMERAL,
    }
    marks: Final = _marks_sent(gateway, provider, lambda model: chat_body(model, _tool_result_turns(marker, result)))
    assert marks == (Mark(_result_label(_LONDON), None),)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_client_cache_control_on_a_tool_result_part_wins_over_the_results_own_breakpoint(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    result: Final[dict[str, JsonValue]] = {
        "content": [_text(_CLOUDY), _text(_SUNNY, cache_control=EPHEMERAL)],
        "prompt_cache_breakpoint": _EXPLICIT,
    }
    marks: Final = _marks_sent(gateway, provider, lambda model: chat_body(model, _tool_result_turns(marker, result)))
    sent_on: Final = _result_label(_LONDON) if provider == "bedrock_converse" else _label(_SUNNY)
    assert marks == (Mark(sent_on, None),)


@pytest.mark.parametrize("client_marked", (False, True), ids=("breakpoints", "client_marked_part"))
@pytest.mark.parametrize("provider", PROVIDERS)
def test_part_marks_on_one_tool_result_take_one_slot_where_the_route_sends_one_mark_under_drop_params(
    gateway: Gateway, provider: Provider, client_marked: bool
) -> None:
    marker: Final = new_marker()
    first: Final = (
        _text(_CLOUDY, cache_control=EPHEMERAL) if client_marked else _text(_CLOUDY, prompt_cache_breakpoint=_EXPLICIT)
    )
    result: Final[dict[str, JsonValue]] = {"content": [first, _text(_SUNNY, prompt_cache_breakpoint=_EXPLICIT)]}
    marks: Final = _marks_sent(
        gateway,
        provider,
        lambda model: chat_body(model, _tool_result_turns(marker, result, leading=True)),
        drop_params=True,
    )
    one_mark_per_result: Final = [_label(SYSTEM), _label(ASK), _result_label(_LONDON), _label(final_text(marker))]
    mark_per_part: Final = [_label(ASK), _label(_CLOUDY), _label(_SUNNY), _label(final_text(marker))]
    assert [mark.label for mark in marks] == (one_mark_per_result if provider == "bedrock_converse" else mark_per_part)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_breakpoint_on_an_unanswered_tool_search_call_takes_no_slot_under_drop_params(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    search_call: Final[dict[str, JsonValue]] = {
        "id": f"srvtoolu_{marker[:24]}",
        "type": "function",
        "function": {"name": "tool_search_tool_regex", "arguments": json.dumps({"query": "weather"})},
        "prompt_cache_breakpoint": _EXPLICIT,
    }
    turns: Final = _noted_turns(marker, {"role": "assistant", "content": None, "tool_calls": [search_call]})
    marks: Final = _marks_sent(
        gateway,
        provider,
        lambda model: {**chat_body(model, turns, drop_params=True), "tools": _SEARCH_TOOLS},
    )
    assert [mark.label for mark in marks] == [_label(SYSTEM), *_note_labels()]


def test_breakpoints_on_tool_history_converse_sends_as_text_take_no_slot_under_drop_params(gateway: Gateway) -> None:
    marker: Final = new_marker()
    turns: Final = _noted_turns(
        marker,
        {"role": "assistant", "content": None, "tool_calls": [tool_call(_LONDON, prompt_cache_breakpoint=_EXPLICIT)]},
        {"role": "tool", "tool_call_id": call_id(_LONDON), "content": "sunny.", "prompt_cache_breakpoint": _EXPLICIT},
    )
    marks: Final = _marks_sent(
        gateway,
        "bedrock_converse",
        lambda model: {"model": model, "messages": turns, "max_tokens": 64, "drop_params": True},
    )
    assert [mark.label for mark in marks] == [_label(SYSTEM), *_note_labels()]


def _function_marked_tool_body(model: str, marker: str) -> dict[str, JsonValue]:
    function: Final = TOOL["function"]
    assert isinstance(function, dict), TOOL
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "prompt_cache_breakpoint": _EXPLICIT,
        "function": {**function, "cache_control": EPHEMERAL},
    }
    return {**chat_body(model, _noted_turns(marker), drop_params=True), "tools": [tool]}


@pytest.mark.parametrize("provider", ANTHROPIC_BODY_PROVIDERS)
def test_a_tool_breakpoint_beside_a_function_cache_control_leaves_room_for_four_marks_under_drop_params(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    marks: Final = _marks_sent(gateway, provider, lambda model: _function_marked_tool_body(model, marker))
    labels: Final = [mark.label for mark in marks]
    assert len(labels) == 4 and labels[1:] == _note_labels(), labels


def test_a_tool_breakpoint_beside_a_function_cache_control_keeps_the_latest_breakpoints_on_converse_under_drop_params(
    gateway: Gateway,
) -> None:
    marker: Final = new_marker()
    marks: Final = _marks_sent(gateway, "bedrock_converse", lambda model: _function_marked_tool_body(model, marker))
    labels: Final = [mark.label for mark in marks]
    assert len(labels) <= 4 and labels[-3:] == _note_labels(), labels


@pytest.mark.parametrize("provider", PROVIDERS)
def test_a_function_breakpoint_the_tool_cache_control_overrides_is_never_reported(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    function: Final = TOOL["function"]
    assert isinstance(function, dict), TOOL
    tool: Final[dict[str, JsonValue]] = {
        "type": "function",
        "cache_control": EPHEMERAL,
        "function": {**function, "prompt_cache_breakpoint": _THIRTY_MINUTES},
    }
    marks: Final = _marks_sent(gateway, provider, lambda model: {**chat_body(model, [_final(marker)]), "tools": [tool]})
    assert marks == (Mark("tool:lookup_weather", None),)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_assistant_text_read_before_a_one_hour_tool_call_mark_is_sent_as_one_hour(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    turns: Final[list[JsonValue]] = [
        {"role": "user", "content": ASK},
        {
            "role": "assistant",
            "content": _CHECKING,
            "prompt_cache_breakpoint": _EXPLICIT,
            "tool_calls": [tool_call(_LONDON, prompt_cache_breakpoint=_ONE_HOUR_BREAKPOINT)],
        },
        {"role": "tool", "tool_call_id": call_id(_LONDON), "content": "sunny."},
        _final(marker),
    ]
    marks: Final = _marks_sent(gateway, provider, lambda model: chat_body(model, turns))
    assert marks == (Mark(_label(_CHECKING), "1h"), Mark(_call_label(_LONDON), "1h"))


@pytest.mark.parametrize("value", _MALFORMED, ids=_MALFORMED_IDS)
@pytest.mark.parametrize("shape", ("list_content", "developer"))
def test_a_malformed_message_breakpoint_is_a_400_before_any_provider_call(
    gateway: Gateway, shape: Shape, value: JsonValue
) -> None:
    marker: Final = new_marker()
    _rejected(gateway, "anthropic", lambda model: chat_body(model, _shape_turns(shape, marker, value)))


@pytest.mark.parametrize("value", _MALFORMED, ids=_MALFORMED_IDS)
@pytest.mark.parametrize("shape", ("list_content", "developer"))
def test_a_malformed_message_breakpoint_sends_no_mark_under_drop_params(
    gateway: Gateway, shape: Shape, value: JsonValue
) -> None:
    marker: Final = new_marker()
    marks: Final = _marks_sent(
        gateway, "anthropic", lambda model: chat_body(model, _shape_turns(shape, marker, value), drop_params=True)
    )
    assert marks == ()


@pytest.mark.parametrize(
    ("provider", "result"),
    (
        (
            "anthropic",
            {
                "content": [_text(_CLOUDY), _text(_SUNNY, prompt_cache_breakpoint=_THIRTY_MINUTES)],
                "prompt_cache_breakpoint": _EXPLICIT,
            },
        ),
        (
            "bedrock_converse",
            {
                "content": [
                    _text(_CLOUDY, cache_control=EPHEMERAL),
                    _text(_SUNNY, prompt_cache_breakpoint=_THIRTY_MINUTES),
                ]
            },
        ),
    ),
    ids=("replaced_by_the_results_own_breakpoint", "replaced_by_a_client_marked_part"),
)
def test_a_replaced_part_breakpoint_with_a_ttl_claude_lacks_is_still_a_400(
    gateway: Gateway, provider: Provider, result: dict[str, JsonValue]
) -> None:
    marker: Final = new_marker()
    text: Final = _rejected(gateway, provider, lambda model: chat_body(model, _tool_result_turns(marker, result)))
    assert "30m" in text, text


@pytest.mark.parametrize(
    ("first", "last", "status", "labels"),
    (
        ({"mode": "implicit"}, _EXPLICIT, 200, [_label(_LAST_PART)]),
        (_EXPLICIT, {"mode": "implicit"}, 400, []),
    ),
    ids=("last_explicit", "last_implicit"),
)
def test_a_message_breakpoint_sent_twice_takes_the_last_value(
    gateway: Gateway, first: JsonValue, last: JsonValue, status: int, labels: list[str]
) -> None:
    marker: Final = new_marker()
    with wire_server(lambda request: respond("anthropic", request)) as wire, gateway.scenario() as scenario:
        model: Final = deployment(gateway, scenario, wire, "anthropic")
        turns: Final = _list_content_turns("user", marker, "duplicated", {})
        raw: Final = json.dumps(chat_body(model, turns)).replace(
            '"prompt_cache_breakpoint": "duplicated"',
            f'"prompt_cache_breakpoint": {json.dumps(first)}, "prompt_cache_breakpoint": {json.dumps(last)}',
        )
        response: Final = gateway.client.post(
            "/v1/chat/completions",
            content=raw,
            headers={"Authorization": f"Bearer {gateway.key}", "Content-Type": "application/json"},
        )
        received: Final = wire.drain()
    assert response.status_code == status, response.text
    assert [[mark.label for mark in sent_marks("anthropic", request)] for request in received] == (
        [labels] if status == 200 else []
    )


@pytest.mark.parametrize("provider", ("anthropic", "bedrock_converse"))
def test_a_null_message_breakpoint_on_list_content_is_no_mark(gateway: Gateway, provider: Provider) -> None:
    marker: Final = new_marker()
    marks: Final = _marks_sent(
        gateway, provider, lambda model: chat_body(model, _list_content_turns("user", marker, None, {}))
    )
    assert marks == ()


@pytest.mark.parametrize("provider", ("anthropic", "bedrock_converse"))
def test_a_client_cache_control_on_the_last_part_wins_over_the_message_breakpoint(
    gateway: Gateway, provider: Provider
) -> None:
    marker: Final = new_marker()
    last: Final[dict[str, JsonValue]] = {"cache_control": _ONE_HOUR}
    marks: Final = _marks_sent(
        gateway, provider, lambda model: chat_body(model, _list_content_turns("user", marker, _EXPLICIT, last))
    )
    assert marks == (Mark(_label(_LAST_PART), "1h"),)


@pytest.mark.parametrize(
    ("message_mark", "part_mark"),
    ((_EXPLICIT, _ONE_HOUR_BREAKPOINT), (_ONE_HOUR_BREAKPOINT, _EXPLICIT)),
    ids=("one_hour_on_the_part", "one_hour_on_the_message"),
)
@pytest.mark.parametrize("provider", ("anthropic", "bedrock_converse"))
def test_a_message_breakpoint_and_its_last_parts_own_breakpoint_join_into_one_mark(
    gateway: Gateway, provider: Provider, message_mark: JsonValue, part_mark: JsonValue
) -> None:
    marker: Final = new_marker()
    last: Final[dict[str, JsonValue]] = {"prompt_cache_breakpoint": part_mark}
    marks: Final = _marks_sent(
        gateway, provider, lambda model: chat_body(model, _list_content_turns("user", marker, message_mark, last))
    )
    assert marks == (Mark(_label(_LAST_PART), "1h"),)

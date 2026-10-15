import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Request, wire_server
from integration.providers._cache_control_marks_support import (
    ASK,
    ASK_LABEL,
    EPHEMERAL,
    POINTS,
    SYSTEM,
    SYSTEM_LABEL,
    anthropic_labels,
    anthropic_peer,
    ask,
    call_id,
    final_text,
    marker_of,
    post_chat,
    tool_call,
    tool_use_label,
)
from integration.providers._tool_search_support import (
    KEPT_FIELDS,
    Event,
    MODEL,
    SEARCH_CALL,
    SEARCH_ID,
    SEARCH_TURN,
    SEARCH_USE,
    SEARCH_RESULT,
    WEATHER_CALL,
    WEATHER_ID,
    answer,
    answer_text,
    answered_ids,
    assistant_from,
    chat,
    chat_body,
    deployment,
    first_messages,
    hand_built_assistant,
    replay_messages,
    scripted_peer,
    wire_assistant,
)
from pydantic import JsonValue

pytestmark = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_LANE_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_CITIES: Final = ("London", "Paris")
_SYSTEM_NOTE: Final = "Answer in one sentence."


def _modify_params_config(directory: Path) -> Path:
    lane: Final = object_value(yaml.safe_load(_LANE_CONFIG.read_text()))
    settings: Final = object_value(lane["litellm_settings"])
    path: Final = directory / "proxy_config_modify_params.yaml"
    path.write_text(yaml.safe_dump({**lane, "litellm_settings": {**settings, "modify_params": True}}, sort_keys=False))
    return path


@pytest.fixture(scope="module")
def modify_params_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    with gateway_from_environment() as lane:
        directory: Final = tmp_path_factory.mktemp("modify_params")
        with owned_proxy(lane, directory, {}, config=_modify_params_config(directory)) as owned:
            yield owned


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_modify_params_round_trip_keeps_the_search_result_and_no_placeholder_answer(
    modify_params_gateway: Gateway, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(scripted_peer()) as wire, modify_params_gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, modify_params_gateway.upstream_url)
        first: Final = chat(modify_params_gateway, chat_body(model, first_messages(marker), stream))
        assert first.status_code == 200, first.text
        second: Final = chat(
            modify_params_gateway, chat_body(model, replay_messages(marker, assistant_from(first, stream)), stream)
        )
        assert second.status_code == 200, second.text
        assert answer_text(second, stream) == answer(marker), second.text
        received: Final = wire.drain()
    assert len(received) == 2, received
    assert wire_assistant(received[1]) == list(SEARCH_TURN), received[1].body
    assert answered_ids(received[1]) == [WEATHER_ID], received[1].body


def test_modify_params_placeholder_for_the_kept_search_is_dropped_when_no_answer_was_sent(
    modify_params_gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    assistant: Final = hand_built_assistant((SEARCH_CALL, WEATHER_CALL), KEPT_FIELDS)
    with wire_server(scripted_peer()) as wire, modify_params_gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, modify_params_gateway.upstream_url)
        response: Final = chat(
            modify_params_gateway,
            chat_body(model, replay_messages(marker, assistant, ({"role": "user", "content": "go on"},)), False),
        )
        received: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    assert wire_assistant(received[0])[:2] == [SEARCH_USE, SEARCH_RESULT], received[0].body
    assert SEARCH_ID not in answered_ids(received[0]), received[0].body


def _marked_search_call(search_id: str) -> dict[str, JsonValue]:
    return {
        "id": search_id,
        "type": "function",
        "function": {"name": "tool_search_tool_bm25", "arguments": '{"query": "weather"}'},
        "cache_control": EPHEMERAL,
    }


def _marked_search_turn() -> dict[str, JsonValue]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [_marked_search_call(SEARCH_ID)],
        "provider_specific_fields": KEPT_FIELDS,
    }


def _trailing_search_conversation(marker: str) -> list[JsonValue]:
    return [
        {"role": "system", "content": SYSTEM},
        ask(marked=True),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call(city, cache_control=EPHEMERAL) for city in _CITIES],
        },
        *({"role": "tool", "tool_call_id": call_id(city), "content": "sunny."} for city in _CITIES),
        {"role": "user", "content": final_text(marker)},
        _marked_search_turn(),
    ]


def test_modify_params_trailing_search_turn_stays_within_the_cache_mark_cap(modify_params_gateway: Gateway) -> None:
    messages: Final = _trailing_search_conversation(uuid.uuid4().hex)
    with wire_server(anthropic_peer) as wire, modify_params_gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, modify_params_gateway.upstream_url)
        status, _, text = post_chat(
            modify_params_gateway, chat_body(model, messages, stream=False, cache_control_injection_points=POINTS)
        )
        received: Final = wire.drain()
    assert status == 200, text
    assert len(received) == 1, received
    labels: Final = anthropic_labels(received[0])
    assert labels == [ASK_LABEL, *(tool_use_label(city) for city in _CITIES), f"assistant:tool_use:{SEARCH_ID}"], labels
    assert SEARCH_ID in answered_ids(received[0]), received[0].body


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
def test_modify_params_trailing_search_turn_before_a_system_message_keeps_the_system_mark_within_the_cap(
    modify_params_gateway: Gateway, stream: bool
) -> None:
    messages: Final = [*_trailing_search_conversation(uuid.uuid4().hex), {"role": "system", "content": _SYSTEM_NOTE}]
    with wire_server(anthropic_peer) as wire, modify_params_gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, modify_params_gateway.upstream_url)
        response: Final = chat(
            modify_params_gateway, chat_body(model, messages, stream, cache_control_injection_points=POINTS)
        )
        received: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert answer_text(response, stream) == "sunny", response.text
    assert len(received) == 1, received
    labels: Final = anthropic_labels(received[0])
    assert labels == [SYSTEM_LABEL, ASK_LABEL, *(tool_use_label(city) for city in _CITIES)], labels
    assert wire_assistant(received[0], 3)[-2:] == [SEARCH_USE, SEARCH_RESULT], received[0].body
    assert SEARCH_ID not in answered_ids(received[0]), received[0].body


_SECOND_SEARCH_ID: Final = "srvtoolu_02ScriptedSearch"


@pytest.mark.parametrize(
    ("model", "expected_labels"),
    [
        pytest.param(
            f"anthropic/{MODEL}",
            [SYSTEM_LABEL, ASK_LABEL, f"assistant:tool_use:{SEARCH_ID}", f"assistant:tool_use:{_SECOND_SEARCH_ID}"],
            id="keeps-mid-conversation-system",
        ),
        pytest.param(
            "anthropic/claude-sonnet-4-6",
            [SYSTEM_LABEL, ASK_LABEL, f"user:text:{_SYSTEM_NOTE}"],
            id="converts-mid-conversation-system",
        ),
    ],
)
def test_modify_params_client_answer_to_one_of_two_marked_search_calls_fills_the_cache_mark_budget(
    modify_params_gateway: Gateway, model: str, expected_labels: list[str]
) -> None:
    second_result: Final[Event] = {**SEARCH_RESULT, "tool_use_id": _SECOND_SEARCH_ID}
    messages: Final[list[JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        ask(marked=True),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_marked_search_call(SEARCH_ID), _marked_search_call(_SECOND_SEARCH_ID)],
            "provider_specific_fields": {"tool_results": [SEARCH_RESULT, second_result]},
        },
        {"role": "tool", "tool_call_id": SEARCH_ID, "content": "found get_weather"},
        {"role": "system", "content": _SYSTEM_NOTE},
    ]
    with wire_server(anthropic_peer) as wire, modify_params_gateway.scenario() as scenario:
        deployed: Final = scenario.model(model=model, api_base=wire.url, api_key="scripted-anthropic-key")
        status, _, text = post_chat(
            modify_params_gateway, chat_body(deployed, messages, stream=False, cache_control_injection_points=POINTS)
        )
        received: Final = wire.drain()
    assert status == 200, text
    assert len(received) == 1, received
    assert anthropic_labels(received[0]) == expected_labels, received[0].body


_STRAY_ANSWER: Final = "cloudy."
_STRAY_TOOL_ANSWERS: Final = (
    pytest.param(call_id("London"), False, id="answers-an-earlier-turn"),
    pytest.param("call_never_issued", False, id="unknown-id"),
    pytest.param(7, False, id="integer-id"),
    pytest.param(f"call_{'x' * 5120}", False, id="five-kilobyte-id"),
    pytest.param(call_id("Paris"), False, id="answers-the-latest-turn-twice"),
    pytest.param("", True, id="empty-id"),
)


def _two_tool_turns(marker: str, stray_id: JsonValue) -> list[JsonValue]:
    return [
        {"role": "system", "content": SYSTEM},
        ask(marked=False),
        {"role": "assistant", "content": "", "tool_calls": [tool_call("London")]},
        {"role": "tool", "tool_call_id": call_id("London"), "content": "sunny."},
        {"role": "user", "content": "And Paris?"},
        {"role": "assistant", "content": "", "tool_calls": [tool_call("Paris")]},
        {"role": "tool", "tool_call_id": stray_id, "content": _STRAY_ANSWER},
        {"role": "tool", "tool_call_id": call_id("Paris"), "content": "sunny."},
        {"role": "user", "content": final_text(marker)},
    ]


def _function_call(city: str) -> dict[str, JsonValue]:
    return {
        "type": "function_call",
        "call_id": call_id(city),
        "name": "lookup_weather",
        "arguments": json.dumps({"city": city}),
    }


def _two_tool_turns_as_responses_input(marker: str, stray_id: JsonValue) -> list[JsonValue]:
    return [
        {"role": "user", "content": ASK},
        _function_call("London"),
        {"type": "function_call_output", "call_id": call_id("London"), "output": "sunny."},
        {"role": "user", "content": "And Paris?"},
        _function_call("Paris"),
        {"type": "function_call_output", "call_id": stray_id, "output": _STRAY_ANSWER},
        {"type": "function_call_output", "call_id": call_id("Paris"), "output": "sunny."},
        {"role": "user", "content": final_text(marker)},
    ]


def _assert_only_the_latest_turn_answers_reach_the_upstream(request: Request, marker: str, stray_kept: bool) -> None:
    ids: Final = answered_ids(request)
    assert marker_of(request) == marker, request.body
    assert ids.count(call_id("London")) == 1, request.body
    assert ids.count(call_id("Paris")) == 1, request.body
    assert len(ids) == 2 + stray_kept, request.body
    assert (_STRAY_ANSWER.encode() in request.body) is stray_kept, request.body


@pytest.mark.parametrize("stream", [False, True], ids=["non-stream", "stream"])
@pytest.mark.parametrize(("stray_id", "stray_kept"), _STRAY_TOOL_ANSWERS)
def test_modify_params_drops_a_tool_answer_that_answers_no_call_of_the_latest_assistant_turn(
    modify_params_gateway: Gateway, stray_id: JsonValue, stray_kept: bool, stream: bool
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, modify_params_gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, modify_params_gateway.upstream_url)
        response: Final = chat(modify_params_gateway, chat_body(model, _two_tool_turns(marker, stray_id), stream))
        received: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    _assert_only_the_latest_turn_answers_reach_the_upstream(received[0], marker, stray_kept)


def test_modify_params_responses_input_drops_an_output_for_an_earlier_function_call(
    modify_params_gateway: Gateway,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(anthropic_peer) as wire, modify_params_gateway.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, modify_params_gateway.upstream_url)
        response: Final = modify_params_gateway.request(
            "POST",
            "/v1/responses",
            {
                "model": model,
                "max_output_tokens": 64,
                "tools": [{"type": "function", "name": "lookup_weather", "parameters": {"type": "object"}}],
                "input": _two_tool_turns_as_responses_input(marker, call_id("London")),
            },
        )
        received: Final = wire.drain()
    assert response.status_code == 200, response.text
    assert len(received) == 1, received
    _assert_only_the_latest_turn_answers_reach_the_upstream(received[0], marker, False)


def _with_tokyo_call(identity: JsonValue) -> dict[str, JsonValue]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [tool_call("Tokyo", cache_control=EPHEMERAL, id=identity)],
    }


_MALFORMED_MESSAGES: Final = (
    pytest.param(_with_tokyo_call({"id": call_id("Tokyo")}), id="tool-call-id-object"),
    pytest.param(_with_tokyo_call([call_id("Tokyo")]), id="tool-call-id-list"),
)


@pytest.mark.parametrize("malformed", _MALFORMED_MESSAGES)
@pytest.mark.parametrize("modify_params", [False, True], ids=["modify-params-off", "modify-params-on"])
def test_marked_search_call_next_to_a_malformed_tool_id_and_a_later_system_message_answers_400(
    gateway: Gateway,
    modify_params_gateway: Gateway,
    modify_params: bool,
    malformed: dict[str, JsonValue],
) -> None:
    proxy: Final = modify_params_gateway if modify_params else gateway
    marker: Final = uuid.uuid4().hex
    messages: Final[list[JsonValue]] = [
        {"role": "system", "content": SYSTEM},
        ask(marked=False),
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call(city, cache_control=EPHEMERAL) for city in _CITIES],
        },
        *({"role": "tool", "tool_call_id": call_id(city), "content": "sunny."} for city in _CITIES),
        _marked_search_turn(),
        malformed,
        {"role": "user", "content": final_text(marker)},
        {"role": "system", "content": _SYSTEM_NOTE},
    ]
    with wire_server(anthropic_peer) as wire, proxy.scenario() as scenario:
        model: Final = deployment(scenario, "anthropic", wire.url, proxy.upstream_url)
        status, _, text = post_chat(
            proxy, chat_body(model, messages, stream=False, cache_control_injection_points=POINTS)
        )
        received: Final = wire.drain()
    assert status == 400, text
    assert '"error"' in text, text
    assert received == (), [request.target for request in received]
    assert proxy.client.get("/health/liveliness").status_code == 200

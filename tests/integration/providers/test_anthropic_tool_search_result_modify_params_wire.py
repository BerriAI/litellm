import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import wire_server
from integration.providers._cache_control_marks_support import (
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
    post_chat,
    tool_call,
    tool_use_label,
)
from integration.providers._tool_search_support import (
    KEPT_FIELDS,
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


def _marked_search_turn() -> dict[str, JsonValue]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": SEARCH_ID,
                "type": "function",
                "function": {"name": "tool_search_tool_bm25", "arguments": '{"query": "weather"}'},
                "cache_control": EPHEMERAL,
            }
        ],
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

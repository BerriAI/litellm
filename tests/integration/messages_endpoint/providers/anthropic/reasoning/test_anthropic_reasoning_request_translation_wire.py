import json
import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_HAIKU_4_5: Final = "claude-haiku-4-5"
_OPUS_4_5: Final = "claude-opus-4-5"
_OPUS_4_6: Final = "claude-opus-4-6"
_OPUS_4_7: Final = "claude-opus-4-7"
_FABLE_5_1: Final = "claude-fable-5-1"
_ADAPTIVE: Final = {"type": "adaptive", "display": "omitted"}
_ADAPTIVE_SUMMARIZED: Final = {"type": "adaptive", "display": "summarized"}


def _budget(tokens: int) -> dict[str, JsonValue]:
    return {"type": "enabled", "budget_tokens": tokens}


def _client_body(**reasoning: JsonValue) -> dict[str, JsonValue]:
    base: Final = {
        key: value
        for key, value in cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}").items()
        if key != "thinking"
    }
    return {**base, "stream": False, **reasoning}


def _without(body: Mapping[str, JsonValue], *keys: str) -> dict[str, JsonValue]:
    return {key: value for key, value in body.items() if key not in keys}


def _diff(expected: Mapping[str, JsonValue], body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        key: {"expected": expected.get(key), "upstream": body.get(key)}
        for key in expected.keys() | body.keys()
        if expected.get(key) != body.get(key)
    }


def _forwarded_body(
    gateway: Gateway, upstream_model: str, client_body: Mapping[str, JsonValue]
) -> dict[str, JsonValue]:
    def respond(request: Request) -> Reply:
        return Reply(
            body=json.dumps(
                {
                    "id": f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "model": upstream_model,
                    "content": [{"type": "text", "text": "PONG"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 12, "output_tokens": 4},
                }
            ).encode()
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{upstream_model}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        response: Final = gateway.request("POST", "/v1/messages", {**client_body, "model": model})
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
        assert len(received) == 1, received
        return cc.JSON_OBJECT.validate_json(received[0].body)


def _assert_forwarded(
    gateway: Gateway,
    upstream_model: str,
    client_body: dict[str, JsonValue],
    expected_changes: Mapping[str, JsonValue],
    removed: tuple[str, ...],
) -> None:
    expected: Final = {**_without(client_body, *removed), **expected_changes, "model": upstream_model}
    body: Final = _forwarded_body(gateway, upstream_model, client_body)
    assert body == expected, _diff(expected, body)


@pytest.mark.parametrize(
    ("upstream_model", "effort", "expected_changes", "removed"),
    (
        pytest.param(
            _OPUS_4_5,
            "high",
            {},
            ("thinking",),
            id="opus-4.5-keeps-supported-effort-drops-adaptive",
        ),
        pytest.param(
            _OPUS_4_5,
            "xhigh",
            {"thinking": _budget(8192)},
            ("output_config",),
            id="opus-4.5-xhigh-falls-back-to-budget",
        ),
        pytest.param(_HAIKU_4_5, "low", {"thinking": _budget(1024)}, ("output_config",), id="haiku-4.5-low"),
        pytest.param(_HAIKU_4_5, "medium", {"thinking": _budget(2048)}, ("output_config",), id="haiku-4.5-medium"),
        pytest.param(_HAIKU_4_5, "high", {"thinking": _budget(4096)}, ("output_config",), id="haiku-4.5-high"),
        pytest.param(_HAIKU_4_5, "xhigh", {"thinking": _budget(8192)}, ("output_config",), id="haiku-4.5-xhigh"),
        pytest.param(_HAIKU_4_5, "max", {"thinking": _budget(16384)}, ("output_config",), id="haiku-4.5-max"),
        pytest.param(_OPUS_4_6, "high", {}, (), id="opus-4.6-adaptive-unchanged"),
        pytest.param(_OPUS_4_7, "xhigh", {}, (), id="opus-4.7-adaptive-unchanged"),
    ),
)
def test_adaptive_thinking_and_effort_are_reshaped_only_for_models_without_adaptive_thinking(
    gateway: Gateway,
    upstream_model: str,
    effort: str,
    expected_changes: dict[str, JsonValue],
    removed: tuple[str, ...],
) -> None:
    client_body: Final = _client_body(thinking=dict(_ADAPTIVE), output_config={"effort": effort})
    _assert_forwarded(gateway, upstream_model, client_body, expected_changes, removed)


def test_adaptive_effort_fallback_budget_is_capped_below_max_tokens(gateway: Gateway) -> None:
    client_body: Final = _client_body(thinking=dict(_ADAPTIVE), output_config={"effort": "max"}, max_tokens=4000)
    _assert_forwarded(gateway, _HAIKU_4_5, client_body, {"thinking": _budget(3999)}, ("output_config",))


def test_adaptive_effort_fallback_drops_thinking_when_max_tokens_cannot_fit_the_minimum_budget(
    gateway: Gateway,
) -> None:
    client_body: Final = _client_body(thinking=dict(_ADAPTIVE), output_config={"effort": "high"}, max_tokens=1024)
    _assert_forwarded(gateway, _HAIKU_4_5, client_body, {}, ("thinking", "output_config"))


@pytest.mark.parametrize(
    ("budget_tokens", "effort"),
    (
        pytest.param(1024, "low", id="below-medium-threshold"),
        pytest.param(2048, "medium", id="medium-threshold"),
        pytest.param(4096, "high", id="high-threshold"),
        pytest.param(8192, "xhigh", id="xhigh-threshold"),
    ),
)
def test_legacy_thinking_budget_becomes_adaptive_effort_on_models_that_reject_budgets(
    gateway: Gateway, budget_tokens: int, effort: str
) -> None:
    client_body: Final = _client_body(thinking=_budget(budget_tokens))
    _assert_forwarded(
        gateway,
        _OPUS_4_7,
        client_body,
        {"thinking": {"type": "adaptive"}, "output_config": {"effort": effort}},
        (),
    )


def test_legacy_thinking_translation_keeps_the_callers_effort(gateway: Gateway) -> None:
    client_body: Final = _client_body(thinking=_budget(8192), output_config={"effort": "medium"})
    _assert_forwarded(gateway, _OPUS_4_7, client_body, {"thinking": {"type": "adaptive"}}, ())


@pytest.mark.parametrize(
    ("upstream_model", "removed"),
    (
        pytest.param(_FABLE_5_1, ("thinking",), id="always-on-model-drops-disabled"),
        pytest.param(_OPUS_4_7, (), id="other-model-keeps-disabled"),
    ),
)
def test_disabled_thinking_is_dropped_only_for_always_on_thinking_models(
    gateway: Gateway, upstream_model: str, removed: tuple[str, ...]
) -> None:
    client_body: Final = _client_body(thinking={"type": "disabled"})
    _assert_forwarded(gateway, upstream_model, client_body, {}, removed)


@pytest.mark.parametrize(
    ("reasoning_effort", "effort"),
    (
        pytest.param("minimal", "low", id="minimal"),
        pytest.param("low", "low", id="low"),
        pytest.param("medium", "medium", id="medium"),
        pytest.param("high", "high", id="high"),
        pytest.param("xhigh", "xhigh", id="xhigh"),
        pytest.param("max", "max", id="max"),
    ),
)
def test_reasoning_effort_becomes_adaptive_thinking_and_effort_on_adaptive_models(
    gateway: Gateway, reasoning_effort: str, effort: str
) -> None:
    client_body: Final = _client_body(reasoning_effort=reasoning_effort)
    _assert_forwarded(
        gateway,
        _OPUS_4_7,
        client_body,
        {"thinking": dict(_ADAPTIVE_SUMMARIZED), "output_config": {"effort": effort}},
        ("reasoning_effort",),
    )


@pytest.mark.parametrize(
    ("reasoning_effort", "budget_tokens"),
    (
        pytest.param("minimal", 1024, id="minimal"),
        pytest.param("low", 1024, id="low"),
        pytest.param("medium", 2048, id="medium"),
        pytest.param("high", 4096, id="high"),
        pytest.param("xhigh", 8192, id="xhigh"),
        pytest.param("max", 16384, id="max"),
    ),
)
def test_reasoning_effort_becomes_a_thinking_budget_on_models_without_adaptive_thinking(
    gateway: Gateway, reasoning_effort: str, budget_tokens: int
) -> None:
    client_body: Final = _client_body(reasoning_effort=reasoning_effort)
    _assert_forwarded(gateway, _HAIKU_4_5, client_body, {"thinking": _budget(budget_tokens)}, ("reasoning_effort",))


def test_reasoning_effort_none_clears_thinking_and_effort(gateway: Gateway) -> None:
    client_body: Final = _client_body(
        reasoning_effort="none", thinking=dict(_ADAPTIVE), output_config={"effort": "high"}
    )
    _assert_forwarded(gateway, _OPUS_4_7, client_body, {}, ("reasoning_effort", "thinking", "output_config"))


def test_caller_thinking_wins_over_reasoning_effort(gateway: Gateway) -> None:
    client_body: Final = _client_body(reasoning_effort="high", thinking=_budget(2000))
    _assert_forwarded(gateway, _HAIKU_4_5, client_body, {}, ("reasoning_effort",))


def test_caller_effort_wins_over_reasoning_effort(gateway: Gateway) -> None:
    client_body: Final = _client_body(reasoning_effort="high", output_config={"effort": "low"})
    _assert_forwarded(gateway, _OPUS_4_7, client_body, {"thinking": dict(_ADAPTIVE_SUMMARIZED)}, ("reasoning_effort",))


def test_reasoning_effort_budget_is_capped_below_max_tokens(gateway: Gateway) -> None:
    client_body: Final = _client_body(reasoning_effort="max", max_tokens=4000)
    _assert_forwarded(gateway, _HAIKU_4_5, client_body, {"thinking": _budget(3999)}, ("reasoning_effort",))


def test_reasoning_effort_is_dropped_when_max_tokens_cannot_fit_the_minimum_budget(gateway: Gateway) -> None:
    client_body: Final = _client_body(reasoning_effort="high", max_tokens=1024)
    _assert_forwarded(gateway, _HAIKU_4_5, client_body, {}, ("reasoning_effort",))


@pytest.mark.parametrize(
    ("upstream_model", "reasoning", "expected_changes", "removed"),
    (
        pytest.param(
            _HAIKU_4_5,
            {"thinking": dict(_ADAPTIVE), "output_config": {"effort": "high"}},
            {"thinking": _budget(4096)},
            ("output_config", "temperature"),
            id="haiku-4.5-effort-translated-to-budget",
        ),
        pytest.param(
            _OPUS_4_5,
            {"thinking": dict(_ADAPTIVE), "output_config": {"effort": "high"}},
            {},
            ("thinking", "temperature"),
            id="opus-4.5-effort-kept",
        ),
        pytest.param(_HAIKU_4_5, {"thinking": _budget(2048)}, {}, ("temperature",), id="haiku-4.5-legacy-budget"),
    ),
)
def test_non_default_temperature_is_dropped_when_a_non_adaptive_model_thinks(
    gateway: Gateway,
    upstream_model: str,
    reasoning: dict[str, JsonValue],
    expected_changes: dict[str, JsonValue],
    removed: tuple[str, ...],
) -> None:
    client_body: Final = _client_body(temperature=0, **reasoning)
    _assert_forwarded(gateway, upstream_model, client_body, expected_changes, removed)


@pytest.mark.parametrize(
    ("upstream_model", "temperature", "reasoning"),
    (
        pytest.param(_HAIKU_4_5, 1, {"thinking": _budget(2048)}, id="temperature-1-with-thinking"),
        pytest.param(_HAIKU_4_5, 0, {}, id="temperature-0-without-thinking"),
        pytest.param(
            _OPUS_4_6,
            0,
            {"thinking": dict(_ADAPTIVE), "output_config": {"effort": "high"}},
            id="adaptive-model",
        ),
    ),
)
def test_temperature_is_kept_when_it_does_not_conflict_with_thinking(
    gateway: Gateway, upstream_model: str, temperature: int, reasoning: dict[str, JsonValue]
) -> None:
    client_body: Final = _client_body(temperature=temperature, **reasoning)
    _assert_forwarded(gateway, upstream_model, client_body, {}, ())


_SIGNED_THINKING: Final = {"type": "thinking", "thinking": "check the config first", "signature": "EqQBCkgIBRABGAIiQL"}
_TOOL_CALL: Final = {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"file_path": "/repo/config.yaml"}}
_TOOL_RESULT: Final = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}]}


def _history_body(assistant_content: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    base: Final = _client_body(thinking=_budget(2048))
    first_turn: Final = base["messages"]
    assert isinstance(first_turn, list)
    return {**base, "messages": [*first_turn, {"role": "assistant", "content": list(assistant_content)}, _TOOL_RESULT]}


def _with_assistant_content(
    body: Mapping[str, JsonValue], assistant_content: tuple[dict[str, JsonValue], ...]
) -> dict[str, JsonValue]:
    messages: Final = body["messages"]
    assert isinstance(messages, list)
    return {
        **body,
        "messages": [*messages[:-2], {"role": "assistant", "content": list(assistant_content)}, messages[-1]],
    }


def test_encrypted_reasoning_from_another_provider_is_stripped_and_anthropic_signed_thinking_is_kept(
    gateway: Gateway,
) -> None:
    client_body: Final = _history_body(
        (
            {"type": "thinking", "thinking": "bridge reasoning", "signature": "litellm_encrypted_reasoning:gAAAAB"},
            {"type": "redacted_thinking", "data": "litellm_encrypted_reasoning:gAAAAC"},
            _SIGNED_THINKING,
            _TOOL_CALL,
        )
    )
    expected: Final = {**_with_assistant_content(client_body, (_SIGNED_THINKING, _TOOL_CALL)), "model": _HAIKU_4_5}
    body: Final = _forwarded_body(gateway, _HAIKU_4_5, client_body)
    assert body == expected, _diff(expected, body)


def test_empty_thinking_block_is_stripped_and_redacted_thinking_is_kept(gateway: Gateway) -> None:
    redacted: Final = {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix"}
    client_body: Final = _history_body(
        ({"type": "thinking", "thinking": "", "signature": "EqQBCkgIBRABGAIiQM"}, redacted, _TOOL_CALL)
    )
    expected: Final = {**_with_assistant_content(client_body, (redacted, _TOOL_CALL)), "model": _HAIKU_4_5}
    body: Final = _forwarded_body(gateway, _HAIKU_4_5, client_body)
    assert body == expected, _diff(expected, body)


@pytest.mark.parametrize(
    ("upstream_model", "reasoning_effort"),
    (
        pytest.param(_HAIKU_4_5, "turbo", id="unknown-value"),
        pytest.param(_OPUS_4_6, "xhigh", id="level-the-model-lacks"),
    ),
)
def test_unsupported_reasoning_effort_is_rejected_before_reaching_anthropic(
    gateway: Gateway, upstream_model: str, reasoning_effort: str
) -> None:
    with wire_server(lambda request: Reply()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{upstream_model}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        response: Final = gateway.request(
            "POST", "/v1/messages", {**_client_body(reasoning_effort=reasoning_effort), "model": model}
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["type"] == "invalid_request_error", response.text
        assert wire.drain() == ()

import uuid
from collections.abc import Mapping
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue


def _claude_code_turn(sent: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    default_turn: Final = cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}")
    without_reasoning: Final = {key: value for key, value in default_turn.items() if key != "thinking"}
    return {**without_reasoning, "stream": False, **sent}


def _forward(gateway: Gateway, upstream_model: str, sent: Mapping[str, JsonValue]) -> cc.Forwarded:
    client_body: Final = _claude_code_turn(sent)

    def respond(request: Request) -> Reply:
        return Reply(
            body=cc.message_reply(
                f"msg_{uuid.uuid4().hex}",
                upstream_model,
                ({"type": "text", "text": "PONG"},),
                {"input_tokens": 12, "output_tokens": 4},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{upstream_model}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**client_body, "model": model},
            headers=cc.cli_headers(gateway.key, cc.FRONTIER_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        received: Final = wire.drain()
    assert len(received) == 1, received
    return cc.forwarded(client_body, received[0])


@pytest.mark.parametrize(
    ("upstream_model", "sent", "received"),
    (
        pytest.param(
            "claude-haiku-4-5",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "low"}},
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            id="haiku-4.5-low",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "medium"}},
            {"thinking": {"type": "enabled", "budget_tokens": 2048}},
            id="haiku-4.5-medium",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"}},
            {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            id="haiku-4.5-high",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "xhigh"}},
            {"thinking": {"type": "enabled", "budget_tokens": 8192}},
            id="haiku-4.5-xhigh",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "max"}},
            {"thinking": {"type": "enabled", "budget_tokens": 16384}},
            id="haiku-4.5-max",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "max"},
                "max_tokens": 4000,
            },
            {"thinking": {"type": "enabled", "budget_tokens": 3999}},
            id="haiku-4.5-budget-capped-below-max-tokens",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "high"},
                "max_tokens": 1024,
            },
            {},
            id="haiku-4.5-max-tokens-below-minimum-budget",
        ),
        pytest.param(
            "claude-opus-4-5",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"}},
            {"output_config": {"effort": "high"}},
            id="opus-4.5-keeps-effort-drops-adaptive",
        ),
        pytest.param(
            "claude-opus-4-5",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "xhigh"}},
            {"thinking": {"type": "enabled", "budget_tokens": 8192}},
            id="opus-4.5-xhigh-falls-back-to-budget",
        ),
        pytest.param(
            "claude-opus-4-6",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"}},
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"}},
            id="opus-4.6-unchanged",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "xhigh"}},
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "xhigh"}},
            id="opus-4.7-unchanged",
        ),
        pytest.param(
            "claude-fable-5-1",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"}},
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "high"}},
            id="fable-5.1-unchanged",
        ),
        pytest.param(
            "claude-opus-5-5",
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "xhigh"}},
            {"thinking": {"type": "adaptive", "display": "omitted"}, "output_config": {"effort": "xhigh"}},
            id="opus-5.5-unchanged",
        ),
    ),
)
def test_adaptive_thinking_and_effort_are_rewritten_only_for_models_without_adaptive_thinking(
    gateway: Gateway, upstream_model: str, sent: dict[str, JsonValue], received: dict[str, JsonValue]
) -> None:
    forwarded: Final = _forward(gateway, upstream_model, sent)
    assert forwarded.reasoning == received, forwarded
    assert forwarded.other_changes == {}, forwarded.other_changes
    assert forwarded.reasoning_betas == cc.CLAUDE_CODE_REASONING_BETAS, forwarded.reasoning_betas


@pytest.mark.parametrize(
    ("upstream_model", "sent", "received"),
    (
        pytest.param(
            "claude-opus-4-7",
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
            id="opus-4.7-1024-is-low",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"thinking": {"type": "enabled", "budget_tokens": 2048}},
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}},
            id="opus-4.7-2048-is-medium",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
            id="opus-4.7-4096-is-high",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"thinking": {"type": "enabled", "budget_tokens": 8192}},
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "xhigh"}},
            id="opus-4.7-8192-is-xhigh",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"thinking": {"type": "enabled", "budget_tokens": 8192}, "output_config": {"effort": "medium"}},
            {"thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}},
            id="opus-4.7-keeps-the-callers-effort",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"thinking": {"type": "enabled", "budget_tokens": 2048}},
            {"thinking": {"type": "enabled", "budget_tokens": 2048}},
            id="haiku-4.5-unchanged",
        ),
    ),
)
def test_legacy_thinking_budget_becomes_adaptive_effort_only_on_models_that_reject_budgets(
    gateway: Gateway, upstream_model: str, sent: dict[str, JsonValue], received: dict[str, JsonValue]
) -> None:
    forwarded: Final = _forward(gateway, upstream_model, sent)
    assert forwarded.reasoning == received, forwarded
    assert forwarded.other_changes == {}, forwarded.other_changes
    assert forwarded.reasoning_betas == cc.CLAUDE_CODE_REASONING_BETAS, forwarded.reasoning_betas


@pytest.mark.parametrize(
    ("upstream_model", "sent", "received"),
    (
        pytest.param(
            "claude-fable-5-1",
            {"thinking": {"type": "disabled"}},
            {},
            id="fable-5.1-always-thinks-so-disabled-is-dropped",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"thinking": {"type": "disabled"}},
            {"thinking": {"type": "disabled"}},
            id="opus-4.7-keeps-disabled",
        ),
    ),
)
def test_disabled_thinking_is_dropped_only_for_always_on_thinking_models(
    gateway: Gateway, upstream_model: str, sent: dict[str, JsonValue], received: dict[str, JsonValue]
) -> None:
    forwarded: Final = _forward(gateway, upstream_model, sent)
    assert forwarded.reasoning == received, forwarded
    assert forwarded.other_changes == {}, forwarded.other_changes
    assert forwarded.reasoning_betas == cc.CLAUDE_CODE_REASONING_BETAS, forwarded.reasoning_betas


@pytest.mark.parametrize(
    ("upstream_model", "sent", "received"),
    (
        pytest.param(
            "claude-opus-4-7",
            {"reasoning_effort": "minimal"},
            {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "low"}},
            id="opus-4.7-minimal",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"reasoning_effort": "low"},
            {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "low"}},
            id="opus-4.7-low",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"reasoning_effort": "medium"},
            {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "medium"}},
            id="opus-4.7-medium",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"reasoning_effort": "high"},
            {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "high"}},
            id="opus-4.7-high",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"reasoning_effort": "xhigh"},
            {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "xhigh"}},
            id="opus-4.7-xhigh",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"reasoning_effort": "max"},
            {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "max"}},
            id="opus-4.7-max",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"reasoning_effort": "minimal"},
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            id="haiku-4.5-minimal",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"reasoning_effort": "low"},
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            id="haiku-4.5-low",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"reasoning_effort": "medium"},
            {"thinking": {"type": "enabled", "budget_tokens": 2048}},
            id="haiku-4.5-medium",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"reasoning_effort": "high"},
            {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            id="haiku-4.5-high",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"reasoning_effort": "xhigh"},
            {"thinking": {"type": "enabled", "budget_tokens": 8192}},
            id="haiku-4.5-xhigh",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"reasoning_effort": "max"},
            {"thinking": {"type": "enabled", "budget_tokens": 16384}},
            id="haiku-4.5-max",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"reasoning_effort": "max", "max_tokens": 4000},
            {"thinking": {"type": "enabled", "budget_tokens": 3999}},
            id="haiku-4.5-budget-capped-below-max-tokens",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"reasoning_effort": "high", "max_tokens": 1024},
            {},
            id="haiku-4.5-max-tokens-below-minimum-budget",
        ),
        pytest.param(
            "claude-opus-4-7",
            {
                "reasoning_effort": "none",
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "high"},
            },
            {},
            id="none-clears-thinking-and-effort",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"reasoning_effort": "high", "thinking": {"type": "enabled", "budget_tokens": 2000}},
            {"thinking": {"type": "enabled", "budget_tokens": 2000}},
            id="callers-thinking-wins",
        ),
        pytest.param(
            "claude-opus-4-7",
            {"reasoning_effort": "high", "output_config": {"effort": "low"}},
            {"thinking": {"type": "adaptive", "display": "summarized"}, "output_config": {"effort": "low"}},
            id="callers-effort-wins",
        ),
    ),
)
def test_reasoning_effort_becomes_the_thinking_shape_each_model_accepts(
    gateway: Gateway, upstream_model: str, sent: dict[str, JsonValue], received: dict[str, JsonValue]
) -> None:
    forwarded: Final = _forward(gateway, upstream_model, sent)
    assert forwarded.reasoning == received, forwarded
    assert forwarded.other_changes == {}, forwarded.other_changes
    assert forwarded.reasoning_betas == cc.CLAUDE_CODE_REASONING_BETAS, forwarded.reasoning_betas


@pytest.mark.parametrize(
    ("upstream_model", "sent", "received"),
    (
        pytest.param(
            "claude-haiku-4-5",
            {
                "temperature": 0,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "high"},
            },
            {"thinking": {"type": "enabled", "budget_tokens": 4096}},
            id="haiku-4.5-drops-temperature-0-with-effort",
        ),
        pytest.param(
            "claude-opus-4-5",
            {
                "temperature": 0,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "high"},
            },
            {"output_config": {"effort": "high"}},
            id="opus-4.5-drops-temperature-0-with-effort",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"temperature": 0, "thinking": {"type": "enabled", "budget_tokens": 2048}},
            {"thinking": {"type": "enabled", "budget_tokens": 2048}},
            id="haiku-4.5-drops-temperature-0-with-budget",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"temperature": 1, "thinking": {"type": "enabled", "budget_tokens": 2048}},
            {"temperature": 1, "thinking": {"type": "enabled", "budget_tokens": 2048}},
            id="haiku-4.5-keeps-temperature-1",
        ),
        pytest.param(
            "claude-haiku-4-5",
            {"temperature": 0},
            {"temperature": 0},
            id="haiku-4.5-keeps-temperature-without-thinking",
        ),
        pytest.param(
            "claude-opus-4-6",
            {
                "temperature": 0,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "high"},
            },
            {
                "temperature": 0,
                "thinking": {"type": "adaptive", "display": "omitted"},
                "output_config": {"effort": "high"},
            },
            id="opus-4.6-adaptive-keeps-temperature",
        ),
    ),
)
def test_temperature_is_dropped_only_when_a_non_adaptive_model_thinks(
    gateway: Gateway, upstream_model: str, sent: dict[str, JsonValue], received: dict[str, JsonValue]
) -> None:
    forwarded: Final = _forward(gateway, upstream_model, sent)
    assert forwarded.reasoning == received, forwarded
    assert forwarded.other_changes == {}, forwarded.other_changes
    assert forwarded.reasoning_betas == cc.CLAUDE_CODE_REASONING_BETAS, forwarded.reasoning_betas


def _tool_loop(assistant_content: list[JsonValue]) -> dict[str, JsonValue]:
    first_turn: Final = cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}")["messages"]
    assert isinstance(first_turn, list)
    tool_result: Final = {"type": "tool_result", "tool_use_id": "toolu_01", "content": "ok"}
    return {
        "thinking": {"type": "enabled", "budget_tokens": 2048},
        "messages": [
            *first_turn,
            {"role": "assistant", "content": assistant_content},
            {"role": "user", "content": [tool_result]},
        ],
    }


@pytest.mark.parametrize(
    ("sent_history", "received_history"),
    (
        pytest.param(
            [
                {"type": "thinking", "thinking": "bridge reasoning", "signature": "litellm_encrypted_reasoning:gAAAAB"},
                {"type": "redacted_thinking", "data": "litellm_encrypted_reasoning:gAAAAC"},
                {"type": "thinking", "thinking": "check the config", "signature": "EqQBCkgIBRABGAIiQL"},
                {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"file_path": "/repo/config.yaml"}},
            ],
            [
                {"type": "thinking", "thinking": "check the config", "signature": "EqQBCkgIBRABGAIiQL"},
                {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"file_path": "/repo/config.yaml"}},
            ],
            id="encrypted-reasoning-from-another-provider-stripped-anthropic-signed-kept",
        ),
        pytest.param(
            [
                {"type": "thinking", "thinking": "", "signature": "EqQBCkgIBRABGAIiQM"},
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix"},
                {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"file_path": "/repo/config.yaml"}},
            ],
            [
                {"type": "redacted_thinking", "data": "EmwKAhgBEgy3va3pzix"},
                {"type": "tool_use", "id": "toolu_01", "name": "Read", "input": {"file_path": "/repo/config.yaml"}},
            ],
            id="empty-thinking-stripped-redacted-thinking-kept",
        ),
    ),
)
def test_thinking_history_keeps_only_blocks_anthropic_can_verify(
    gateway: Gateway, sent_history: list[JsonValue], received_history: list[JsonValue]
) -> None:
    forwarded: Final = _forward(gateway, "claude-haiku-4-5", _tool_loop(sent_history))
    assert forwarded.assistant_history == (received_history,), forwarded.assistant_history
    assert forwarded.reasoning == {"thinking": {"type": "enabled", "budget_tokens": 2048}}, forwarded
    assert forwarded.other_changes == {}, forwarded.other_changes
    assert forwarded.reasoning_betas == cc.CLAUDE_CODE_REASONING_BETAS, forwarded.reasoning_betas


@pytest.mark.parametrize(
    ("upstream_model", "reasoning_effort"),
    (
        pytest.param("claude-haiku-4-5", "turbo", id="unknown-value"),
        pytest.param("claude-opus-4-6", "xhigh", id="level-the-model-lacks"),
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
            "POST", "/v1/messages", {**_claude_code_turn({"reasoning_effort": reasoning_effort}), "model": model}
        )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["type"] == "invalid_request_error", response.text
        assert wire.drain() == ()

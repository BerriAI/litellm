"""Contract: how OpenAI reasoning controls on `/v1/chat/completions` become Anthropic thinking, and back.

`AnthropicConfig.map_openai_params` (`litellm/llms/anthropic/chat/transformation.py`) maps `reasoning_effort` to
`thinking`: on a pre-4.6 model such as `claude-sonnet-4-5-20250929` it is a legacy `{"type": "enabled",
"budget_tokens": N}` with N from `litellm/constants.py` (low 1024, medium 2048, high 4096, xhigh 8192, max 16384,
minimal raised to the 1024 minimum); on an adaptive model such as `claude-opus-4-6` it is `{"type": "adaptive",
"display": "summarized"}` plus `output_config.effort`. When thinking has a budget and the caller sent no
`max_tokens`, `max_tokens` becomes budget + 4096 (`update_optional_params_with_thinking_tokens`), because Anthropic
requires `max_tokens > budget_tokens`. A caller `thinking` object is forwarded, translated down for models that do
not take its shape. On the way back, signed thinking blocks the caller replays in assistant history must reach
Anthropic unchanged and in front of the turn's tool calls, or Anthropic rejects the signature.
"""

from __future__ import annotations

from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration.providers.dec1_anthropic_support import (
    ADAPTIVE_BACKEND,
    SIGNATURE,
    answering,
    betas,
    chat,
    content_blocks,
    deployment,
    error_message,
    peer,
    sent_message,
    served,
    tool,
    usage_block,
    user,
)
from pydantic import JsonValue


@pytest.mark.parametrize(
    ("effort", "budget"),
    [("minimal", 1024), ("low", 1024), ("medium", 2048), ("high", 4096), ("xhigh", 8192), ("max", 16384)],
)
def test_reasoning_effort_becomes_an_enabled_thinking_budget_on_a_legacy_model(
    gateway: Gateway, effort: str, budget: int
) -> None:
    """On `claude-sonnet-4-5-20250929` the caller sends `reasoning_effort` and no `max_tokens`.

    The upstream body must carry `thinking: {"type": "enabled", "budget_tokens": <budget>}` from the constants, and
    `max_tokens` must be budget + 4096 so it exceeds the budget. No `reasoning_effort` key may be sent upstream.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), reasoning_effort=effort))
        body: Final = provider.only().body
        assert body["thinking"] == {"type": "enabled", "budget_tokens": budget}
        assert body["max_tokens"] == budget + 4096
        assert "reasoning_effort" not in body


def test_reasoning_effort_with_a_caller_max_tokens_keeps_the_caller_value(gateway: Gateway) -> None:
    """The caller sends `reasoning_effort: low` and `max_tokens: 3000`. The budget is 1024 and the upstream
    `max_tokens` must stay 3000; the budget + 4096 rule applies only when the caller set no limit.
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), reasoning_effort="low", max_tokens=3000))
        body: Final = provider.only().body
        assert body["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert body["max_tokens"] == 3000


def test_reasoning_effort_none_sends_no_thinking_on_a_model_that_does_not_think_by_default(gateway: Gateway) -> None:
    """The caller sends `reasoning_effort: none` to `claude-sonnet-4-5-20250929`, which does not think by default.

    No `thinking` and no `output_config` may be sent (`_map_reasoning_effort` returns None for `none`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), reasoning_effort="none"))
        body: Final = provider.only().body
        assert "thinking" not in body
        assert "output_config" not in body


@pytest.mark.parametrize(
    ("effort", "mapped"), [("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high")]
)
def test_reasoning_effort_becomes_adaptive_thinking_and_output_config_effort_on_an_adaptive_model(
    gateway: Gateway, effort: str, mapped: str
) -> None:
    """On `claude-opus-4-6` (`supports_adaptive_thinking: true`) the caller sends `reasoning_effort`.

    The upstream body must carry `thinking: {"type": "adaptive", "display": "summarized"}` and `output_config:
    {"effort": <mapped>}` (`REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT`, minimal maps to low). Without `display`,
    Anthropic would return blank thinking blocks. No `effort-2025-11-24` beta is sent for 4.6 models.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, f"anthropic/{ADAPTIVE_BACKEND}", priced=False)
        served(chat(gateway, model, reasoning_effort=effort))
        call: Final = provider.only()
        assert call.body["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert call.body["output_config"] == {"effort": mapped}
        assert "effort-2025-11-24" not in betas(call)


def test_an_unknown_reasoning_effort_is_refused_with_400_before_the_provider(gateway: Gateway) -> None:
    """The caller sends `reasoning_effort: "turbo"` to `claude-sonnet-4-5-20250929`.

    The caller must get 400 whose message names `Unmapped reasoning effort` and the allowed values, and the peer
    must not be called (`_map_reasoning_effort` raises BadRequestError).
    """
    with peer() as provider, gateway.scenario() as scenario:
        response: Final = chat(gateway, deployment(scenario, provider), reasoning_effort="turbo")
        assert response.status_code == 400, response.text
        assert "Unmapped reasoning effort" in error_message(response)
        assert provider.posts() == ()


def test_a_caller_enabled_thinking_object_is_forwarded_with_a_fitting_max_tokens(gateway: Gateway) -> None:
    """The caller sends `thinking: {"type": "enabled", "budget_tokens": 2000}` and no `max_tokens` to
    `claude-sonnet-4-5-20250929`. Upstream `thinking` must be the same object and `max_tokens` 6096 (2000 + 4096).
    """
    thinking: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 2000}
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), thinking=thinking))
        body: Final = provider.only().body
        assert body["thinking"] == thinking
        assert body["max_tokens"] == 6096


def test_adaptive_thinking_on_a_pre_adaptive_model_becomes_a_medium_legacy_budget(gateway: Gateway) -> None:
    """The caller (Claude Code style) sends `thinking: {"type": "adaptive"}` and `max_tokens: 5000` to
    `claude-sonnet-4-5-20250929`, which rejects adaptive thinking.

    Upstream `thinking` must be `{"type": "enabled", "budget_tokens": 2048}` (the medium budget, under max_tokens).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), thinking={"type": "adaptive"}, max_tokens=5000))
        assert provider.only().body["thinking"] == {"type": "enabled", "budget_tokens": 2048}


def test_adaptive_thinking_on_a_pre_adaptive_model_is_dropped_when_max_tokens_cannot_fit_a_budget(
    gateway: Gateway,
) -> None:
    """Same adaptive request with `max_tokens: 1000`, which is not above the 1024 minimum budget.

    No `thinking` may be sent and `max_tokens` stays 1000 (`cap_thinking_budget_to_max_tokens` returns None).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), thinking={"type": "adaptive"}, max_tokens=1000))
        body: Final = provider.only().body
        assert "thinking" not in body
        assert body["max_tokens"] == 1000


def test_adaptive_thinking_with_a_small_max_tokens_caps_the_legacy_budget_below_it(gateway: Gateway) -> None:
    """The caller sends `thinking: {"type": "adaptive"}` and `max_tokens: 1500` to `claude-sonnet-4-5-20250929`.

    The medium budget 2048 does not fit, so upstream `thinking` must be `{"type": "enabled", "budget_tokens": 1499}`
    (max_tokens - 1, `cap_thinking_budget_to_max_tokens`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), thinking={"type": "adaptive"}, max_tokens=1500))
        assert provider.only().body["thinking"] == {"type": "enabled", "budget_tokens": 1499}


def test_legacy_thinking_on_a_model_without_legacy_support_becomes_adaptive_with_an_effort(gateway: Gateway) -> None:
    """The caller sends `thinking: {"type": "enabled", "budget_tokens": 5000}` to `claude-opus-4-7`, an adaptive
    model without `supports_legacy_thinking`.

    Upstream `thinking` must be `{"type": "adaptive"}` and `output_config` `{"effort": "high"}`, since 5000 is at
    least the 4096 high budget (`translate_legacy_thinking_for_adaptive_model_silently`, `_legacy_budget_to_effort`).
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, "anthropic/claude-opus-4-7", priced=False)
        served(chat(gateway, model, thinking={"type": "enabled", "budget_tokens": 5000}, max_tokens=8000))
        body: Final = provider.only().body
        assert body["thinking"] == {"type": "adaptive"}
        assert body["output_config"] == {"effort": "high"}


def test_legacy_thinking_on_a_4_6_model_is_forwarded_verbatim(gateway: Gateway) -> None:
    """The caller sends `thinking: {"type": "enabled", "budget_tokens": 3000}` and `max_tokens: 8000` to
    `claude-opus-4-6` (`supports_legacy_thinking: true`). Upstream `thinking` must be the caller's object unchanged
    and no `output_config` may be added.
    """
    thinking: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 3000}
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, f"anthropic/{ADAPTIVE_BACKEND}", priced=False)
        served(chat(gateway, model, thinking=thinking, max_tokens=8000))
        body: Final = provider.only().body
        assert body["thinking"] == thinking
        assert "output_config" not in body


def test_thinking_display_updates_adds_the_display_updates_beta(gateway: Gateway) -> None:
    """The caller sends `thinking: {"type": "adaptive", "display": "updates"}` to `claude-opus-4-6`.

    The request must carry the thinking display updates beta (`thinking_betas`,
    `ANTHROPIC_THINKING_DISPLAY_UPDATES_BETA_HEADER`). UNVERIFIED: the beta string itself; the test only checks a
    beta containing `display` or `thinking` was added.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider, f"anthropic/{ADAPTIVE_BACKEND}", priced=False)
        served(chat(gateway, model, thinking={"type": "adaptive", "display": "updates"}))
        assert any("thinking" in beta or "display" in beta for beta in betas(provider.only()))


def test_a_replayed_signed_thinking_block_precedes_the_tool_use_in_the_assistant_turn(gateway: Gateway) -> None:
    """The history has an assistant turn with `thinking_blocks: [{"type": "thinking", "thinking": "plan",
    "signature": ...}]` and one tool call, then the tool result. The caller sends `reasoning_effort: low`.

    Upstream, the assistant content must be the thinking block (text and signature unchanged) followed by the
    `tool_use` block, and `thinking` stays enabled: Anthropic verifies signatures by position and rejects a tool
    turn whose thinking was dropped.
    """
    thinking: Final[dict[str, JsonValue]] = {"type": "thinking", "thinking": "plan", "signature": SIGNATURE}
    history: Final[list[JsonValue]] = [
        user(),
        {
            "role": "assistant",
            "content": None,
            "thinking_blocks": [thinking],
            "tool_calls": [
                {"id": "toolu_dec1a", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_dec1a", "content": "18C"},
    ]
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                messages=history,
                tools=[tool("get_weather", city={"type": "string"})],
                reasoning_effort="low",
            )
        )
        call: Final = provider.only()
        blocks: Final = content_blocks(sent_message(call, 1))
        assert [b["type"] for b in blocks] == ["thinking", "tool_use"]
        assert blocks[0] == thinking
        assert object_value(call.body["thinking"])["type"] == "enabled"


def test_a_replayed_redacted_thinking_block_is_sent_with_its_data(gateway: Gateway) -> None:
    """The history's assistant turn carries `thinking_blocks: [{"type": "redacted_thinking", "data": ...}]` and
    text `ok`.

    Upstream, the assistant turn must open with the redacted block, its `data` unchanged, then the text.
    """
    redacted: Final[dict[str, JsonValue]] = {"type": "redacted_thinking", "data": "EmwKAhgBEgy3dec1redacted"}
    history: Final[list[JsonValue]] = [
        user(),
        {"role": "assistant", "content": "ok", "thinking_blocks": [redacted]},
        user(),
    ]
    with peer() as provider, gateway.scenario() as scenario:
        served(chat(gateway, deployment(scenario, provider), messages=history, reasoning_effort="low"))
        blocks: Final = content_blocks(sent_message(provider.only(), 1))
        assert blocks[0] == redacted
        assert blocks[-1].get("text") == "ok"


def test_thinking_tokens_are_reported_as_reasoning_tokens_when_the_provider_counts_them(gateway: Gateway) -> None:
    """The provider answers a thinking block and text with `usage.output_tokens: 7` and
    `output_tokens_details: {"thinking_tokens": 4}`.

    The caller's `usage.completion_tokens_details.reasoning_tokens` must be 4 (`thinking_tokens_from_usage` reads
    `AnthropicOutputTokensDetails.thinking_tokens`).
    """
    content: Final[list[JsonValue]] = [
        {"type": "thinking", "thinking": "plan", "signature": SIGNATURE},
        {"type": "text", "text": "done"},
    ]
    usage_in: Final = usage_block(output_tokens_details={"thinking_tokens": 4})
    with peer(answering(content, usage=usage_in)) as provider, gateway.scenario() as scenario:
        body: Final = served(chat(gateway, deployment(scenario, provider), reasoning_effort="low"))
        details: Final = object_value(object_value(body["usage"])["completion_tokens_details"])
        assert details["reasoning_tokens"] == 4


def test_tools_and_thinking_together_never_force_a_tool_choice(gateway: Gateway) -> None:
    """The caller sends a tool, `reasoning_effort: low` and an open-schema `response_format`.

    Anthropic refuses forced tool use while thinking, so the upstream body must not carry `tool_choice: {"type":
    "tool", "name": "json_tool_call"}` (`response_format_tool_can_be_forced` is false when the model will think).
    """
    schema: Final[dict[str, JsonValue]] = {"type": "object", "properties": {"city": {"type": "string"}}}
    with peer() as provider, gateway.scenario() as scenario:
        served(
            chat(
                gateway,
                deployment(scenario, provider),
                reasoning_effort="low",
                response_format={"type": "json_schema", "json_schema": {"name": "place", "schema": schema}},
            )
        )
        body: Final = provider.only().body
        assert body.get("tool_choice") != {"type": "tool", "name": "json_tool_call"}
        assert object_value(body["thinking"])["type"] == "enabled"
        tools: Final = list_value(body.get("tools", []))
        assert all(object_value(t).get("name") != "json_tool_call" or "description" in object_value(t) for t in tools)

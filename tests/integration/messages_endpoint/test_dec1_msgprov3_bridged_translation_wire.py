"""`/v1/messages` on bridged providers: Gemini and OpenAI answers translated back to Anthropic messages.

Providers without a native Messages config are BRIDGED: `ProviderConfigManager.get_provider_anthropic_messages_config`
(`litellm/utils.py:10452-10522`) returns None for `gemini/` and `openai/`, so the proxy turns the Anthropic request
into a chat call (`LiteLLMMessagesToCompletionTransformationHandler`, `litellm/llms/anthropic/messages/adapters/
handler.py`) and the answer back into an Anthropic message (`translate_openai_response_to_anthropic`,
`litellm/llms/anthropic/messages/adapters/transformation.py:1714`). The catalog keeps these targets on Python
(`litellm/rust_bridge/catalog.py:203-207`), so they are the next migration candidates after the native ports.

Gemini: the finish reason goes through `map_finish_reason` (`litellm/litellm_core_utils/core_helpers.py:453-456`,
`MAX_TOKENS` to `length`, `SAFETY` to `content_filter`) and then `_translate_openai_finish_reason_to_anthropic`
(`transformation.py:1607-1616`, `length` to `max_tokens`, `content_filter` to `refusal`). Usage comes from
`usageMetadata` (`litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py:2604,2706-2760`): cached
tokens move out of `input_tokens` into `cache_read_input_tokens` (`transformation.py:1678-1696`) and thought tokens
are added to the output count when `candidatesTokenCount` excludes them. OpenAI: with `thinking` enabled on a
reasoning model the bridge rewrites the model to `openai/responses/<model>` and maps `budget_tokens` to
`reasoning.effort` with the shared thresholds 1024 low, 2048 medium, 4096 high (`handler.py:552-590`,
`litellm/litellm_core_utils/reasoning_effort_utils.py:69-83`, `litellm/constants.py:371-375`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import JSON_OBJECT
from integration.messages_endpoint.dec1_msgprov3_support import (
    OPENAI_KEY,
    bridged_deployment,
    model_calls,
    responses_json,
)
from integration.messages_endpoint.dec1_msgprov_support import (
    PRICES,
    at,
    body_of,
    only_spend_row,
    path_of,
    spend_of,
)
from integration.providers.dec1_gemini_support import gemini_json, reply_json, usage_metadata
from pydantic import JsonValue


def _gemini_answer(**fields: JsonValue) -> Callable[[Request], Reply]:
    reply: Final = reply_json(gemini_json(**fields))  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough
    return lambda _request: reply


def _message(response_content: bytes) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response_content)


def _usage(message: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(message["usage"])


# ---------------------------------------------------------------- Gemini stop reasons


def test_a_gemini_max_tokens_finish_becomes_stop_reason_max_tokens(gateway: Gateway) -> None:
    """Gemini answering `finishReason: MAX_TOKENS` gives the caller `stop_reason: "max_tokens"` and
    `stop_sequence: null`, which Claude Code reads to continue a truncated answer (`MAX_TOKENS` to `length` to
    `max_tokens`)."""
    with bridged_deployment(gateway, "gemini", _gemini_answer(finish="MAX_TOKENS", text="cut")) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        message: Final = _message(response.content)
        assert (message["stop_reason"], message["stop_sequence"]) == ("max_tokens", None)


def test_a_gemini_safety_finish_becomes_a_refusal(gateway: Gateway) -> None:
    """Gemini answering `finishReason: SAFETY` gives `stop_reason: "refusal"` (`SAFETY` to `content_filter` to
    `refusal`), so the caller can tell a blocked answer from a complete one."""
    with bridged_deployment(gateway, "gemini", _gemini_answer(finish="SAFETY", text="")) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        assert _message(response.content)["stop_reason"] == "refusal"


def test_a_gemini_function_call_answer_stops_with_tool_use_even_when_gemini_says_stop(gateway: Gateway) -> None:
    """Gemini reports `finishReason: STOP` next to a `functionCall` part; the caller still gets `stop_reason:
    "tool_use"` and a `tool_use` block named `get_weather` with `input: {"city": "Paris"}`, because
    `_check_finish_reason` prefers `tool_calls` whenever the message has them (`vertex_and_google_ai_studio_gemini.py:
    2776-2789`)."""
    parts: Final[list[JsonValue]] = [{"functionCall": {"name": "get_weather", "args": {"city": "Paris"}}}]
    tool: Final[dict[str, JsonValue]] = {
        "name": "get_weather",
        "description": "Weather for a city",
        "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
    }
    with bridged_deployment(gateway, "gemini", _gemini_answer(parts=parts, finish="STOP")) as rig:
        response: Final = rig.post(rig.body(tools=[tool]))
        assert response.status_code == 200, response.text
        message: Final = _message(response.content)
        assert message["stop_reason"] == "tool_use"
        blocks: Final = [object_value(block) for block in list_value(message["content"])]
        uses: Final = [block for block in blocks if block.get("type") == "tool_use"]
        assert [(block["name"], block["input"]) for block in uses] == [("get_weather", {"city": "Paris"})]


# ---------------------------------------------------------------- Gemini usage


def test_gemini_cached_content_tokens_become_cache_read_input_tokens(gateway: Gateway) -> None:
    """`usageMetadata: {promptTokenCount: 100, candidatesTokenCount: 10, cachedContentTokenCount: 60,
    totalTokenCount: 110}` is answered as `usage.input_tokens: 40`, `cache_read_input_tokens: 60` and
    `output_tokens: 10`: Anthropic counts cached tokens separately from `input_tokens` while Gemini includes them
    in the prompt count."""
    usage: Final = usage_metadata(100, 10, cachedContentTokenCount=60)
    with bridged_deployment(gateway, "gemini", _gemini_answer(usage=usage)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        counts: Final = _usage(_message(response.content))
        assert (counts["input_tokens"], counts.get("cache_read_input_tokens"), counts["output_tokens"]) == (40, 60, 10)


def test_gemini_thought_tokens_are_counted_as_output_tokens(gateway: Gateway) -> None:
    """With `promptTokenCount: 11`, `candidatesTokenCount: 7`, `thoughtsTokenCount: 20` and `totalTokenCount: 38`
    the candidate count excludes thinking (11 + 7 != 38), so `output_tokens` is 27 and no cache field is reported
    (`is_candidate_token_count_inclusive`, `vertex_and_google_ai_studio_gemini.py:2534-2554,2752-2754`)."""
    usage: Final = usage_metadata(11, 7, thoughtsTokenCount=20, totalTokenCount=38)
    with bridged_deployment(gateway, "gemini", _gemini_answer(usage=usage)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        counts: Final = _usage(_message(response.content))
        assert (counts["input_tokens"], counts["output_tokens"]) == (11, 27)
        assert "cache_read_input_tokens" not in counts and "cache_creation_input_tokens" not in counts


def test_a_gemini_thought_part_becomes_a_leading_thinking_block(gateway: Gateway) -> None:
    """A Gemini answer with parts `[{"text": "weighing options", "thought": true}, {"text": "Paris."}]` comes back
    as `content[0] = {"type": "thinking", "thinking": "weighing options", ...}` followed by the text block
    `Paris.`; the thought never leaks into the text (`_translate_openai_content_to_anthropic`,
    `transformation.py:1530-1561`). UNVERIFIED: the `signature` value of an unsigned Gemini thought."""
    parts: Final[list[JsonValue]] = [{"text": "weighing options", "thought": True}, {"text": "Paris."}]
    with bridged_deployment(gateway, "gemini", _gemini_answer(parts=parts)) as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        blocks: Final = [object_value(block) for block in list_value(_message(response.content)["content"])]
        assert (blocks[0]["type"], blocks[0]["thinking"]) == ("thinking", "weighing options")
        assert blocks[-1] == {"type": "text", "text": "Paris."}


def test_a_gemini_call_with_cached_tokens_is_priced_from_the_gemini_row(gateway: Gateway) -> None:
    """With no deployment prices, 1000 prompt tokens of which 400 cached and 100 candidate tokens are priced on the
    `gemini/gemini-2.5-flash` row (input 3e-07, cache read 3e-08, output 2.5e-06): `600 * 3e-07 + 400 * 3e-08 +
    100 * 2.5e-06 = 0.000442`, written on the SpendLogs row with `custom_llm_provider: gemini`."""
    usage: Final = usage_metadata(1000, 100, cachedContentTokenCount=400)
    with bridged_deployment(gateway, "gemini", _gemini_answer(usage=usage)) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert row["custom_llm_provider"] == "gemini"
        assert spend_of(row) == pytest.approx(600 * 3e-07 + 400 * 3e-08 + 100 * 2.5e-06)


# ---------------------------------------------------------------- OpenAI through the Responses API


@pytest.mark.parametrize(("budget", "effort"), ((1024, "low"), (2048, "medium"), (4096, "high")))
def test_openai_thinking_budget_becomes_a_responses_reasoning_effort(
    gateway: Gateway, budget: int, effort: str
) -> None:
    """`thinking: {"type": "enabled", "budget_tokens": <budget>}` on `openai/gpt-5.4-mini` posts to the Responses
    API with `reasoning.effort` bucketed from the budget: 1024 is `low`, 2048 `medium`, 4096 `high`
    (`reasoning_effort_from_thinking_budget`). A port that forwards `budget_tokens` would be rejected by OpenAI."""
    with bridged_deployment(gateway, "openai") as rig:
        thinking: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": budget}
        response: Final = rig.post(rig.body(max_tokens=8192, thinking=thinking))
        assert response.status_code == 200, response.text
        calls: Final = model_calls(rig, "openai")
        assert len(calls) == 1, [call.target for call in calls]
        assert path_of(calls[0]).endswith("/responses")
        assert at(body_of(calls[0]), "reasoning", "effort") == effort


def test_openai_thinking_sends_no_reasoning_summary_by_default(gateway: Gateway) -> None:
    """Without `thinking.summary` and with `reasoning_auto_summary` off (the default, `litellm/llms/anthropic/messages/
    utils.py:243-245`) the Responses `reasoning` object has no `summary` key (`handler.py:582-590`)."""
    with bridged_deployment(gateway, "openai") as rig:
        thinking: Final[dict[str, JsonValue]] = {"type": "enabled", "budget_tokens": 2048}
        assert rig.post(rig.body(max_tokens=8192, thinking=thinking)).status_code == 200
        reasoning: Final = object_value(at(body_of(model_calls(rig, "openai")[0]), "reasoning"))
        assert "summary" not in reasoning, reasoning


def test_openai_bridge_authenticates_with_the_deployment_bearer_key(gateway: Gateway) -> None:
    """The OpenAI call carries `Authorization: Bearer synthetic-openai-key` (the deployment key, never the caller's
    proxy key) and no `x-api-key` or `anthropic-version` header."""
    with bridged_deployment(gateway, "openai") as rig:
        assert rig.post(rig.body(), headers={"anthropic-version": "2023-06-01"}).status_code == 200
        call: Final = model_calls(rig, "openai")[0]
        assert call.headers["authorization"] == f"Bearer {OPENAI_KEY}"
        assert "x-api-key" not in call.headers and "anthropic-version" not in call.headers


def test_an_openai_responses_answer_becomes_an_anthropic_message(gateway: Gateway) -> None:
    """A Responses answer with one `output_text` of `pong` and usage 11 in / 7 out is returned as an Anthropic
    message: `type: message`, `role: assistant`, `content: [{"type": "text", "text": "pong"}]`,
    `stop_reason: end_turn`, `usage.input_tokens: 11`, `usage.output_tokens: 7`."""
    with bridged_deployment(gateway, "openai") as rig:
        response: Final = rig.post(rig.body())
        assert response.status_code == 200, response.text
        message: Final = _message(response.content)
        assert (message["type"], message["role"], message["stop_reason"]) == ("message", "assistant", "end_turn")
        assert message["content"] == [{"type": "text", "text": "pong"}]
        assert (_usage(message)["input_tokens"], _usage(message)["output_tokens"]) == (11, 7)


def test_an_openai_responses_call_is_priced_and_named_openai(gateway: Gateway) -> None:
    """With deployment prices 0.001 in and 0.002 out the Responses answer (11 in, 7 out) writes one SpendLogs row
    with `custom_llm_provider: openai`, `call_type: anthropic_messages` and spend 0.025."""
    reply: Final = Reply(body=JSON_OBJECT.dump_json(responses_json()))
    with bridged_deployment(gateway, "openai", lambda _request: reply, **PRICES) as rig:
        assert rig.post(rig.body()).status_code == 200
        row: Final = only_spend_row(rig)
        assert (row["custom_llm_provider"], row["call_type"]) == ("openai", "anthropic_messages")
        assert spend_of(row) == pytest.approx(11 * 0.001 + 7 * 0.002)

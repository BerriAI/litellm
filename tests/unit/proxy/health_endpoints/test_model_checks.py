import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.proxy.health_endpoints.model_checks import (
    CHAT_COMPLETIONS_SCRIPT,
    MESSAGES_SCRIPT,
    RESPONSES_SCRIPT,
    WEATHER_TOOL_RESULT,
    CalledTurn,
    EndpointScript,
    endpoint_result_payload,
    overall_status,
    run_endpoint_check,
)
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse

_TOOL_CALL_ID: Final = "call_boston"


def _chat_reply(*, tool_call: bool, text: str = "", cached_tokens: int = 0, prompt_tokens: int = 5000) -> ModelResponse:
    message: Final = (
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": _TOOL_CALL_ID,
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": json.dumps({"location": "Boston"})},
                }
            ],
        }
        if tool_call
        else {"role": "assistant", "content": text}
    )
    return ModelResponse(
        id="chatcmpl-1",
        model="gpt-test",
        choices=[{"index": 0, "finish_reason": "tool_calls" if tool_call else "stop", "message": message}],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 12,
            "total_tokens": prompt_tokens + 12,
            "prompt_tokens_details": {"cached_tokens": cached_tokens},
        },
    )


def _responses_reply(*, tool_call: bool, text: str = "", cached_tokens: int = 0) -> ResponsesAPIResponse:
    output: Final = (
        [
            {"type": "reasoning", "id": "rs_1", "summary": []},
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": _TOOL_CALL_ID,
                "name": "get_weather",
                "arguments": json.dumps({"location": "Boston"}),
                "status": "completed",
            },
        ]
        if tool_call
        else [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ]
    )
    return ResponsesAPIResponse.model_validate(
        {
            "id": "resp_1",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "gpt-test",
            "output": output,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 5000,
                "output_tokens": 12,
                "total_tokens": 5012,
                "input_tokens_details": {"cached_tokens": cached_tokens},
            },
        }
    )


def _messages_reply(*, tool_call: bool, text: str = "", cached_tokens: int = 0) -> Mapping[str, object]:
    content: Final = (
        [
            {"type": "thinking", "thinking": "look it up", "signature": "sig"},
            {"type": "tool_use", "id": _TOOL_CALL_ID, "name": "get_weather", "input": {"location": "Boston"}},
        ]
        if tool_call
        else [{"type": "text", "text": text}]
    )
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-test",
        "content": content,
        "stop_reason": "tool_use" if tool_call else "end_turn",
        "usage": {
            "input_tokens": 5000 - cached_tokens,
            "output_tokens": 12,
            "cache_read_input_tokens": cached_tokens,
            "cache_creation_input_tokens": 0,
        },
    }


class _ScriptedProvider:
    """Answers turn 1 with a tool call and every later turn from ``second_turns``, recording what was sent."""

    def __init__(
        self,
        first: CalledTurn,
        second_turns: Sequence[CalledTurn | Exception],
    ) -> None:
        self.first = first
        self.second_turns = list(second_turns)
        self.sent: list[Mapping[str, object]] = []

    async def __call__(self, script: EndpointScript, body: Mapping[str, object], call_id: str) -> CalledTurn:
        self.sent.append(body)
        if len(self.sent) == 1:
            return self.first
        outcome: Final = self.second_turns.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ids() -> Sequence[str]:
    return ("turn-1", "turn-2", "turn-2-retry")


def _run(script: EndpointScript, provider: _ScriptedProvider, spend_rows: Mapping[str, float] | None = None):
    ids: Final = iter(_ids())

    async def spend_lookup(request_ids: Sequence[str]) -> Mapping[str, float]:
        return spend_rows or {}

    return asyncio.run(
        run_endpoint_check(
            script,
            provider,
            new_call_id=lambda: next(ids),
            spend_lookup=spend_lookup if spend_rows is not None else None,
        )
    )


def _checks(result) -> Mapping[str, tuple[str, str]]:
    return {check.name: (check.status, check.detail) for check in result.checks}


@pytest.mark.parametrize(
    ("script", "first", "second"),
    [
        (
            CHAT_COMPLETIONS_SCRIPT,
            _chat_reply(tool_call=True),
            _chat_reply(tool_call=False, text="Boston is 72F and sunny.", cached_tokens=4800),
        ),
        (
            RESPONSES_SCRIPT,
            _responses_reply(tool_call=True),
            _responses_reply(tool_call=False, text="Boston is 72F and sunny.", cached_tokens=4800),
        ),
        (
            MESSAGES_SCRIPT,
            _messages_reply(tool_call=True),
            _messages_reply(tool_call=False, text="Boston is 72F and sunny.", cached_tokens=4800),
        ),
    ],
    ids=["chat", "responses", "messages"],
)
def test_every_endpoint_passes_all_three_checks_and_feeds_turn_one_back_in_its_own_shape(script, first, second):
    provider: Final = _ScriptedProvider(CalledTurn(first, 0.004), [CalledTurn(second, 0.0005)])

    result: Final = _run(script, provider, spend_rows={"turn-1": 0.004, "turn-2": 0.0005})

    assert result.status == "success", result
    checks: Final = _checks(result)
    assert checks["conversation"][0] == "pass", checks
    assert checks["prompt_caching"] == ("pass", "turn 2 read 4800 of 5000 input tokens from cache (96.0%)"), checks
    assert checks["cost_tracking"] == (
        "pass",
        "turn 1 $0.004000, turn 2 $0.000500, spend log rows written ($0.004000, $0.000500)",
    ), checks
    assert result.cache_read_tokens == 4800
    assert result.cache_read_percent == 96.0
    assert result.spend_usd == pytest.approx(0.0045)
    assert len(provider.sent) == 2
    turn_two: Final = provider.sent[1]
    sent_json: Final = json.dumps(turn_two)
    assert _TOOL_CALL_ID in sent_json and WEATHER_TOOL_RESULT in sent_json, turn_two
    assert "cache_control" not in sent_json and "cachePoint" not in sent_json, turn_two
    if script is CHAT_COMPLETIONS_SCRIPT:
        tool_message: Final = turn_two["messages"][-1]
        assert tool_message == {"role": "tool", "tool_call_id": _TOOL_CALL_ID, "content": WEATHER_TOOL_RESULT}
        assert turn_two["messages"][-2]["tool_calls"][0]["id"] == _TOOL_CALL_ID
    elif script is RESPONSES_SCRIPT:
        assert turn_two["input"][-1] == {
            "type": "function_call_output",
            "call_id": _TOOL_CALL_ID,
            "output": WEATHER_TOOL_RESULT,
        }
        assert [item["type"] for item in turn_two["input"][1:-1]] == ["reasoning", "function_call"]
    else:
        assert turn_two["messages"][-1]["content"] == [
            {"type": "tool_result", "tool_use_id": _TOOL_CALL_ID, "content": WEATHER_TOOL_RESULT}
        ]
        assert turn_two["messages"][-2]["content"][0]["type"] == "thinking"


def test_a_model_that_answers_in_text_instead_of_calling_the_tool_fails_the_conversation_and_stops():
    provider: Final = _ScriptedProvider(CalledTurn(_chat_reply(tool_call=False, text="It is sunny."), 0.004), [])

    result: Final = _run(CHAT_COMPLETIONS_SCRIPT, provider)

    assert result.status == "error"
    checks: Final = _checks(result)
    assert checks["conversation"] == ("fail", "turn 1 returned no tool call, the model answered: 'It is sunny.'")
    assert checks["prompt_caching"] == ("fail", "not checked, the conversation did not reach turn 2")
    assert checks["cost_tracking"] == ("pass", "turn 1 $0.004000, spend log not checked (no database)")
    assert len(provider.sent) == 1


def test_a_turn_two_provider_error_is_reported_as_the_conversation_failure():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004), [RuntimeError("400 invalid tool message")]
    )

    result: Final = _run(CHAT_COMPLETIONS_SCRIPT, provider)

    checks: Final = _checks(result)
    assert checks["conversation"] == (
        "fail",
        "turn 2 request with the tool result failed: RuntimeError: 400 invalid tool message",
    )
    assert result.status == "error"


def test_an_answer_that_ignores_the_tool_result_fails_the_conversation():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [CalledTurn(_chat_reply(tool_call=False, text="I cannot tell.", cached_tokens=4800), 0.0005)],
    )

    checks: Final = _checks(_run(CHAT_COMPLETIONS_SCRIPT, provider))

    assert checks["conversation"] == ("fail", "turn 2 did not use the tool result '72F and sunny': 'I cannot tell.'")
    assert checks["prompt_caching"][0] == "pass"


def test_zero_cached_tokens_is_retried_once_and_fails_when_the_retry_misses_too():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=0), 0.004),
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=0), 0.004),
        ],
    )

    result: Final = _run(CHAT_COMPLETIONS_SCRIPT, provider)

    assert len(provider.sent) == 3
    assert _checks(result)["prompt_caching"] == (
        "fail",
        "turn 2 reported 0 cached tokens on 5000 input tokens in 2 attempts, the provider did not reuse the "
        "repeated system prompt",
    )
    assert result.status == "error"
    assert result.cache_read_tokens == 0


def test_a_cache_hit_on_the_retry_passes_and_says_so():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=0), 0.004),
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=5000), 0.0005),
        ],
    )

    result: Final = _run(CHAT_COMPLETIONS_SCRIPT, provider)

    assert _checks(result)["prompt_caching"] == (
        "pass",
        "turn 2 read 5000 of 5000 input tokens from cache (100.0%) on the second attempt",
    )
    assert result.spend_usd == pytest.approx(0.0045)


def test_a_model_without_a_price_fails_cost_tracking_and_a_zero_priced_model_only_warns():
    unpriced: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), None),
        [CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), None)],
    )
    free: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.0),
        [CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0)],
    )

    unpriced_result: Final = _run(CHAT_COMPLETIONS_SCRIPT, unpriced)
    free_result: Final = _run(CHAT_COMPLETIONS_SCRIPT, free)

    assert _checks(unpriced_result)["cost_tracking"] == (
        "fail",
        "turn 1 carried no response_cost, no price resolved for this model (not in the cost map and no custom "
        "pricing on the deployment)",
    )
    assert unpriced_result.status == "error"
    assert _checks(free_result)["cost_tracking"] == (
        "warn",
        "turn 1 response_cost is $0, this model is priced at zero",
    )
    assert free_result.status == "warning"
    assert overall_status([unpriced_result, free_result]) == "error"
    assert overall_status([free_result]) == "warning"


def test_a_missing_spend_log_row_fails_cost_tracking_even_when_the_response_was_priced():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0005)],
    )

    checks: Final = _checks(_run(CHAT_COMPLETIONS_SCRIPT, provider, spend_rows={"turn-1": 0.004}))

    assert checks["cost_tracking"] == (
        "fail",
        "turn 1 $0.004000, turn 2 $0.000500, but no LiteLLM_SpendLogs row landed for request_id turn-2",
    )


def test_a_response_that_is_not_the_endpoint_shape_fails_the_conversation_with_the_parse_error():
    provider: Final = _ScriptedProvider(CalledTurn({"id": "x", "object": "nonsense"}, 0.001), [])

    checks: Final = _checks(_run(MESSAGES_SCRIPT, provider))

    assert checks["conversation"][0] == "fail"
    assert checks["conversation"][1].startswith("turn 1 Messages API response did not parse: content"), checks


def test_payload_lists_the_checks_in_order_with_their_details():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0005)],
    )

    payload: Final = endpoint_result_payload(_run(CHAT_COMPLETIONS_SCRIPT, provider)).model_dump()

    assert payload["endpoint"] == "/v1/chat/completions"
    assert payload["status"] == "success"
    assert [check["name"] for check in payload["checks"]] == ["conversation", "prompt_caching", "cost_tracking"]
    assert all(check["status"] == "pass" for check in payload["checks"])
    assert payload["cache_read_percent"] == 96.0
    assert isinstance(payload["duration_ms"], int)

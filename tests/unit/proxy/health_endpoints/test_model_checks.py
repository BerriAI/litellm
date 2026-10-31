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
from litellm.proxy.health_endpoints.test_model_requests import REQUEST_FILES
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse

_TOOL_CALL_ID: Final = "call_boston"
_TURN_TWO_ANSWER: Final = "Boston is 72F and sunny."
_TURN_THREE_ANSWER: Final = "Yes, 72F is warm enough for a walk."


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
    """Answers turn 1 from ``first`` and every later request from ``later_turns`` in order, recording what was sent."""

    def __init__(
        self,
        first: CalledTurn,
        later_turns: Sequence[CalledTurn | Exception],
    ) -> None:
        self.first = first
        self.later_turns = list(later_turns)
        self.sent: list[Mapping[str, object]] = []

    async def __call__(self, script: EndpointScript, body: Mapping[str, object], call_id: str) -> CalledTurn:
        self.sent.append(body)
        if len(self.sent) == 1:
            return self.first
        outcome: Final = self.later_turns.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _ids() -> Sequence[str]:
    return ("call-1", "call-2", "call-3", "call-4", "call-5")


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
    ("script", "reply", "key"),
    [
        (CHAT_COMPLETIONS_SCRIPT, _chat_reply, "messages"),
        (RESPONSES_SCRIPT, _responses_reply, "input"),
        (MESSAGES_SCRIPT, _messages_reply, "messages"),
    ],
    ids=["chat", "responses", "messages"],
)
def test_every_endpoint_passes_all_three_checks_and_feeds_each_reply_back_in_its_own_shape(script, reply, key):
    provider: Final = _ScriptedProvider(
        CalledTurn(reply(tool_call=True), 0.004),
        [
            CalledTurn(reply(tool_call=False, text=_TURN_TWO_ANSWER, cached_tokens=4800), 0.0005),
            CalledTurn(reply(tool_call=False, text=_TURN_THREE_ANSWER, cached_tokens=4800), 0.0004),
        ],
    )

    result: Final = _run(script, provider, spend_rows={"call-1": 0.004, "call-2": 0.0005, "call-3": 0.0004})

    assert result.status == "success", result
    checks: Final = _checks(result)
    assert checks["conversation"] == (
        "pass",
        f'turn 1 called get_weather({{"location": "Boston"}}), turn 2 answered {_TURN_TWO_ANSWER!r}, '
        f"turn 3 answered {_TURN_THREE_ANSWER!r}",
    ), checks
    assert checks["prompt_caching"] == (
        "pass",
        "turn 2 read 4800 of 5000 input tokens from cache (96.0%), "
        "turn 3 read 4800 of 5000 input tokens from cache (96.0%)",
    ), checks
    assert checks["cost_tracking"] == (
        "pass",
        "turn 1 $0.004000, turn 2 $0.000500, turn 3 $0.000400, "
        "spend log rows written ($0.004000, $0.000500, $0.000400)",
    ), checks
    assert result.cache_read_tokens == 4800
    assert result.cache_read_percent == 96.0
    assert result.spend_usd == pytest.approx(0.0049)
    assert len(provider.sent) == 3
    turn_two: Final = provider.sent[1]
    turn_three: Final = provider.sent[2]
    sent_json: Final = json.dumps(provider.sent[1:])
    assert _TOOL_CALL_ID in sent_json and WEATHER_TOOL_RESULT in sent_json, sent_json
    if script is MESSAGES_SCRIPT:
        assert turn_two["messages"][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}, turn_two["messages"][
            -1
        ]
    else:
        assert "cache_control" not in sent_json and "cachePoint" not in sent_json, sent_json
    assert "<turn " not in sent_json, sent_json
    assert turn_three[key][: len(turn_two[key])] == turn_two[key], turn_three
    assert "Repeat the temperature" in json.dumps(turn_three[key][-2:]), turn_three[key][-2:]
    assert "<system-reminder>" in json.dumps(turn_three[key][-2:]), turn_three[key][-2:]
    if script is CHAT_COMPLETIONS_SCRIPT:
        tool_message: Final = turn_two["messages"][-1]
        assert tool_message == {"role": "tool", "tool_call_id": _TOOL_CALL_ID, "content": WEATHER_TOOL_RESULT}
        assert turn_two["messages"][-2]["tool_calls"][0]["id"] == _TOOL_CALL_ID
        assert turn_three["messages"][-2]["role"] == "system", turn_three
        assert turn_three["messages"][-3]["role"] == "assistant", turn_three
        assert turn_three["messages"][-3]["content"] == _TURN_TWO_ANSWER, turn_three
    elif script is RESPONSES_SCRIPT:
        assert turn_two["input"][-1] == {
            "type": "function_call_output",
            "call_id": _TOOL_CALL_ID,
            "output": WEATHER_TOOL_RESULT,
        }
        assert [item["type"] for item in turn_two["input"][1:-1]] == ["reasoning", "function_call"]
        assert turn_three["input"][-2]["role"] == "system", turn_three
        assert turn_three["input"][-3]["type"] == "message", turn_three
        assert turn_three["input"][-3]["content"][0]["text"] == _TURN_TWO_ANSWER, turn_three
    else:
        assert turn_two["messages"][-1]["content"] == [
            {
                "type": "tool_result",
                "tool_use_id": _TOOL_CALL_ID,
                "content": WEATHER_TOOL_RESULT,
                "cache_control": {"type": "ephemeral"},
            }
        ]
        assert turn_two["messages"][-2]["content"][0]["type"] == "thinking"
        assert [m["role"] for m in turn_three["messages"][-3:]] == ["assistant", "user", "system"], turn_three
        assert turn_three["messages"][-3]["content"] == [{"type": "text", "text": _TURN_TWO_ANSWER}], turn_three


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
        [
            CalledTurn(_chat_reply(tool_call=False, text="I cannot tell.", cached_tokens=4800), 0.0005),
            CalledTurn(_chat_reply(tool_call=False, text=_TURN_THREE_ANSWER, cached_tokens=4800), 0.0004),
        ],
    )

    checks: Final = _checks(_run(CHAT_COMPLETIONS_SCRIPT, provider))

    assert checks["conversation"] == ("fail", "turn 2 did not use the tool result '72F and sunny': 'I cannot tell.'")
    assert checks["prompt_caching"][0] == "pass"


def test_a_follow_up_answer_that_forgets_the_earlier_turn_fails_the_conversation():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [
            CalledTurn(_chat_reply(tool_call=False, text=_TURN_TWO_ANSWER, cached_tokens=4800), 0.0005),
            CalledTurn(_chat_reply(tool_call=False, text="I do not know the temperature.", cached_tokens=4800), 0.0004),
        ],
    )

    checks: Final = _checks(_run(CHAT_COMPLETIONS_SCRIPT, provider))

    assert checks["conversation"] == (
        "fail",
        "turn 3 did not carry the temperature from turn 2 into the follow-up answer: 'I do not know the temperature.'",
    )
    assert checks["prompt_caching"][0] == "pass"
    assert checks["cost_tracking"][0] == "pass"


def test_a_turn_three_provider_error_is_reported_as_the_conversation_failure():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [
            CalledTurn(_chat_reply(tool_call=False, text=_TURN_TWO_ANSWER, cached_tokens=4800), 0.0005),
            RuntimeError("400 invalid assistant message"),
        ],
    )

    checks: Final = _checks(_run(CHAT_COMPLETIONS_SCRIPT, provider))

    assert checks["conversation"] == (
        "fail",
        "turn 3 request with the follow-up question failed: RuntimeError: 400 invalid assistant message",
    )
    assert checks["prompt_caching"] == ("pass", "turn 2 read 4800 of 5000 input tokens from cache (96.0%)")


def test_zero_cached_tokens_is_retried_once_and_fails_when_the_retry_misses_too():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=0), 0.004),
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=0), 0.004),
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0004),
        ],
    )

    result: Final = _run(CHAT_COMPLETIONS_SCRIPT, provider)

    assert len(provider.sent) == 4
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
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=0), 0.004),
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=5000), 0.0004),
        ],
    )

    result: Final = _run(CHAT_COMPLETIONS_SCRIPT, provider)

    assert len(provider.sent) == 5
    assert _checks(result)["prompt_caching"] == (
        "pass",
        "turn 2 read 5000 of 5000 input tokens from cache (100.0%) on the second attempt, "
        "turn 3 read 5000 of 5000 input tokens from cache (100.0%) on the second attempt",
    )
    assert result.spend_usd == pytest.approx(0.0049)


def test_turn_three_reading_fewer_cached_tokens_than_turn_two_fails_prompt_caching():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0005),
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4600), 0.0005),
        ],
    )

    result: Final = _run(CHAT_COMPLETIONS_SCRIPT, provider)

    assert len(provider.sent) == 3
    status, detail = _checks(result)["prompt_caching"]
    assert status == "fail", detail
    assert "turn 3 read 4600 cached tokens, fewer than turn 2's 4800" in detail, detail


def test_a_cache_miss_on_turn_three_fails_prompt_caching_after_its_own_retry():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0005),
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=0), 0.004),
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=0), 0.004),
        ],
    )

    result: Final = _run(CHAT_COMPLETIONS_SCRIPT, provider)

    assert len(provider.sent) == 4
    assert _checks(result)["prompt_caching"] == (
        "fail",
        "turn 3 reported 0 cached tokens on 5000 input tokens in 2 attempts, the provider did not reuse the "
        "repeated system prompt",
    )
    assert _checks(result)["conversation"][0] == "pass"


def test_a_model_without_a_price_fails_cost_tracking_and_a_zero_priced_model_only_warns():
    unpriced: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), None),
        [CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), None)] * 2,
    )
    free: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.0),
        [CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0)] * 2,
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
        "turn 1 response_cost is $0, either this model is priced at zero or no price resolved for it (not in the "
        "cost map and no custom pricing on the deployment)",
    )
    assert free_result.status == "warning"
    assert overall_status([unpriced_result, free_result]) == "error"
    assert overall_status([free_result]) == "warning"


def test_a_missing_spend_log_row_fails_cost_tracking_even_when_the_response_was_priced():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0005),
            CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0004),
        ],
    )

    checks: Final = _checks(_run(CHAT_COMPLETIONS_SCRIPT, provider, spend_rows={"call-1": 0.004, "call-3": 0.0004}))

    assert checks["cost_tracking"] == (
        "fail",
        "turn 1 $0.004000, turn 2 $0.000500, turn 3 $0.000400, but no LiteLLM_SpendLogs row landed for "
        "request_id call-2",
    )


def test_a_response_that_is_not_the_endpoint_shape_fails_the_conversation_with_the_parse_error():
    provider: Final = _ScriptedProvider(CalledTurn({"id": "x", "object": "nonsense"}, 0.001), [])

    checks: Final = _checks(_run(MESSAGES_SCRIPT, provider))

    assert checks["conversation"][0] == "fail"
    assert checks["conversation"][1].startswith("turn 1 Messages API response did not parse: content"), checks


def test_payload_lists_the_checks_in_order_with_their_details():
    provider: Final = _ScriptedProvider(
        CalledTurn(_chat_reply(tool_call=True), 0.004),
        [CalledTurn(_chat_reply(tool_call=False, text="72F", cached_tokens=4800), 0.0005)] * 2,
    )

    payload: Final = endpoint_result_payload(_run(CHAT_COMPLETIONS_SCRIPT, provider)).model_dump()

    assert payload["endpoint"] == "/v1/chat/completions"
    assert payload["status"] == "success"
    assert [check["name"] for check in payload["checks"]] == ["conversation", "prompt_caching", "cost_tracking"]
    assert all(check["status"] == "pass" for check in payload["checks"])
    assert payload["cache_read_percent"] == 96.0
    assert isinstance(payload["duration_ms"], int)


def test_the_three_request_files_send_one_system_prompt_and_one_tool_so_endpoint_verdicts_are_comparable():
    chat, responses, messages = (REQUEST_FILES[name] for name in ("chat_completions", "responses", "messages"))

    assert set(REQUEST_FILES) == {"chat_completions", "messages", "responses"}
    for name, file in REQUEST_FILES.items():
        appended: Final = json.dumps(file.turn_2_appended_messages)
        assert WEATHER_TOOL_RESULT in appended, (name, appended)
        assert "<turn 1 response: assistant output>" in appended and "<turn 1 response: tool call id>" in appended, name
        third: Final = json.dumps(file.turn_3_appended_messages)
        assert "<turn 2 response: assistant output>" in third, (name, third)
        assert "<system-reminder>" in third, (name, third)
    assert chat.turn_1["messages"][0]["content"] == responses.turn_1["instructions"] == messages.turn_1["system"]
    assert len(messages.turn_1["system"]) > 20000
    assert messages.turn_1["messages"][-1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert chat.turn_1["tools"][0]["function"]["parameters"] == responses.turn_1["tools"][0]["parameters"]
    assert chat.turn_1["tools"][0]["function"]["parameters"] == messages.turn_1["tools"][0]["input_schema"]
    assert CHAT_COMPLETIONS_SCRIPT.turn_one() == chat.turn_1
    assert RESPONSES_SCRIPT.turn_one() == responses.turn_1
    assert MESSAGES_SCRIPT.turn_one() == messages.turn_1

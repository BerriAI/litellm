from typing import Final

from litellm.completion_extras.litellm_interactions_transformation import (
    transform_chat_to_interactions_request,
    transform_interactions_response_to_chat_response,
)
from litellm.types.utils import ChatCompletionMessageToolCall, Choices, ModelResponse, Usage


def test_thinking_tokens_and_cached_tokens_are_billed_in_chat_usage() -> None:
    raw: Final = {
        "id": "v1_int_think_1",
        "object": "interaction",
        "status": "completed",
        "steps": [
            {
                "type": "thought",
                "signature": "sig_thought_abc",
                "summary": [{"type": "text", "text": " difference of squares "}],
            },
            {
                "type": "model_output",
                "content": [
                    {
                        "type": "text",
                        "text": "1591",
                        "annotations": [
                            {
                                "start_index": 0,
                                "end_index": 4,
                                "uri": "https://example.org/math",
                                "title": "Math Ref",
                            }
                        ],
                    }
                ],
            },
        ],
        "usage": {
            "total_input_tokens": 62,
            "total_cached_tokens": 20,
            "total_output_tokens": 171,
            "total_thought_tokens": 297,
            "total_tokens": 530,
        },
    }

    resp: Final = transform_interactions_response_to_chat_response(
        raw_response=raw,
        model="gemini/gemini-3-flash-preview",
        model_response=ModelResponse(id="fallback_id"),
    )

    assert resp.id == "v1_int_think_1"
    assert resp.model == "gemini/gemini-3-flash-preview"
    choice: Final = resp.choices[0]
    assert isinstance(choice, Choices)
    assert choice.finish_reason == "stop"
    assert choice.message.content == "1591"
    assert choice.message.reasoning_content == " difference of squares "
    assert choice.message.thinking_blocks == [
        {"type": "thinking", "thinking": " difference of squares ", "signature": "sig_thought_abc"}
    ]
    assert choice.message.annotations == [
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://example.org/math",
                "title": "Math Ref",
                "start_index": 0,
                "end_index": 4,
            },
        }
    ]
    usage: Final = resp.model_extra.get("usage") if resp.model_extra else None
    assert isinstance(usage, Usage)
    assert usage.prompt_tokens == 62
    assert usage.completion_tokens == 171 + 297
    assert usage.total_tokens == 530
    assert usage.completion_tokens_details is not None
    assert usage.completion_tokens_details.reasoning_tokens == 297
    assert usage.prompt_tokens_details is not None
    assert usage.prompt_tokens_details.cached_tokens == 20


def test_stateless_openai_client_stripping_extra_fields_roundtrips_signature_cleanly() -> None:
    turn1_raw: Final = {
        "status": "requires_action",
        "steps": [
            {"type": "thought", "signature": "thought_sig_001"},
            {
                "type": "function_call",
                "id": "call_tokyo",
                "signature": "fc_sig_tokyo",
                "name": "get_weather",
                "arguments": {"city": "Tokyo"},
            },
            {
                "type": "function_call",
                "id": "call_paris",
                "name": "get_weather",
                "arguments": {"city": "Paris"},
            },
        ],
        "usage": {"total_input_tokens": 49, "total_output_tokens": 32, "total_tokens": 81},
    }

    turn1_resp: Final = transform_interactions_response_to_chat_response(
        raw_response=turn1_raw,
        model="gemini-3-flash-preview",
        model_response=ModelResponse(id="chatcmpl-stateless"),
    )
    assert turn1_resp.id == "chatcmpl-stateless"
    choice: Final = turn1_resp.choices[0]
    assert isinstance(choice, Choices)
    assert choice.finish_reason == "tool_calls"
    tool_calls: Final = choice.message.tool_calls or []
    fn_calls: Final = [tc for tc in tool_calls if isinstance(tc, ChatCompletionMessageToolCall)]
    assert len(fn_calls) == 2

    stripped_openai_assistant: Final = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name or "", "arguments": tc.function.arguments},
            }
            for tc in fn_calls
        ],
    }
    turn2_prepared: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Weather in Tokyo and Paris?"},
            stripped_openai_assistant,
            {"role": "tool", "tool_call_id": fn_calls[0].id, "content": '{"temp": 20}'},
            {"role": "tool", "tool_call_id": fn_calls[1].id, "content": '{"temp": 16}'},
        ],
        model="gemini-3-flash-preview",
        optional_params={},
        litellm_params={},
    )

    assert turn2_prepared.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Weather in Tokyo and Paris?"}]},
        {
            "type": "function_call",
            "id": "call_tokyo",
            "name": "get_weather",
            "arguments": {"city": "Tokyo"},
            "signature": "fc_sig_tokyo",
        },
        {
            "type": "function_call",
            "id": "call_paris",
            "name": "get_weather",
            "arguments": {"city": "Paris"},
        },
        {
            "type": "function_result",
            "call_id": "call_tokyo",
            "name": "get_weather",
            "result": [{"type": "text", "text": '{"temp": 20}'}],
        },
        {
            "type": "function_result",
            "call_id": "call_paris",
            "name": "get_weather",
            "result": [{"type": "text", "text": '{"temp": 16}'}],
        },
    ]

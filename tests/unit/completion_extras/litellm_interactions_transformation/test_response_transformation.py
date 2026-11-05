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
        {"type": "thought", "signature": "fc_sig_tokyo"},
        {
            "type": "function_call",
            "id": "call_tokyo",
            "name": "get_weather",
            "arguments": {"city": "Tokyo"},
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


def test_thought_step_signature_inherits_cleanly_when_function_call_omits_inline_signature() -> None:
    turn1_raw: Final = {
        "status": "requires_action",
        "steps": [
            {"type": "thought", "signature": "argon_thought_sig_base64"},
            {
                "type": "function_call",
                "id": "call_bash_1",
                "name": "bash",
                "arguments": {"cmd": "ls"},
            },
        ],
        "usage": {"total_input_tokens": 20, "total_output_tokens": 10, "total_tokens": 30},
    }

    turn1_resp: Final = transform_interactions_response_to_chat_response(
        raw_response=turn1_raw,
        model="gemini-4-argon",
        model_response=ModelResponse(id="chatcmpl-argon"),
    )
    choice: Final = turn1_resp.choices[0]
    assert isinstance(choice, Choices)
    fn_calls: Final = [tc for tc in (choice.message.tool_calls or []) if isinstance(tc, ChatCompletionMessageToolCall)]
    assert len(fn_calls) == 1
    assert fn_calls[0].id == "call_bash_1__thought__argon_thought_sig_base64"

    stripped_assistant: Final = {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": fn_calls[0].id,
                "type": "function",
                "function": {"name": fn_calls[0].function.name or "", "arguments": fn_calls[0].function.arguments},
            }
        ],
    }
    turn2_prepared: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "List files"},
            stripped_assistant,
            {"role": "tool", "tool_call_id": fn_calls[0].id, "content": "README.md\n"},
        ],
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
    )
    assert turn2_prepared.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "List files"}]},
        {"type": "thought", "signature": "argon_thought_sig_base64"},
        {
            "type": "function_call",
            "id": "call_bash_1",
            "name": "bash",
            "arguments": {"cmd": "ls"},
        },
        {
            "type": "function_result",
            "call_id": "call_bash_1",
            "name": "bash",
            "result": [{"type": "text", "text": "README.md\n"}],
        },
    ]


def test_preamble_and_parallel_tool_calls_roundtrip_without_inverting_order_or_duplicating_signature() -> None:
    turn1_steps: Final = [
        {"type": "thought", "signature": "argon_turn1_sig"},
        {"type": "model_output", "content": [{"type": "text", "text": "I will inspect the module and run pytest."}]},
        {"type": "function_call", "id": "call_view", "name": "bash", "arguments": {"cmd": "sed -n '1,40p' mod.py"}},
        {"type": "function_call", "id": "call_test", "name": "bash", "arguments": {"cmd": "pytest -q"}},
    ]
    turn1_resp: Final = transform_interactions_response_to_chat_response(
        raw_response={
            "id": "v1_turn1_argon",
            "status": "requires_action",
            "steps": turn1_steps,
            "usage": {
                "total_input_tokens": 120,
                "total_output_tokens": 45,
                "total_thought_tokens": 80,
                "total_tokens": 245,
            },
        },
        model="gemini-4-argon",
        model_response=ModelResponse(id="fallback"),
    )
    choice: Final = turn1_resp.choices[0]
    assert isinstance(choice, Choices)
    fn_calls: Final = [tc for tc in (choice.message.tool_calls or []) if isinstance(tc, ChatCompletionMessageToolCall)]
    assert len(fn_calls) == 2

    expected_turn2_steps: Final = [
        {"type": "user_input", "content": [{"type": "text", "text": "Fix the bug in mod.py"}]},
        *turn1_steps,
        {
            "type": "function_result",
            "call_id": "call_view",
            "name": "bash",
            "result": [{"type": "text", "text": "def f(): pass\n"}],
        },
        {
            "type": "function_result",
            "call_id": "call_test",
            "name": "bash",
            "result": [{"type": "text", "text": "1 failed\n"}],
        },
    ]

    preserved_turn2: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Fix the bug in mod.py"},
            choice.message,
            {"role": "tool", "tool_call_id": fn_calls[0].id, "content": "def f(): pass\n"},
            {"role": "tool", "tool_call_id": fn_calls[1].id, "content": "1 failed\n"},
        ],
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
    )
    assert preserved_turn2.input_steps == expected_turn2_steps

    stripped_assistant: Final = {
        "role": "assistant",
        "content": choice.message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name or "", "arguments": tc.function.arguments},
            }
            for tc in fn_calls
        ],
    }
    stripped_turn2: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Fix the bug in mod.py"},
            stripped_assistant,
            {"role": "tool", "tool_call_id": fn_calls[0].id, "content": "def f(): pass\n"},
            {"role": "tool", "tool_call_id": fn_calls[1].id, "content": "1 failed\n"},
        ],
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
    )
    assert stripped_turn2.input_steps == expected_turn2_steps


def test_interleaved_distinct_thought_signatures_roundtrip_across_preserved_and_stripped_clients() -> None:
    turn1_steps: Final = [
        {"type": "thought", "signature": "sig_step_a", "summary": [{"type": "text", "text": "first thought"}]},
        {"type": "model_output", "content": [{"type": "text", "text": "Checking both locations."}]},
        {"type": "function_call", "id": "call_a", "name": "bash", "arguments": {"cmd": "ls a"}},
        {"type": "thought", "signature": "sig_step_b", "summary": [{"type": "text", "text": "second thought"}]},
        {"type": "function_call", "id": "call_b", "name": "bash", "arguments": {"cmd": "ls b"}},
    ]
    turn1_resp: Final = transform_interactions_response_to_chat_response(
        raw_response={
            "id": "v1_interleaved",
            "status": "requires_action",
            "steps": turn1_steps,
            "usage": {"total_input_tokens": 30, "total_output_tokens": 20, "total_tokens": 50},
        },
        model="gemini-4-argon",
        model_response=ModelResponse(id="fallback"),
    )
    choice: Final = turn1_resp.choices[0]
    assert isinstance(choice, Choices)
    fn_calls: Final = [tc for tc in (choice.message.tool_calls or []) if isinstance(tc, ChatCompletionMessageToolCall)]
    assert len(fn_calls) == 2
    assert fn_calls[0].id == "call_a__thought__sig_step_a"
    assert fn_calls[1].id == "call_b__thought__sig_step_b"

    preserved_turn2: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Compare a and b"},
            choice.message,
            {"role": "tool", "tool_call_id": fn_calls[0].id, "content": "a1\n"},
            {"role": "tool", "tool_call_id": fn_calls[1].id, "content": "b1\n"},
        ],
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
    )
    assert preserved_turn2.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Compare a and b"}]},
        *turn1_steps,
        {"type": "function_result", "call_id": "call_a", "name": "bash", "result": [{"type": "text", "text": "a1\n"}]},
        {"type": "function_result", "call_id": "call_b", "name": "bash", "result": [{"type": "text", "text": "b1\n"}]},
    ]

    stripped_assistant: Final = {
        "role": "assistant",
        "content": choice.message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name or "", "arguments": tc.function.arguments},
            }
            for tc in fn_calls
        ],
    }
    stripped_turn2: Final = transform_chat_to_interactions_request(
        messages=[
            {"role": "user", "content": "Compare a and b"},
            stripped_assistant,
            {"role": "tool", "tool_call_id": fn_calls[0].id, "content": "a1\n"},
            {"role": "tool", "tool_call_id": fn_calls[1].id, "content": "b1\n"},
        ],
        model="gemini-4-argon",
        optional_params={},
        litellm_params={},
    )
    assert stripped_turn2.input_steps == [
        {"type": "user_input", "content": [{"type": "text", "text": "Compare a and b"}]},
        {"type": "thought", "signature": "sig_step_a"},
        {"type": "model_output", "content": [{"type": "text", "text": "Checking both locations."}]},
        {"type": "function_call", "id": "call_a", "name": "bash", "arguments": {"cmd": "ls a"}},
        {"type": "thought", "signature": "sig_step_b"},
        {"type": "function_call", "id": "call_b", "name": "bash", "arguments": {"cmd": "ls b"}},
        {"type": "function_result", "call_id": "call_a", "name": "bash", "result": [{"type": "text", "text": "a1\n"}]},
        {"type": "function_result", "call_id": "call_b", "name": "bash", "result": [{"type": "text", "text": "b1\n"}]},
    ]


def test_multiturn_steps_do_not_leak_signature_across_function_result_or_unsigned_thought() -> None:
    resp: Final = transform_interactions_response_to_chat_response(
        raw_response={
            "id": "v1_multiturn_noleak",
            "status": "requires_action",
            "steps": [
                {"type": "thought", "signature": "sig_turn_1"},
                {"type": "function_call", "id": "call_1", "name": "bash", "arguments": {"cmd": "ls"}},
                {
                    "type": "function_result",
                    "call_id": "call_1",
                    "name": "bash",
                    "result": [{"type": "text", "text": "ok"}],
                },
                {"type": "thought", "summary": [{"type": "text", "text": "unsigned second thought"}]},
                {"type": "function_call", "id": "call_2", "name": "bash", "arguments": {"cmd": "pwd"}},
                {"type": "thought", "signature": "sig_unused_before_unsigned"},
                {"type": "thought", "signature": ""},
                {"type": "function_call", "id": "call_3", "name": "bash", "arguments": {"cmd": "whoami"}},
            ],
            "usage": {"total_input_tokens": 20, "total_output_tokens": 10, "total_tokens": 30},
        },
        model="gemini-4-argon",
        model_response=ModelResponse(id="fallback"),
    )
    choice: Final = resp.choices[0]
    assert isinstance(choice, Choices)
    fn_calls: Final = [tc for tc in (choice.message.tool_calls or []) if isinstance(tc, ChatCompletionMessageToolCall)]
    assert [tc.id for tc in fn_calls] == ["call_1__thought__sig_turn_1", "call_2", "call_3"]
    assert [getattr(tc, "provider_specific_fields", None) for tc in fn_calls] == [
        {"thought_signature": "sig_turn_1"},
        None,
        None,
    ]

import json
from dataclasses import replace
from typing import Final

import pytest
from integration._support.client import JSON_OBJECT, Gateway, object_value
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from integration.translation.case import SSEFrame, StreamingTranslationTestCase
from integration.translation.chat_completions.bases.openai_responses import GPT_6_1_SOL_TEST_CASE
from integration.translation.chat_completions.cross_model.contracts import (
    Message,
    assert_completion,
    assert_final_text,
    assert_stream_preserved,
)
from integration.translation.response_scenarios import text_completion, two_messages_before_tools
from integration.translation.runner import (
    assert_provider_request,
    assert_translation,
    parse_sse_frames,
    streaming_reply,
)
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionChunk
from pydantic import JsonValue, TypeAdapter

OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])

GPT_6_1_SOL_TWO_MESSAGES_BEFORE_TOOLS_TEST_CASE: Final = two_messages_before_tools(GPT_6_1_SOL_TEST_CASE)
GPT_6_1_SOL_STRING_TOOL_OUTPUT_TEST_CASE: Final = replace(
    GPT_6_1_SOL_TEST_CASE,
    scenario="plain_string_tool_output",
    litellm_request={
        **GPT_6_1_SOL_TEST_CASE.litellm_request,
        "messages": [
            {"role": "user", "content": "Read the validation file."},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_contract_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"shop/validation.py"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_contract_1", "content": '{"result":"quantity must be positive"}'},
        ],
    },
    expected_provider_request={
        **{
            key: value
            for key, value in GPT_6_1_SOL_TEST_CASE.expected_provider_request.items()
            if key != "instructions"
        },
        "input": [
            {"role": "user", "content": [{"type": "input_text", "text": "Read the validation file."}]},
            {
                "type": "function_call",
                "call_id": "call_contract_1",
                "name": "read_file",
                "arguments": '{"path":"shop/validation.py"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_contract_1",
                "output": '{"result":"quantity must be positive"}',
            },
        ],
    },
)


def test_multiple_responses_messages_and_tools_produce_one_complete_choice(
    gateway: Gateway, provider: SharedProvider
) -> None:
    body: Final = assert_translation(GPT_6_1_SOL_TWO_MESSAGES_BEFORE_TOOLS_TEST_CASE, gateway, provider)
    assert_completion(body)


def test_chat_to_responses_keeps_plain_string_tool_outputs_as_strings(
    gateway: Gateway, provider: SharedProvider
) -> None:
    pytest.skip("BUG: LIT-9038 plain-string tool outputs are wrapped in a list that strict Responses backends reject")
    body: Final = assert_translation(GPT_6_1_SOL_STRING_TOOL_OUTPUT_TEST_CASE, gateway, provider)
    assert_completion(body)


def test_azure_deployment_capability_override_filters_temperature_even_with_base_model(
    gateway: Gateway, provider: SharedProvider
) -> None:
    with gateway.scenario() as scenario:
        alias: Final = scenario.model(
            model="azure/contract-deployment",
            api_base=provider.wire.url,
            api_key="synthetic-azure-key",
            api_version="2025-04-01-preview",
            base_model="azure/gpt-6.1-sol",
            drop_params=True,
            model_info={"supports_none_reasoning_effort": False},
        )
        provider.expect(Reply(body=json.dumps(text_completion("gpt-6.1-sol")).encode()))
        response: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": alias,
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 64,
                "temperature": 0.2,
                "use_chat_completions_api": True,
            },
        )
        assert_provider_request(
            provider.received(),
            "/openai/deployments/contract-deployment/chat/completions?api-version=2025-04-01-preview",
            {
                "content-type": "application/json",
                "api-key": "synthetic-azure-key",
                "authorization": "Bearer synthetic-azure-key",
            },
            {
                "model": "contract-deployment",
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_completion_tokens": 64,
                "stream": False,
            },
        )
        assert response.status_code == 200, response.text
        assert response.json() == {
            **text_completion(alias),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello."},
                    "finish_reason": "stop",
                    "provider_specific_fields": {},
                }
            ],
        }
        assert_final_text(JSON_OBJECT.validate_json(response.content))


def _cached_stream(missing_index: bool, terminal_choices: bool) -> StreamingTranslationTestCase:
    index: Final[dict[str, JsonValue]] = {} if missing_index else {"index": 0}
    envelope: Final[dict[str, JsonValue]] = {
        "id": "chatcmpl-cached-stream",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-6.1-sol",
    }
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": 1000,
        "completion_tokens": 100,
        "total_tokens": 1100,
        "prompt_tokens_details": {"cached_tokens": 600},
    }
    finish: Final[dict[str, JsonValue]] = {**index, "delta": {}, "finish_reason": "stop"}
    return StreamingTranslationTestCase(
        scenario="choice_bearing_cached_usage" if terminal_choices else "usage_only_cached_usage",
        litellm_endpoint="/v1/chat/completions",
        litellm_request={
            "model": "openai/gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 64,
            "stream": True,
            "stream_options": {"include_usage": True},
            "use_chat_completions_api": True,
            "cache": {"no-cache": True},
        },
        expected_provider_endpoint="/chat/completions",
        expected_provider_headers=GPT_6_1_SOL_TEST_CASE.expected_provider_headers,
        expected_provider_request={
            "model": "gpt-6.1-sol",
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_completion_tokens": 64,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        mock_provider_frames=(
            SSEFrame(
                data={
                    **envelope,
                    "choices": [{**index, "delta": {"role": "assistant", "content": "Hello."}, "finish_reason": None}],
                }
            ),
            *(() if terminal_choices else (SSEFrame(data={**envelope, "choices": [finish]}),)),
            SSEFrame(data={**envelope, "choices": [finish] if terminal_choices else [], "usage": usage}),
            SSEFrame(data="[DONE]"),
        ),
        expected_litellm_message={"role": "assistant", "content": "Hello."},
        expected_litellm_usage={
            "prompt_tokens": 1000,
            "completion_tokens": 100,
            "total_tokens": 1100,
            "prompt_tokens_details": {"cached_tokens": 600},
            "completion_tokens_details": {"reasoning_tokens": 0, "text_tokens": 100},
        },
    )


@pytest.mark.parametrize("missing_index", (False, True), ids=("indexed", "omitted_index"))
@pytest.mark.parametrize("terminal_choices", (False, True), ids=("usage_only", "choices_and_usage"))
def test_cached_usage_survives_raw_streaming_and_sdk_assembly(
    missing_index: bool, terminal_choices: bool, gateway: Gateway, provider: SharedProvider
) -> None:
    if missing_index:
        pytest.skip("BUG: LIT-9468 omitted upstream choice indexes reach the SDK as null and abort the stream")
    case: Final = _cached_stream(missing_index, terminal_choices)
    provider.expect(streaming_reply(case))
    response: Final = gateway.request("POST", case.litellm_endpoint, case.litellm_request)
    assert_provider_request(
        provider.received(),
        case.expected_provider_endpoint,
        case.expected_provider_headers,
        case.expected_provider_request,
    )
    assert response.status_code == 200, response.text
    frames: Final = parse_sse_frames(response.text)
    assert frames and tuple(frame.data == "[DONE]" for frame in frames) == (False,) * (len(frames) - 1) + (True,)
    chunks: Final = tuple(JSON_OBJECT.validate_python(frame.data) for frame in frames[:-1])
    state: Final = ChatCompletionStreamState()
    for chunk in chunks:
        assert chunk["object"] == "chat.completion.chunk", chunk
        for choice in OBJECTS.validate_python(chunk["choices"]):
            assert type(choice["index"]) is int and choice["index"] == 0, choice
        state.handle_chunk(ChatCompletionChunk.model_validate(chunk))
    body: Final = JSON_OBJECT.validate_json(state.get_final_completion().model_dump_json())
    assert_final_text(body)
    message: Final = Message.model_validate(object_value(OBJECTS.validate_python(body["choices"])[0]["message"]))
    usage: Final = object_value(body["usage"])
    assert_stream_preserved(message, usage, case)
    raw_usage: Final = tuple(chunk["usage"] for chunk in chunks if chunk.get("usage"))
    assert len(raw_usage) == 1, response.text
    assert_stream_preserved(message, object_value(raw_usage[0]), case)

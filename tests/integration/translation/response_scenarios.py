from collections.abc import Mapping
from dataclasses import replace
from typing import Final

from integration._support.client import object_value
from integration.translation.case import TranslationTestCase
from pydantic import JsonValue

COMMENTARY: Final = ("I found the validation entry point. ", "I will read it next.")


def text_completion(model: str) -> dict[str, JsonValue]:
    return {
        "id": "chatcmpl-contract-text",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello."}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }


MESSAGE_ITEMS: Final[tuple[Mapping[str, JsonValue], ...]] = tuple(
    {
        "type": "message",
        "id": f"msg_contract_{index}",
        "status": "completed",
        "role": "assistant",
        "phase": "commentary",
        "content": [{"type": "output_text", "text": text, "annotations": [], "logprobs": []}],
    }
    for index, text in enumerate(COMMENTARY)
)


def items(value: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    assert isinstance(value, list), value
    return tuple(object_value(item) for item in value)


def two_messages_before_tools(case: TranslationTestCase) -> TranslationTestCase:
    provider_tools: Final = tuple(
        item for item in items(case.mock_provider_response["output"]) if item["type"] == "function_call"
    )
    assert provider_tools, case.id
    return replace(
        case,
        scenario="two_messages_before_tools",
        mock_provider_response={
            **case.mock_provider_response,
            "output": [*(dict(item) for item in MESSAGE_ITEMS), *provider_tools],
        },
        expected_litellm_response=_messages_then_tools_response(case),
    )


def _messages_then_tools_response(case: TranslationTestCase) -> Mapping[str, JsonValue]:
    expected: Final = case.expected_litellm_response
    if case.litellm_endpoint == "/v1/chat/completions":
        choice: Final = items(expected["choices"])[0]
        return {
            **expected,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "".join(COMMENTARY),
                        "tool_calls": object_value(choice["message"])["tool_calls"],
                    },
                }
            ],
        }
    if case.litellm_endpoint == "/v1/messages":
        tools: Final = tuple(item for item in items(expected["content"]) if item["type"] == "tool_use")
        return {**expected, "content": [*({"type": "text", "text": text} for text in COMMENTARY), *tools]}
    assert case.litellm_endpoint == "/v1/responses", case.litellm_endpoint
    calls: Final = tuple(item for item in items(expected["output"]) if item["type"] == "function_call")
    return {**expected, "output": [*(dict(item) for item in MESSAGE_ITEMS), *calls]}

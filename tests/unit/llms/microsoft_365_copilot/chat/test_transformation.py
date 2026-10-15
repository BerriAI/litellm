from copy import deepcopy
from typing import Final, Protocol, cast

import pytest
from pydantic import TypeAdapter

import litellm
from litellm.llms.anthropic.pass_through.adapters.transformation import LiteLLMAnthropicMessagesAdapter
from litellm.llms.microsoft_365_copilot.chat.transformation import (
    Microsoft365CopilotChatConfig,
    build_chat_request,
    extract_graph_error_message,
    map_graph_response,
)
from litellm.llms.microsoft_365_copilot.common_utils import Microsoft365CopilotError
from litellm.types.llms.anthropic import AllAnthropicPassThroughMessageValues
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import LlmProviders, Usage
from litellm.utils import ProviderConfigManager, token_counter


class _ModelResponseWithUsage(Protocol):
    usage: Usage


def test_build_request_preserves_messages_and_maps_history() -> None:
    message_values: Final[list[AllMessageValues]] = cast(
        list[AllMessageValues],
        [
            {"role": "system", "content": "system context"},
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "second"},
                    {"type": "text", "text": "question", "cache_control": {"type": "ephemeral"}},
                ],
            },
        ],
    )
    original_messages: Final = deepcopy(message_values)

    request: Final = build_chat_request(messages=message_values, optional_params={})

    assert request == {
        "message": {"text": "second\nquestion"},
        "additionalContext": [
            {"text": "system context", "description": "system message"},
            {"text": "first question", "description": "user message"},
            {"text": "first answer", "description": "assistant message"},
        ],
        "locationHint": {"timeZone": "UTC"},
    }
    assert message_values == original_messages


def test_build_request_uses_provider_time_zone_and_omits_empty_history() -> None:
    messages: Final[list[AllMessageValues]] = TypeAdapter(list[AllMessageValues]).validate_python(
        [{"role": "user", "content": "hello"}]
    )

    request: Final = build_chat_request(
        messages=messages,
        optional_params={"time_zone": "America/New_York"},
    )

    assert request == {
        "message": {"text": "hello"},
        "locationHint": {"timeZone": "America/New_York"},
    }


def test_build_request_uses_last_user_message_when_assistant_is_last() -> None:
    messages: Final[list[AllMessageValues]] = TypeAdapter(list[AllMessageValues]).validate_python(
        [
            {"role": "user", "content": "the prompt"},
            {"role": "assistant", "content": "the response"},
        ]
    )

    request: Final = build_chat_request(messages=messages, optional_params={})

    assert request == {
        "message": {"text": "the prompt"},
        "additionalContext": [{"text": "the response", "description": "assistant message"}],
        "locationHint": {"timeZone": "UTC"},
    }


def test_build_request_uses_last_user_message_before_trailing_system() -> None:
    environment: Final = "# Environment\nYou have been invoked in the following environment: ..."
    messages: Final[list[AllMessageValues]] = TypeAdapter(list[AllMessageValues]).validate_python(
        [
            {"role": "user", "content": "whats 1+1"},
            {"role": "system", "content": environment},
        ]
    )

    request: Final = build_chat_request(messages=messages, optional_params={})

    assert request == {
        "message": {"text": "whats 1+1"},
        "additionalContext": [{"text": environment, "description": "system message"}],
        "locationHint": {"timeZone": "UTC"},
    }


def test_build_request_preserves_context_order_after_last_user_message() -> None:
    messages: Final[list[AllMessageValues]] = TypeAdapter(list[AllMessageValues]).validate_python(
        [
            {"role": "system", "content": "system A"},
            {"role": "user", "content": "question one"},
            {"role": "assistant", "content": "response one"},
            {"role": "user", "content": "question two"},
            {"role": "system", "content": "system B"},
        ]
    )

    request: Final = build_chat_request(messages=messages, optional_params={})

    assert request == {
        "message": {"text": "question two"},
        "additionalContext": [
            {"text": "system A", "description": "system message"},
            {"text": "question one", "description": "user message"},
            {"text": "response one", "description": "assistant message"},
            {"text": "system B", "description": "system message"},
        ],
        "locationHint": {"timeZone": "UTC"},
    }


def test_build_request_requires_a_user_message() -> None:
    messages: Final[list[AllMessageValues]] = TypeAdapter(list[AllMessageValues]).validate_python(
        [{"role": "assistant", "content": "not a prompt"}]
    )

    with pytest.raises(Microsoft365CopilotError) as error:
        build_chat_request(messages=messages, optional_params={})

    assert error.value.status_code == 400
    assert str(error.value) == "at least one message must have role 'user'"


def test_anthropic_adapter_preserves_trailing_system_context_for_m365() -> None:
    environment: Final = "# Environment\nYou have been invoked in the following environment: ..."
    desktop_messages: Final[list[AllAnthropicPassThroughMessageValues]] = cast(
        list[AllAnthropicPassThroughMessageValues],
        [
            {"role": "user", "content": "whats 1+1"},
            {"role": "system", "content": [{"type": "text", "text": environment}]},
        ],
    )
    translated_messages: Final = TypeAdapter(list[AllMessageValues]).validate_python(
        LiteLLMAnthropicMessagesAdapter().translate_anthropic_messages_to_openai(
            desktop_messages,
            model="microsoft_365_copilot/chat",
            custom_llm_provider="microsoft_365_copilot",
        )
    )

    request: Final = build_chat_request(messages=translated_messages, optional_params={})

    assert request == {
        "message": {"text": "whats 1+1"},
        "additionalContext": [{"text": environment, "description": "system message"}],
        "locationHint": {"timeZone": "UTC"},
    }


def test_build_request_rejects_non_text_content() -> None:
    messages: Final[list[AllMessageValues]] = cast(
        list[AllMessageValues],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                ],
            }
        ],
    )

    with pytest.raises(Microsoft365CopilotError) as error:
        build_chat_request(messages=messages, optional_params={})

    assert error.value.status_code == 400
    assert str(error.value) == "only text content is supported"


def test_map_graph_response_uses_last_message_and_estimates_usage() -> None:
    messages: Final = TypeAdapter(tuple[AllMessageValues, ...]).validate_python(
        [{"role": "user", "content": "known Copilot prompt"}]
    )
    reply: Final = "Copilot reply"
    graph_response: Final = {
        "messages": [
            {"text": "prompt echo"},
            {"text": reply},
        ]
    }

    response: Final = map_graph_response(
        graph_response=graph_response,
        model="microsoft_365_copilot/chat",
        messages=messages,
    )

    assert response.model == "microsoft_365_copilot/chat"
    assert response.choices[0].message.content == "Copilot reply"
    assert response.choices[0].message.role == "assistant"
    assert response.choices[0].finish_reason == "stop"
    response_with_usage: Final = cast(_ModelResponseWithUsage, response)
    expected_prompt_tokens: Final = token_counter(
        model="microsoft_365_copilot/chat",
        messages=messages,
    )
    expected_completion_tokens: Final = token_counter(
        model="microsoft_365_copilot/chat",
        text=reply,
        count_response_tokens=True,
    )
    assert expected_prompt_tokens > 0
    assert expected_completion_tokens > 0
    assert response_with_usage.usage.prompt_tokens == expected_prompt_tokens
    assert response_with_usage.usage.completion_tokens == expected_completion_tokens
    assert response_with_usage.usage.total_tokens == expected_prompt_tokens + expected_completion_tokens
    single_message_response: Final = map_graph_response(
        graph_response={"messages": [{"text": "single reply"}]},
        model="microsoft_365_copilot/chat",
        messages=messages,
    )
    assert single_message_response.choices[0].message.content == "single reply"


@pytest.mark.parametrize(
    ("reply", "expected_reply"),
    [
        ("pongpong", "pong"),
        ("1 + 1 = **2**.1 + 1 = **2**.", "1 + 1 = **2**."),
        ("pong", "pong"),
        ("pongpon", "pongpon"),
        ("pongPONG", "pongPONG"),
        ("", ""),
        ("aaaa", "aa"),
    ],
)
def test_map_graph_response_collapses_only_exact_duplicate_halves(reply: str, expected_reply: str) -> None:
    messages: Final = TypeAdapter(tuple[AllMessageValues, ...]).validate_python(
        [{"role": "user", "content": "known Copilot prompt"}]
    )

    response: Final = map_graph_response(
        graph_response={"messages": [{"text": reply}]},
        model="microsoft_365_copilot/chat",
        messages=messages,
    )

    assert response.choices[0].message.content == expected_reply


def test_map_graph_response_counts_tokens_for_collapsed_reply() -> None:
    messages: Final = TypeAdapter(tuple[AllMessageValues, ...]).validate_python(
        [{"role": "user", "content": "known Copilot prompt"}]
    )
    doubled_reply_response: Final = map_graph_response(
        graph_response={"messages": [{"text": "pongpong"}]},
        model="microsoft_365_copilot/chat",
        messages=messages,
    )
    single_reply_response: Final = map_graph_response(
        graph_response={"messages": [{"text": "pong"}]},
        model="microsoft_365_copilot/chat",
        messages=messages,
    )

    doubled_reply_with_usage: Final = cast(_ModelResponseWithUsage, doubled_reply_response)
    single_reply_with_usage: Final = cast(_ModelResponseWithUsage, single_reply_response)

    assert doubled_reply_with_usage.usage.completion_tokens == single_reply_with_usage.usage.completion_tokens


@pytest.mark.parametrize(
    "graph_response",
    [
        {},
        {"messages": []},
        {"messages": [{"text": "prompt"}, {}]},
    ],
)
def test_map_graph_response_rejects_missing_reply_text(graph_response: object) -> None:
    with pytest.raises(Microsoft365CopilotError) as error:
        map_graph_response(
            graph_response=graph_response,
            model="microsoft_365_copilot/chat",
            messages=(),
        )

    assert error.value.status_code == 502


def test_extract_graph_error_message_uses_final_stringified_message() -> None:
    graph_error: Final = {
        "error": {"message": ('{"messages":[{"text":"prompt echo"},{"text":"Copilot requires a valid license"}]}')}
    }

    message: Final = extract_graph_error_message(graph_error)
    empty_reply_error: Final = extract_graph_error_message({"error": {"message": '{"messages":[{"text":""}]}'}})

    assert message == "Copilot requires a valid license"
    assert empty_reply_error == ""


def test_provider_config_accepts_openai_compatibility_params() -> None:
    config: Final = Microsoft365CopilotChatConfig()

    assert config.get_supported_openai_params("microsoft_365_copilot/chat") == [
        "stream",
        "max_tokens",
        "max_completion_tokens",
    ]


def test_provider_is_registered_for_model_resolution_and_chat_config() -> None:
    _, provider, _, _ = litellm.get_llm_provider(model="microsoft_365_copilot/chat")
    config: Final = ProviderConfigManager.get_provider_chat_config(
        model="microsoft_365_copilot/chat",
        provider=LlmProviders.MICROSOFT_365_COPILOT,
    )

    assert provider == LlmProviders.MICROSOFT_365_COPILOT.value
    assert LlmProviders.MICROSOFT_365_COPILOT in litellm.provider_list
    assert isinstance(config, Microsoft365CopilotChatConfig)

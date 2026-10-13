"""
`merge_system_message` hoists every system message to the front of `messages` and joins
them into one leading system message before the provider call. vLLM and other strict
OpenAI-compatible servers reject a system message anywhere else with HTTP 400.

The flag is opt-in, so it has to be invisible to the provider: neither the request body
nor `extra_body` may carry it.
"""

import asyncio
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import litellm
from litellm.types.utils import all_litellm_params
from litellm.utils import get_non_default_completion_params

CONVERSATION: Final = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"},
    {"role": "system", "content": "Be terse."},
    {"role": "system", "content": "Answer in English."},
]


def _mock_openai_client() -> MagicMock:
    response = MagicMock()
    response.model_dump.return_value = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }

    raw_response = MagicMock()
    raw_response.headers = {}
    raw_response.parse.return_value = response

    client = MagicMock()
    client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw_response)
    return client


def _sent_messages(client: MagicMock) -> list[dict]:
    create_kwargs = client.chat.completions.with_raw_response.create.call_args.kwargs
    return create_kwargs["messages"]


def _acompletion(client: MagicMock, **extra_params: object) -> MagicMock:
    return asyncio.run(
        litellm.acompletion(
            model="openai/gpt-4o-mini",
            messages=[dict(message) for message in CONVERSATION],
            api_key="sk-test",
            client=client,
            **extra_params,
        )
    )


def test_merge_system_message_is_a_known_litellm_param():
    assert "merge_system_message" in all_litellm_params


def test_merge_system_message_not_forwarded_as_provider_param():
    forwarded = get_non_default_completion_params({"merge_system_message": True, "temperature": 0.5})

    assert "merge_system_message" not in forwarded


def test_hoists_and_joins_system_messages_when_flag_is_set():
    client = _mock_openai_client()

    _acompletion(client, merge_system_message=True)

    assert _sent_messages(client) == [
        {"role": "system", "content": "Be terse.\n\nAnswer in English."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]


def test_sends_messages_untouched_without_the_flag():
    client = _mock_openai_client()

    _acompletion(client)

    assert _sent_messages(client) == CONVERSATION


def test_sends_messages_untouched_when_flag_is_false():
    client = _mock_openai_client()

    _acompletion(client, merge_system_message=False)

    assert _sent_messages(client) == CONVERSATION


def test_flag_is_not_leaked_into_provider_request_body():
    client = _mock_openai_client()

    _acompletion(client, merge_system_message=True)

    create_kwargs = client.chat.completions.with_raw_response.create.call_args.kwargs
    assert "merge_system_message" not in create_kwargs
    assert "merge_system_message" not in (create_kwargs.get("extra_body") or {})

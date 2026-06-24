"""
The OpenAI chat handler sends the request via client.post(body=data) instead of
chat.completions.create(**data), skipping the SDK's async_maybe_transform whose
per-field typing introspection over the message-param union dominates CPU on large
multi-message requests. That is only correct if the transform is a no-op for the
chat-completion request body. These tests lock that assumption and the request-option
extraction so a future SDK change can never silently corrupt outbound requests.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, os.path.abspath("../../../.."))

from openai._constants import RAW_RESPONSE_HEADER
from openai._utils import async_maybe_transform
from openai.types.chat import completion_create_params

from litellm.llms.openai.openai import OpenAIChatCompletion

REPRESENTATIVE_BODY = {
    "model": "gpt-4o",
    "messages": [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "result"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
            ],
        },
    ],
    "temperature": 0.5,
    "max_tokens": 64,
    "tools": [
        {
            "type": "function",
            "function": {"name": "f", "parameters": {"type": "object"}},
        }
    ],
    "response_format": {"type": "json_object"},
    "stream": False,
}


async def test_bypass_body_matches_sdk_transform():
    transformed = await async_maybe_transform(
        dict(REPRESENTATIVE_BODY),
        completion_create_params.CompletionCreateParamsNonStreaming,
    )
    kwargs = OpenAIChatCompletion._chat_completion_post_kwargs(
        dict(REPRESENTATIVE_BODY), timeout=30.0
    )
    assert kwargs["body"] == transformed


def test_bypass_extracts_request_options_and_sets_raw_header():
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_headers": {"x-custom": "1"},
        "extra_body": {"vendor_flag": True},
        "extra_query": {"q": "1"},
    }
    kwargs = OpenAIChatCompletion._chat_completion_post_kwargs(dict(data), timeout=12.0)
    body = kwargs["body"]
    options = kwargs["options"]

    for k in ("extra_headers", "extra_body", "extra_query"):
        assert k not in body
    assert options["headers"][RAW_RESPONSE_HEADER] == "true"
    assert options["headers"]["x-custom"] == "1"
    assert options["extra_json"] == {"vendor_flag": True}
    assert options["params"] == {"q": "1"}
    assert options["timeout"] == 12.0


async def test_async_request_posts_to_chat_completions_and_returns_headers():
    openai_chat = OpenAIChatCompletion()
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"choices": []}
    mock_raw = MagicMock()
    mock_raw.headers = {"x-request-id": "abc"}
    mock_raw.parse.return_value = mock_response

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_raw)

    headers, response = await openai_chat.make_openai_chat_completion_request(
        openai_aclient=mock_client,
        data={"messages": [{"role": "user", "content": "hi"}]},
        timeout=30,
        logging_obj=MagicMock(),
    )

    assert response is mock_response
    assert headers == {"x-request-id": "abc"}
    assert mock_client.post.await_args.args[0] == "/chat/completions"
    assert mock_client.post.await_args.kwargs["body"]["messages"] == [
        {"role": "user", "content": "hi"}
    ]


def test_bypass_keeps_stream_in_body_and_flags_post():
    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }
    kwargs = OpenAIChatCompletion._chat_completion_post_kwargs(dict(data), timeout=30.0)
    assert kwargs["body"]["stream"] is True
    assert kwargs["stream"] is True

    no_stream = OpenAIChatCompletion._chat_completion_post_kwargs(
        {"model": "gpt-4o", "messages": []}, timeout=30.0
    )
    assert no_stream["stream"] is False


def test_bypass_options_match_sdk_make_request_options():
    from openai._base_client import make_request_options

    data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "extra_headers": {"x-custom": "1"},
        "extra_body": {"vendor_flag": True},
        "extra_query": {"q": "1"},
    }
    options = OpenAIChatCompletion._chat_completion_post_kwargs(
        dict(data), timeout=7.0
    )["options"]
    expected = make_request_options(
        extra_headers={RAW_RESPONSE_HEADER: "true", "x-custom": "1"},
        extra_query={"q": "1"},
        extra_body={"vendor_flag": True},
        timeout=7.0,
    )
    assert options == expected


def _mock_client_with_both_paths():
    raw = MagicMock()
    raw.headers = {}
    raw.parse.return_value = MagicMock()
    client = MagicMock()
    client.post.return_value = raw
    client.chat.completions.with_raw_response.create.return_value = raw
    return client


def test_escape_hatch_uses_create_when_flag_disabled(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "skip_openai_chat_transform", False)
    client = _mock_client_with_both_paths()
    OpenAIChatCompletion().make_sync_openai_chat_completion_request(
        openai_client=client,
        data={"messages": [{"role": "user", "content": "hi"}]},
        timeout=30,
        logging_obj=MagicMock(),
    )
    client.chat.completions.with_raw_response.create.assert_called_once()
    client.post.assert_not_called()


def test_falls_back_to_create_when_bypass_symbols_unavailable(monkeypatch):
    from litellm.llms.openai import openai as openai_mod

    monkeypatch.setattr(openai_mod, "_OPENAI_CHAT_TRANSFORM_BYPASS_AVAILABLE", False)
    client = _mock_client_with_both_paths()
    openai_mod.OpenAIChatCompletion().make_sync_openai_chat_completion_request(
        openai_client=client,
        data={"messages": [{"role": "user", "content": "hi"}]},
        timeout=30,
        logging_obj=MagicMock(),
    )
    client.chat.completions.with_raw_response.create.assert_called_once()
    client.post.assert_not_called()


async def test_async_escape_hatch_uses_create_when_flag_disabled(monkeypatch):
    import litellm

    monkeypatch.setattr(litellm, "skip_openai_chat_transform", False)
    raw = MagicMock()
    raw.headers = {}
    raw.parse.return_value = MagicMock()
    client = MagicMock()
    client.post = AsyncMock(return_value=raw)
    client.chat.completions.with_raw_response.create = AsyncMock(return_value=raw)
    await OpenAIChatCompletion().make_openai_chat_completion_request(
        openai_aclient=client,
        data={"messages": [{"role": "user", "content": "hi"}]},
        timeout=30,
        logging_obj=MagicMock(),
    )
    client.chat.completions.with_raw_response.create.assert_awaited_once()
    client.post.assert_not_awaited()

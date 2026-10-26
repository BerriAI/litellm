import asyncio
import json
from collections.abc import Callable, Iterator
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm import acompletion, completion
from litellm.caching.llm_caching_handler import LLMClientCache
from litellm.llms.cloudflare.chat.transformation import CloudflareChatConfig
from unittest.mock import AsyncMock, patch
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from typing import Any, Dict

FAKE_API_BASE = "https://fake-cloudflare.example.com/client/v4/accounts/fake-acct/ai/v1"
FAKE_API_KEY = "fake-cf-api-key"


@pytest.fixture
def _cloudflare_httpx_transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    client_cache: Final = LLMClientCache()
    monkeypatch.setattr(litellm, "in_memory_llm_clients_cache", client_cache)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "force_ipv4", False)
    monkeypatch.setattr(litellm, "sync_transport", None, raising=False)
    yield
    client_cache.flush_cache()


def test_supported_params_include_tools_and_tool_choice():
    config = CloudflareChatConfig()

    params = config.get_supported_openai_params(model="@cf/meta/llama-2-7b-chat-int8")

    assert "tools" in params
    assert "tool_choice" in params
    assert "stream" in params
    assert "max_tokens" in params


def test_get_complete_url_defaults_to_openai_compatible_endpoint(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base=None,
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"
    assert "/ai/run/" not in url


def test_get_complete_url_appends_chat_completions_to_explicit_base():
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/v1",
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"
    assert "/ai/run/" not in url


def test_get_complete_url_is_idempotent_for_full_base():
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions",
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"


def test_get_complete_url_falls_back_to_account_id_when_base_is_empty(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acct")
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base="",
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"


def test_get_complete_url_raises_when_account_id_and_base_missing(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    config = CloudflareChatConfig()

    with pytest.raises(ValueError, match="Missing CLOUDFLARE_ACCOUNT_ID"):
        config.get_complete_url(
            api_base=None,
            api_key="cf-key",
            model="@cf/meta/llama-2-7b-chat-int8",
            optional_params={},
            litellm_params={},
        )


def test_get_complete_url_raises_when_account_id_is_empty(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "   ")
    config = CloudflareChatConfig()

    with pytest.raises(ValueError, match="Missing CLOUDFLARE_ACCOUNT_ID"):
        config.get_complete_url(
            api_base=None,
            api_key="cf-key",
            model="@cf/meta/llama-2-7b-chat-int8",
            optional_params={},
            litellm_params={},
        )


def test_get_complete_url_migrates_legacy_ai_run_base():
    config = CloudflareChatConfig()

    url = config.get_complete_url(
        api_base="https://api.cloudflare.com/client/v4/accounts/acct/ai/run/",
        api_key="cf-key",
        model="@cf/meta/llama-2-7b-chat-int8",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://api.cloudflare.com/client/v4/accounts/acct/ai/v1/chat/completions"
    assert "/ai/run" not in url


def test_transform_request_passes_tools_through_in_openai_format():
    config = CloudflareChatConfig()
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "weather in nyc?"}]

    body = config.transform_request(
        model="@cf/meta/llama-2-7b-chat-int8",
        messages=messages,
        optional_params={"tools": tools, "tool_choice": "auto"},
        litellm_params={},
        headers={},
    )

    assert body["messages"] == messages
    assert body["model"] == "@cf/meta/llama-2-7b-chat-int8"
    assert body["tools"] == tools
    assert body["tool_choice"] == "auto"


def test_validate_environment_requires_api_key():
    config = CloudflareChatConfig()

    with pytest.raises(ValueError, match="Missing Cloudflare API Key"):
        config.validate_environment(
            headers={},
            model="@cf/meta/llama-2-7b-chat-int8",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )


def test_validate_environment_sets_bearer_and_content_type():
    config = CloudflareChatConfig()

    headers = config.validate_environment(
        headers={},
        model="@cf/meta/llama-2-7b-chat-int8",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="cf-key",
    )

    assert headers["Authorization"] == "Bearer cf-key"
    assert headers["Content-Type"] == "application/json"


def _chat_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-cf",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "@cf/meta/llama-2-7b-chat-int8",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "I am a large language model created to assist you."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 11, "total_tokens": 19},
    }


def _tool_call_response() -> dict[str, object]:
    return {
        "id": "chatcmpl-cf-tools",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "@cf/meta/llama-2-7b-chat-int8",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "New York"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 9, "total_tokens": 29},
    }


def _streaming_chunks() -> tuple[str, ...]:
    base: Final = {
        "id": "chatcmpl-cf",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "@cf/meta/llama-2-7b-chat-int8",
    }
    return (
        json.dumps({**base, "choices": [{"index": 0, "delta": {"content": "I am"}}]}),
        json.dumps({**base, "choices": [{"index": 0, "delta": {"content": " a language"}}]}),
        json.dumps({**base, "choices": [{"index": 0, "delta": {"content": " model."}, "finish_reason": "stop"}]}),
    )


def _mock_post_response(mock_post: MagicMock, response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    def _respond(request: httpx.Request) -> httpx.Response:
        mock_post(request)
        return response

    return _respond


@pytest.mark.parametrize("sync_mode", [True, False])
def test_completion_cloudflare(sync_mode):
    messages = [{"role": "user", "content": "what llm are you"}]
    mock_resp = _make_mock_response(_chat_response())

    if sync_mode:
        with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
            response = completion(
                model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                messages=messages,
                max_tokens=15,
                api_base=FAKE_API_BASE,
                api_key=FAKE_API_KEY,
            )
            mock_post.assert_called_once()
    else:
        with patch.object(
            AsyncHTTPHandler, "post", new_callable=AsyncMock, return_value=mock_resp
        ) as mock_post:
            response = asyncio.run(
                acompletion(
                    model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                    messages=messages,
                    max_tokens=15,
                    api_base=FAKE_API_BASE,
                    api_key=FAKE_API_KEY,
                )
            )
            mock_post.assert_called_once()

    assert response is not None
    assert response.choices[0].message.content is not None
    assert "language model" in response.choices[0].message.content.lower()

    called_url = mock_post.call_args.kwargs.get("url") or mock_post.call_args.args[0]
    assert called_url.endswith("/ai/v1/chat/completions")
    assert "/ai/run/" not in called_url


def test_completion_cloudflare_tool_calls_sent_to_openai_endpoint():
    messages = [{"role": "user", "content": "weather in New York?"}]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    mock_resp = _make_mock_response(_tool_call_response())

    with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
        response = completion(
            model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            api_base=FAKE_API_BASE,
            api_key=FAKE_API_KEY,
        )
        mock_post.assert_called_once()

    sent_body = json.loads(mock_post.call_args.kwargs["data"])
    assert sent_body["tools"] == tools
    assert sent_body["tool_choice"] == "auto"

    assert response.choices[0].finish_reason == "tool_calls"
    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None and len(tool_calls) == 1
    assert tool_calls[0].function.name == "get_weather"


@pytest.mark.parametrize("sync_mode", [True])
def test_completion_cloudflare_stream(sync_mode: bool, _cloudflare_httpx_transport: None) -> None:
    messages: Final = [{"role": "user", "content": "what llm are you"}]
    raw_chunks: Final = _streaming_chunks()
    body: Final = "".join(f"data: {chunk}\n\n" for chunk in raw_chunks) + "data: [DONE]\n\n"
    with respx.mock(assert_all_called=True) as api:
        mock_post: Final = MagicMock()
        api.post(f"{FAKE_API_BASE}/chat/completions").mock(
            side_effect=_mock_post_response(
                mock_post,
                httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    content=body,
                ),
            )
        )
        response: Final = completion(
            model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
            messages=messages,
            max_tokens=15,
            stream=sync_mode,
            api_base=FAKE_API_BASE,
            api_key=FAKE_API_KEY,
        )
        chunks_received: Final = tuple(response)
        mock_post.assert_called_once()
    assert len(chunks_received) > 0
    content: Final = "".join((c.choices[0].delta.content for c in chunks_received if c.choices[0].delta.content))
    assert "language" in content.lower()


def _make_mock_response(json_data: Dict[str, Any]) -> MagicMock:
    mock = MagicMock(spec=httpx.Response)
    mock.status_code = 200
    mock.headers = {"content-type": "application/json"}
    mock.json.return_value = json_data
    mock.text = json.dumps(json_data)
    return mock


@pytest.mark.parametrize("sync_mode", [False])
def test_acompletion_cloudflare_stream(sync_mode, monkeypatch):
    monkeypatch.setattr(litellm, "disable_hf_tokenizer_download", True)
    messages = [{"role": "user", "content": "what llm are you"}]
    raw_chunks = _streaming_chunks()

    if sync_mode:

        def _iter_lines():
            for chunk in raw_chunks:
                yield f"data: {chunk}"
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.iter_lines.return_value = _iter_lines()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        with patch.object(HTTPHandler, "post", return_value=mock_resp) as mock_post:
            response = completion(
                model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                messages=messages,
                max_tokens=15,
                stream=True,
                api_base=FAKE_API_BASE,
                api_key=FAKE_API_KEY,
            )
            chunks_received = list(response)
            mock_post.assert_called_once()
    else:

        async def _aiter_lines():
            for chunk in raw_chunks:
                yield f"data: {chunk}"
            yield "data: [DONE]"

        mock_resp = MagicMock()
        mock_resp.aiter_lines.return_value = _aiter_lines()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "text/event-stream"}

        async def _run():
            with patch.object(AsyncHTTPHandler, "post", new_callable=AsyncMock, return_value=mock_resp) as mock_post:
                resp = await acompletion(
                    model="cloudflare/@cf/meta/llama-2-7b-chat-int8",
                    messages=messages,
                    max_tokens=15,
                    stream=True,
                    api_base=FAKE_API_BASE,
                    api_key=FAKE_API_KEY,
                )
                received = []
                async for chunk in resp:
                    received.append(chunk)
                mock_post.assert_called_once()
                return received

        chunks_received = asyncio.run(_run())

    assert len(chunks_received) > 0
    content = "".join(c.choices[0].delta.content for c in chunks_received if c.choices[0].delta.content)
    assert "language" in content.lower()

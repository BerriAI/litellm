"""
Tests for Tensormesh chat transformation and provider routing.
"""

import json
import os
import sys
from typing import Any
from unittest.mock import patch

from .helpers import (
    _FakeOpenAIRawResponse,
    _make_fake_openai_chat_client,
    _make_fake_openai_streaming_chat_client,
    _openai_stream_chunk,
)

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)


def test_tensormesh_provider_configured(monkeypatch):
    from litellm.llms.tensormesh.chat.transformation import TensormeshChatConfig
    from litellm.llms.tensormesh.common_utils import TENSORMESH_API_BASE

    monkeypatch.delenv("TENSORMESH_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("TENSORMESH_SERVERLESS_BASE_URL", raising=False)

    config = TensormeshChatConfig()
    supported_params = config.get_supported_openai_params(
        model="MiniMaxAI/MiniMax-M2.7"
    )

    assert TENSORMESH_API_BASE == "https://serverless.tensormesh.ai/v1"
    assert config.custom_llm_provider == "tensormesh"
    assert config.get_api_base() == "https://serverless.tensormesh.ai/v1"
    assert config.get_api_key() is None
    assert "tools" in supported_params
    assert "tool_choice" in supported_params
    assert "response_format" in supported_params


def test_tensormesh_provider_resolution_preserves_slash_model_ids(monkeypatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.delenv("TENSORMESH_INFERENCE_API_KEY", raising=False)
    monkeypatch.delenv("TENSORMESH_SERVERLESS_BASE_URL", raising=False)

    model, provider, api_key, api_base = get_llm_provider(
        model="tensormesh/Qwen/Qwen3-Coder-30B-A3B-Instruct",
    )

    assert model == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert provider == "tensormesh"
    assert api_key is None
    assert api_base == "https://serverless.tensormesh.ai/v1"


def test_tensormesh_provider_resolution_uses_api_base_env(monkeypatch):
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    monkeypatch.setenv(
        "TENSORMESH_SERVERLESS_BASE_URL",
        "https://staging.serverless.tensormesh.ai/v1",
    )

    model, provider, _, api_base = get_llm_provider(
        model="tensormesh/openai/gpt-oss-120b",
    )

    assert model == "openai/gpt-oss-120b"
    assert provider == "tensormesh"
    assert api_base == "https://staging.serverless.tensormesh.ai/v1"


def test_tensormesh_provider_config_manager_returns_dedicated_config():
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_chat_config(
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        provider=LlmProviders.TENSORMESH,
    )

    assert config is not None
    assert config.custom_llm_provider == "tensormesh"
    supported_params = config.get_supported_openai_params(
        model="MiniMaxAI/MiniMax-M2.7"
    )
    assert "tools" in supported_params
    assert "tool_choice" in supported_params
    assert "response_format" in supported_params


def test_tensormesh_is_registered_as_openai_compatible_provider():
    import litellm

    assert "tensormesh" in litellm.openai_compatible_providers
    assert "tensormesh" in litellm.openai_text_completion_compatible_providers
    assert "tensormesh" in litellm.LITELLM_CHAT_PROVIDERS
    assert "https://serverless.tensormesh.ai/v1" in litellm.openai_compatible_endpoints


def test_tensormesh_public_endpoint_support():
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../../..")
    )
    support_files = (
        os.path.join(repo_root, "provider_endpoints_support.json"),
        os.path.join(repo_root, "litellm/provider_endpoints_support_backup.json"),
    )
    expected_endpoints = {
        "chat_completions": True,
        "messages": True,
        "responses": True,
        "embeddings": False,
        "image_generations": False,
        "audio_transcriptions": False,
        "audio_speech": False,
        "moderations": False,
        "batches": False,
        "rerank": False,
        "a2a": False,
        "interactions": False,
        "text_completion": True,
    }

    for support_file in support_files:
        with open(support_file, encoding="utf-8") as f:
            endpoints = json.load(f)["providers"]["tensormesh"]["endpoints"]

        assert endpoints == expected_endpoints


def test_tensormesh_chat_completion_builds_sdk_compatible_request(monkeypatch):
    import litellm

    monkeypatch.delenv("TENSORMESH_SERVERLESS_BASE_URL", raising=False)
    captured: dict[str, Any] = {}

    with patch(
        "litellm.llms.openai.openai.OpenAI",
        _make_fake_openai_chat_client(captured, "chatcmpl-tensormesh-test"),
    ):
        response = litellm.completion(
            model="tensormesh/Qwen/Qwen3-Coder-30B-A3B-Instruct",
            messages=[{"role": "user", "content": "Say ok."}],
            api_key="tm-chat-test-key",
            max_completion_tokens=12,
        )

    assert response.choices[0].message.content == "ok"
    assert captured["api_key"] == "tm-chat-test-key"
    assert captured["base_url"] == "https://serverless.tensormesh.ai/v1"
    assert captured["body"]["model"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert captured["body"]["messages"] == [{"role": "user", "content": "Say ok."}]
    assert captured["body"]["max_tokens"] == 12
    assert "max_completion_tokens" not in captured["body"]


def test_tensormesh_chat_completion_allows_tools_and_response_format(monkeypatch):
    import litellm

    monkeypatch.delenv("TENSORMESH_SERVERLESS_BASE_URL", raising=False)
    captured: dict[str, Any] = {}
    weather_tool = {
        "type": "function",
        "function": {
            "name": "weather",
            "description": "Return weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
            },
            "strict": True,
        },
    }

    with patch(
        "litellm.llms.openai.openai.OpenAI",
        _make_fake_openai_chat_client(captured, "chatcmpl-tensormesh-tools-test"),
    ):
        litellm.completion(
            model="tensormesh/MiniMaxAI/MiniMax-M2.7",
            messages=[{"role": "user", "content": "Use the weather tool."}],
            api_key="tm-chat-tools-test-key",
            tools=[weather_tool],
            tool_choice="required",
            response_format=response_format,
            max_tokens=64,
        )

    assert captured["body"]["tools"] == [weather_tool]
    assert captured["body"]["tool_choice"] == "required"
    assert captured["body"]["response_format"] == response_format


def test_tensormesh_streaming_preserves_reasoning_only_deltas(monkeypatch):
    import litellm

    monkeypatch.delenv("TENSORMESH_SERVERLESS_BASE_URL", raising=False)
    captured: dict[str, Any] = {}
    chunks = [
        _openai_stream_chunk({"role": "assistant"}),
        _openai_stream_chunk({"reasoning_content": "streamed reasoning text"}),
        _openai_stream_chunk({"reasoning": " alternate reasoning field"}),
        _openai_stream_chunk({}, finish_reason="stop"),
    ]

    with patch(
        "litellm.llms.openai.openai.OpenAI",
        _make_fake_openai_streaming_chat_client(captured, chunks),
    ):
        response = litellm.completion(
            model="tensormesh/MiniMaxAI/MiniMax-M2.7",
            messages=[{"role": "user", "content": "Stream a short response."}],
            api_key="tm-chat-stream-test-key",
            stream=True,
            max_tokens=64,
        )
        streamed_chunks = list(response)

    reasoning_parts: list[str] = []
    for chunk in streamed_chunks:
        if not chunk.choices:
            continue
        reasoning_content = getattr(chunk.choices[0].delta, "reasoning_content", None)
        if reasoning_content:
            reasoning_parts.append(reasoning_content)

    assert captured["body"]["stream"] is True
    assert "streamed reasoning text alternate reasoning field" in "".join(
        reasoning_parts
    )


def test_tensormesh_on_demand_chat_completion_uses_api_base_and_user_id_header():
    import litellm

    captured: dict[str, Any] = {}

    with patch(
        "litellm.llms.openai.openai.OpenAI",
        _make_fake_openai_chat_client(captured, "chatcmpl-tensormesh-on-demand-test"),
    ):
        response = litellm.completion(
            model="tensormesh/served-coding-model",
            messages=[{"role": "user", "content": "Say ok."}],
            api_key="tm-on-demand-test-key",
            api_base="https://external.example.tensormesh.ai/v1",
            extra_headers={"X-User-Id": "00000000-0000-0000-0000-000000000000"},
            max_tokens=12,
        )

    assert response.choices[0].message.content == "ok"
    assert captured["api_key"] == "tm-on-demand-test-key"
    assert captured["base_url"] == "https://external.example.tensormesh.ai/v1"
    assert captured["body"]["model"] == "served-coding-model"
    assert captured["body"]["extra_headers"] == {
        "X-User-Id": "00000000-0000-0000-0000-000000000000"
    }
    assert captured["body"]["max_tokens"] == 12


def test_tensormesh_on_demand_anthropic_messages_uses_api_base_and_user_id_header():
    import litellm

    captured: dict[str, Any] = {}

    with patch(
        "litellm.llms.openai.openai.OpenAI",
        _make_fake_openai_chat_client(
            captured, "chatcmpl-tensormesh-on-demand-messages-test"
        ),
    ):
        response = litellm.anthropic.messages.create(
            model="tensormesh/served-coding-model",
            messages=[{"role": "user", "content": "Say ok."}],
            api_key="tm-on-demand-messages-test-key",
            api_base="https://external.example.tensormesh.ai/v1",
            extra_headers={"X-User-Id": "00000000-0000-0000-0000-000000000000"},
            max_tokens=12,
            temperature=0,
        )

    assert response["type"] == "message"
    assert response["content"][0]["text"] == "ok"
    assert captured["api_key"] == "tm-on-demand-messages-test-key"
    assert captured["base_url"] == "https://external.example.tensormesh.ai/v1"
    assert captured["body"]["model"] == "served-coding-model"
    assert captured["body"]["extra_headers"] == {
        "X-User-Id": "00000000-0000-0000-0000-000000000000"
    }
    assert captured["body"]["max_tokens"] == 12


def test_tensormesh_anthropic_messages_uses_chat_completions_adapter(monkeypatch):
    import litellm

    monkeypatch.delenv("TENSORMESH_SERVERLESS_BASE_URL", raising=False)
    captured: dict[str, Any] = {}

    with patch(
        "litellm.llms.openai.openai.OpenAI",
        _make_fake_openai_chat_client(captured, "chatcmpl-tensormesh-messages-test"),
    ):
        response = litellm.anthropic.messages.create(
            model="tensormesh/Qwen/Qwen3-Coder-30B-A3B-Instruct",
            messages=[{"role": "user", "content": "Say ok."}],
            api_key="tm-messages-test-key",
            max_tokens=12,
            temperature=0,
        )

    assert response["type"] == "message"
    assert response["content"][0]["text"] == "ok"
    assert captured["api_key"] == "tm-messages-test-key"
    assert captured["base_url"] == "https://serverless.tensormesh.ai/v1"
    assert captured["body"]["model"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert captured["body"]["messages"] == [{"role": "user", "content": "Say ok."}]
    assert captured["body"]["max_tokens"] == 12


def test_tensormesh_text_completion_uses_v1_completions_base(monkeypatch):
    import litellm

    monkeypatch.delenv("TENSORMESH_SERVERLESS_BASE_URL", raising=False)
    captured: dict[str, Any] = {}

    class FakeCompletions:
        def __init__(self) -> None:
            self.with_raw_response = self

        def create(self, **data: Any) -> _FakeOpenAIRawResponse:
            captured["body"] = data
            return _FakeOpenAIRawResponse(
                {
                    "id": "cmpl-tensormesh-test",
                    "object": "text_completion",
                    "created": 1,
                    "model": data["model"],
                    "choices": [
                        {
                            "index": 0,
                            "text": "ok",
                            "logprobs": None,
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 1,
                        "completion_tokens": 1,
                        "total_tokens": 2,
                    },
                }
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured["api_key"] = kwargs["api_key"]
            captured["base_url"] = kwargs["base_url"]
            self.completions = FakeCompletions()

    with patch("litellm.llms.openai.completion.handler.OpenAI", FakeOpenAI):
        response = litellm.text_completion(
            model="tensormesh/openai/gpt-oss-120b",
            prompt="Reply with ok.",
            api_key="tm-text-test-key",
            max_tokens=7,
        )

    assert response.choices[0].text == "ok"
    assert captured["api_key"] == "tm-text-test-key"
    assert captured["base_url"] == "https://serverless.tensormesh.ai/v1"
    assert captured["body"]["model"] == "openai/gpt-oss-120b"
    assert captured["body"]["prompt"] == "Reply with ok."
    assert captured["body"]["max_tokens"] == 7

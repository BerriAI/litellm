import asyncio
import base64
import contextlib
import copy
import io
import json
import logging
import os
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx
from openai import APITimeoutError
from openai.types.chat.chat_completion import ChatCompletion

import litellm
from litellm import acompletion, completion
from litellm import acompletion_with_retries, aresponses_with_retries, completion_with_retries, responses_with_retries
from litellm import main as litellm_main
from litellm.constants import CONTROL_OPTIONS_KEY
from litellm.caching.base_cache import BaseCache
from litellm.caching.caching import Cache
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.custom_prompt_management import CustomPromptManagement
from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
from litellm.litellm_core_utils.get_litellm_params import stored_control_options
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging
from litellm.litellm_core_utils.prompt_templates.factory import anthropic_messages_pt
from litellm.llms.custom_httpx.http_handler import HTTPHandler
from litellm.types.litellm_params import ControlOptions
from litellm.types.llms.openai import AllMessageValues, HttpxBinaryResponseContent
from litellm.types.prompts.init_prompts import PromptSpec
from litellm.types.utils import Delta, ModelResponseStream, StandardCallbackDynamicParams, StreamingChoices, Usage


@pytest.fixture(autouse=True)
def clear_client_cache():
    """
    Clear the HTTP client cache before each test to ensure mocks are used.
    This prevents cached real clients from being reused across tests.
    """
    cache = getattr(litellm, "in_memory_llm_clients_cache", None)
    if cache is not None:
        cache.flush_cache()
    yield
    if cache is not None:
        cache.flush_cache()


@pytest.fixture(autouse=True)
def add_api_keys_to_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-1234567890")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-api03-1234567890")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "my-fake-aws-access-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "my-fake-aws-secret-access-key")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    # Keep these transformation tests on the simple access-key path. A leaked
    # session token or role/web-identity env var pushes Bedrock auth down a
    # different branch and fails before the mocked HTTP client is exercised.
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv("AWS_ROLE_ARN", raising=False)
    monkeypatch.delenv("AWS_WEB_IDENTITY_TOKEN_FILE", raising=False)


@pytest.fixture
def preserve_litellm_completion_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "set_verbose", litellm.set_verbose)
    monkeypatch.setattr(litellm, "custom_prompt_dict", litellm.custom_prompt_dict.copy())
    monkeypatch.setattr(litellm, "known_tokenizer_config", litellm.known_tokenizer_config.copy())


WHITE_PNG: Final = (Path(__file__).parents[1] / "white_100x100.png").read_bytes()


@pytest.fixture
def openai_api_response():
    mock_response_data = {
        "id": "chatcmpl-B0W3vmiM78Xkgx7kI7dr7PC949DMS",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": "",
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": None,
                },
            }
        ],
        "created": 1739462947,
        "model": "gpt-4o-mini-2024-07-18",
        "object": "chat.completion",
        "service_tier": "default",
        "system_fingerprint": "fp_bd83329f63",
        "usage": {
            "completion_tokens": 1,
            "prompt_tokens": 121,
            "total_tokens": 122,
            "completion_tokens_details": {
                "accepted_prediction_tokens": 0,
                "audio_tokens": 0,
                "reasoning_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
        },
    }

    return mock_response_data


def test_completion_missing_role(openai_api_response):
    from openai import OpenAI

    from litellm.types.utils import ModelResponse

    client = OpenAI(api_key="test_api_key")

    mock_raw_response = MagicMock()
    mock_raw_response.headers = {
        "x-request-id": "123",
        "openai-organization": "org-123",
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-remaining-requests": "99",
    }
    mock_raw_response.parse.return_value = ModelResponse(**openai_api_response)

    print(f"openai_api_response: {openai_api_response}")

    with patch.object(
        client.chat.completions.with_raw_response, "create", MagicMock(return_value=mock_raw_response)
    ) as mock_create:
        litellm.completion(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": "Hey"},
                {
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_m0vFJjQmTH1McvaHBPR2YFwY",
                            "function": {
                                "arguments": '{"input": "dksjsdkjdhskdjshdskhjkhlk"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 0,
                        },
                        {
                            "id": "call_Vw6RaqV2n5aaANXEdp5pYxo2",
                            "function": {
                                "arguments": '{"input": "jkljlkjlkjlkjlk"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 1,
                        },
                        {
                            "id": "call_hBIKwldUEGlNh6NlSXil62K4",
                            "function": {
                                "arguments": '{"input": "jkjlkjlkjlkj;lj"}',
                                "name": "tool_name",
                            },
                            "type": "function",
                            "index": 2,
                        },
                    ],
                },
            ],
            client=client,
        )

        mock_create.assert_called_once()


@pytest.mark.parametrize("model", ["gpt-4o-mini"])
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_url_with_format_param_openai(model, sync_mode):
    from openai import AsyncOpenAI, OpenAI

    from litellm import acompletion, completion

    if sync_mode:
        client = OpenAI()
    else:
        client = AsyncOpenAI()

    args = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png",
                            "format": "image/png",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
    }
    with patch.object(client.chat.completions.with_raw_response, "create") as mock_client:
        try:
            if sync_mode:
                response = completion(**args, client=client)
            else:
                response = await acompletion(**args, client=client)
            print(response)
        except Exception as e:
            print(e)

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        json_str = json.dumps(mock_client.call_args.kwargs)

        assert "format" not in json_str


@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-1.5-flash",
        "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "bedrock/invoke/anthropic.claude-haiku-4-5-20251001-v1:0",
        "anthropic/claude-3-5-sonnet",
    ],
)
@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_url_with_format_param(model, sync_mode, monkeypatch):
    from litellm import acompletion, completion
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

    if sync_mode:
        client = HTTPHandler()
    else:
        client = AsyncHTTPHandler()

    image_url: Final = (
        "https://awsmp-logos.s3.amazonaws.com/seller-xw5kijmvmzasy/c233c9ade2ccb5491072ae232c814942.png"
        f"?case={sync_mode}-{model}"
    )
    args = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url,
                            "format": "image/png",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
    }
    if model.startswith("gemini/"):
        args["api_key"] = "test-api-key"
    monkeypatch.setattr(litellm, "user_url_validation", False)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "module_level_aclient", AsyncHTTPHandler(transport=httpx.AsyncHTTPTransport()))
    with (
        respx.mock(assert_all_called=False) as image_host,
        patch.object(client, "post", new=MagicMock()) as mock_client,
    ):
        image_route = image_host.get(image_url).mock(
            return_value=httpx.Response(200, content=WHITE_PNG, headers={"content-type": "image/png"})
        )
        try:
            if sync_mode:
                response = completion(**args, client=client)
            else:
                response = await acompletion(**args, client=client)
            print(response)
        except Exception as e:
            pass

        mock_client.assert_called()

        print(mock_client.call_args.kwargs)

        if "data" in mock_client.call_args.kwargs:
            json_str = mock_client.call_args.kwargs["data"]
        else:
            json_str = json.dumps(mock_client.call_args.kwargs["json"])

        if isinstance(json_str, bytes):
            json_str = json_str.decode("utf-8")

        print(f"type of json_str: {type(json_str)}")

        if model.startswith("bedrock/invoke/"):
            assert "https://awsmp-logos.s3.amazonaws.com" not in json_str
            assert '"type":"base64"' in json_str or '"type": "base64"' in json_str
            assert '"data"' in json_str
        elif model.startswith("bedrock/"):
            assert "https://awsmp-logos.s3.amazonaws.com" not in json_str
            assert '"bytes"' in json_str or '"bytes":' in json_str
        elif model.startswith("anthropic/"):
            assert "https://awsmp-logos.s3.amazonaws.com" in json_str
            assert '"type":"url"' in json_str or '"type": "url"' in json_str
        else:
            assert "png" in json_str
            assert "jpeg" not in json_str

        fetches_image: Final = not model.startswith("anthropic/")
        assert image_route.called is fetches_image
        assert (base64.b64encode(WHITE_PNG).decode() in json_str) is fetches_image


def test_bedrock_latency_optimized_inference():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    with patch.object(client, "post") as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                performanceConfig={"latency": "optimized"},
                client=client,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert json_data["performanceConfig"]["latency"] == "optimized"


@pytest.mark.parametrize(
    ("custom_llm_provider", "model", "expected"),
    [
        ("anthropic", "claude-sonnet-5", True),
        ("bedrock", "us.anthropic.claude-sonnet-5-20260501-v1:0", True),
        ("bedrock", "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abc123", True),
        ("bedrock", "us.amazon.nova-2-lite-v1:0", False),
        ("vertex_ai", "claude-sonnet-5", True),
        ("vertex_ai", "gemini-3.8-flash", False),
        ("azure_ai", "claude-sonnet-4-6", True),
        ("azure_ai", "gpt-5.6", False),
        ("openai", "gpt-5.6", False),
        ("gemini", "gemini-3.8-flash", False),
    ],
)
def test_is_claude_tool_target(custom_llm_provider: str, model: str, expected: bool):
    assert litellm_main._is_claude_tool_target(custom_llm_provider=custom_llm_provider, model=model) is expected


@pytest.mark.parametrize("key", ["input_examples", "eager_input_streaming"])
def test_drop_anthropic_only_tool_keys_strips_tool_and_function_levels(key: str):
    tools = [
        {"type": "function", "name": "example_tool", key: True, "function": {"name": "example_tool", key: True}},
        "opaque_tool",
    ]

    cleaned = litellm_main._drop_anthropic_only_tool_keys(tools=tools)

    assert cleaned == [
        {"type": "function", "name": "example_tool", "function": {"name": "example_tool"}},
        "opaque_tool",
    ]
    assert tools[0][key] is True
    assert tools[0]["function"][key] is True


def test_completion_strips_eager_input_streaming_before_openai(respx_mock: respx.MockRouter, openai_api_response):
    api_base: Final = "http://localhost:12346/v1"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/chat/completions.*").mock(
        return_value=httpx.Response(status_code=200, json=openai_api_response)
    )

    litellm.completion(
        model="openai/gpt-5.6",
        messages=[{"role": "user", "content": "Write the file"}],
        tools=[
            {
                "type": "function",
                "function": {"name": "write_file", "parameters": {"type": "object", "properties": {}}},
                "eager_input_streaming": True,
            }
        ],
        api_base=api_base,
        api_key="fake_openai_api_key",
    )

    assert mock_route.called
    sent_tool: Final = json.loads(respx_mock.calls[0].request.content)["tools"][0]
    assert "eager_input_streaming" not in sent_tool
    assert sent_tool["function"]["name"] == "write_file"


def test_embedding_keeps_an_internal_prefixed_kwarg_out_of_the_provider_request(respx_mock: respx.MockRouter) -> None:
    api_base: Final = "http://localhost:12346/v1"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/embeddings.*").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
        )
    )

    litellm.embedding(
        model="openai/text-embedding-3-small",
        input="hi",
        api_base=api_base,
        api_key="fake_openai_api_key",
        _litellm_undeclared_sentinel="internal",
    )

    assert mock_route.called
    sent: Final = json.loads(respx_mock.calls[0].request.content)
    assert "_litellm_undeclared_sentinel" not in sent, sent
    assert sent["model"] == "text-embedding-3-small"


def test_custom_provider_with_extra_headers():

    with patch.object(litellm.llms.custom_httpx.http_handler.HTTPHandler, "post") as mock_post:
        response = litellm.completion(
            model="custom/custom",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            headers={"X-Custom-Header": "custom-value"},
            api_base="https://example.com/api/v1",
        )

        mock_post.assert_called_once()
        assert mock_post.call_args[1]["headers"]["X-Custom-Header"] == "custom-value"


def test_custom_provider_with_extra_body():

    with patch.object(litellm.llms.custom_httpx.http_handler.HTTPHandler, "post") as mock_post:
        response = litellm.completion(
            model="custom/custom",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            extra_body={
                "X-Custom-BodyValue": "custom-value",
                "X-Custom-BodyValue2": "custom-value2",
            },
            api_base="https://example.com/api/v1",
        )
        mock_post.assert_called_once()

        assert mock_post.call_args[1]["json"]["X-Custom-BodyValue"] == "custom-value"
        assert mock_post.call_args[1]["json"] == {
            "model": "custom",
            "params": {
                "prompt": ["Hello, how are you?"],
                "max_tokens": None,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            },
            "X-Custom-BodyValue": "custom-value",
            "X-Custom-BodyValue2": "custom-value2",
        }

    # test that extra_body is not passed if not provided
    with patch.object(litellm.llms.custom_httpx.http_handler.HTTPHandler, "post") as mock_post:
        response = litellm.completion(
            model="custom/custom",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            api_base="https://example.com/api/v1",
        )
        mock_post.assert_called_once()
        assert mock_post.call_args[1]["json"] == {
            "model": "custom",
            "params": {
                "prompt": ["Hello, how are you?"],
                "max_tokens": None,
                "temperature": None,
                "top_p": None,
                "top_k": None,
            },
        }


@pytest.fixture(autouse=True)
def set_openrouter_api_key():
    original_api_key = os.environ.get("OPENROUTER_API_KEY")
    os.environ["OPENROUTER_API_KEY"] = "fake-key-for-testing"
    yield
    if original_api_key is not None:
        os.environ["OPENROUTER_API_KEY"] = original_api_key
    else:
        del os.environ["OPENROUTER_API_KEY"]


@pytest.mark.asyncio
async def test_extra_body_with_fallback(respx_mock: respx.MockRouter, set_openrouter_api_key, monkeypatch):
    """
    test regression for https://github.com/BerriAI/litellm/issues/8425.

    This was perhaps a wider issue with the acompletion function not passing kwargs such as extra_body correctly when fallbacks are specified.
    """

    # Save original state to restore after test
    original_disable_aiohttp = litellm.disable_aiohttp_transport

    try:
        # since this uses respx, we need to set use_aiohttp_transport to False
        # Set both the global variable and environment variable to ensure it takes effect
        litellm.disable_aiohttp_transport = True
        monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
        # Flush cache to ensure no stale aiohttp clients are used
        litellm.in_memory_llm_clients_cache.flush_cache()

        # Set up test parameters
        model = "openrouter/deepseek/deepseek-chat"
        messages = [{"role": "user", "content": "Hello, world!"}]
        extra_body = {
            "provider": {
                "order": ["DeepSeek"],
                "allow_fallbacks": False,
                "require_parameters": True,
            }
        }
        fallbacks = [{"model": "openrouter/google/gemini-flash-1.5-8b"}]

        # Set up mock to respond to any POST request to the OpenRouter endpoint
        # This ensures it works for both primary and fallback models
        mock_route = respx_mock.post("https://openrouter.ai/api/v1/chat/completions")
        mock_route.return_value = httpx.Response(
            200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from mocked response!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
        )

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            extra_body=extra_body,
            fallbacks=fallbacks,
            api_key="fake-openrouter-api-key",
        )

        # Verify the response
        assert response is not None
        assert len(respx_mock.calls) > 0, "Mock was not called - check if aiohttp transport is properly disabled"

        # Get the request from the mock
        request: httpx.Request = respx_mock.calls[0].request
        request_body = request.read()
        request_body = json.loads(request_body)

        # Verify basic parameters
        assert request_body["model"] == "deepseek/deepseek-chat"
        assert request_body["messages"] == messages

        # Verify the extra_body parameters remain under the provider key
        assert request_body["provider"]["order"] == ["DeepSeek"]
        assert request_body["provider"]["allow_fallbacks"] is False
        assert request_body["provider"]["require_parameters"] is True
    finally:
        # Restore original state to prevent test pollution
        litellm.disable_aiohttp_transport = original_disable_aiohttp
        litellm.in_memory_llm_clients_cache.flush_cache()


@pytest.mark.parametrize("env_base", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_openai_env_base(respx_mock: respx.MockRouter, env_base, openai_api_response, monkeypatch):
    "This tests OpenAI env variables are honored, including legacy OPENAI_API_BASE"
    # Ensure aiohttp transport is disabled to use httpx which respx can mock
    litellm.disable_aiohttp_transport = True

    expected_base_url = "http://localhost:12345/v1"

    # Assign the environment variable based on env_base, and use a fake API key.
    monkeypatch.setenv(env_base, expected_base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_api_key")

    model = "gpt-4o"
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    # Configure respx mock to intercept the request
    mock_route = respx_mock.post(url__regex=r"http://localhost:12345/v1/chat/completions.*").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello from mocked response!",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
        )
    )

    try:
        response = await litellm.acompletion(model=model, messages=messages)

        # verify we had a response
        assert response.choices[0].message.content == "Hello from mocked response!"

        # Verify the mock was called
        assert mock_route.called, "Mock route was not called - request may have bypassed respx"
    finally:
        # Clean up to avoid affecting other tests
        litellm.disable_aiohttp_transport = False


def build_database_url(username, password, host, dbname):
    username_enc = urllib.parse.quote_plus(username)
    password_enc = urllib.parse.quote_plus(password)
    dbname_enc = urllib.parse.quote_plus(dbname)
    return f"postgresql://{username_enc}:{password_enc}@{host}/{dbname_enc}"


def test_build_database_url():
    url = build_database_url("user@name", "p@ss:word", "localhost", "db/name")
    assert url == "postgresql://user%40name:p%40ss%3Aword@localhost/db%2Fname"


def test_bedrock_llama():
    litellm.turn_on_debug()
    from litellm.types.utils import CallTypes
    from litellm.utils import return_raw_request

    model = "bedrock/invoke/us.meta.llama4-scout-17b-instruct-v1:0"

    request = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": model,
            "messages": [
                {"role": "user", "content": "hi"},
            ],
        },
    )
    print(request)

    assert (
        request["raw_request_body"]["prompt"]
        == "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nhi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def _mocked_openai_chat_response(model: str) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1677652288,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello from mocked response!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21,
            },
        },
    )


def test_return_raw_request_does_not_call_provider(respx_mock: respx.MockRouter):
    """Regression for #33952: return_raw_request must transform without contacting the provider.

    Previously return_raw_request invoked the real endpoint with a fake key and relied on the
    provider rejecting it, which sent an unintended inference request and (in the async proxy
    route) blocked the event loop on provider I/O.
    """
    from litellm.types.utils import CallTypes
    from litellm.utils import return_raw_request

    model = "gpt-4o"
    route = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_mocked_openai_chat_response(model)
    )

    request = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": model,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert route.call_count == 0
    assert request.get("error") is None
    assert request["raw_request_body"]["model"] == model
    assert request["raw_request_body"]["messages"] == [{"role": "user", "content": "hi"}]


def test_return_raw_request_ignores_turn_off_message_logging(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    from litellm.types.utils import CallTypes
    from litellm.utils import return_raw_request

    model: Final = "gpt-4o"
    messages: Final = [{"role": "user", "content": "PRIVATE-PHRASE"}]
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=_mocked_openai_chat_response(model)
    )
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)

    request: Final = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={"model": model, "messages": messages},
    )

    assert route.call_count == 0
    assert request.get("error") is None
    assert request["raw_request_body"]["messages"] == messages


def test_completion_forwards_verbosity_in_raw_request(respx_mock: respx.MockRouter):
    """Regression test: completion() must forward the verbosity param to the provider request body."""
    from litellm.types.utils import CallTypes
    from litellm.utils import return_raw_request

    model = "gpt-5.2"
    messages = [{"role": "user", "content": "hi"}]
    respx_mock.post("https://api.openai.com/v1/chat/completions").mock(return_value=_mocked_openai_chat_response(model))

    request = return_raw_request(
        endpoint=CallTypes.completion,
        kwargs={
            "model": model,
            "messages": messages,
            "verbosity": "high",
        },
    )

    assert request["raw_request_body"]["verbosity"] == "high"
    assert request["raw_request_body"]["model"] == model
    assert request["raw_request_body"]["messages"] == messages


@pytest.mark.asyncio
async def test_acompletion_forwards_verbosity_to_provider_request(respx_mock: respx.MockRouter, monkeypatch):
    """Regression test: acompletion() must forward the verbosity param to the provider request body."""
    original_disable_aiohttp = litellm.disable_aiohttp_transport
    try:
        litellm.disable_aiohttp_transport = True
        monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
        litellm.in_memory_llm_clients_cache.flush_cache()

        model = "gpt-5.2"
        messages = [{"role": "user", "content": "hi"}]
        mock_route = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
            return_value=_mocked_openai_chat_response(model)
        )

        response = await litellm.acompletion(
            model=model,
            messages=messages,
            verbosity="low",
            api_key="fake-openai-api-key",
        )

        assert response.choices[0].message.content == "Hello from mocked response!"
        assert mock_route.called
        request_body = json.loads(respx_mock.calls[0].request.read())
        assert request_body["verbosity"] == "low"
        assert request_body["model"] == model
        assert request_body["messages"] == messages
    finally:
        litellm.disable_aiohttp_transport = original_disable_aiohttp
        litellm.in_memory_llm_clients_cache.flush_cache()


def test_responses_api_bridge_check_strips_responses_prefix():
    """Test that responses_api_bridge_check strips 'responses/' prefix and sets mode."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 4096}

        model_info, model = responses_api_bridge_check(
            model="responses/gpt-4-responses",
            custom_llm_provider="openai",
        )

        assert model == "gpt-4-responses"
        assert model_info["mode"] == "responses"


def test_responses_api_bridge_check_gpt_5_4_pro():
    """Test that gpt-5.4-pro routes through responses API bridge, not chat completions.

    Regression test for https://github.com/BerriAI/litellm/issues/23014
    gpt-5.4-pro is a responses-only model and must not be sent to /v1/chat/completions.
    """
    from litellm.main import responses_api_bridge_check

    for model_name in ["gpt-5.4-pro", "gpt-5.4-pro-2026-03-05"]:
        model_info, model = responses_api_bridge_check(
            model=model_name,
            custom_llm_provider="openai",
        )
        assert model_info.get("mode") == "responses", (
            f"{model_name} should have mode='responses', got '{model_info.get('mode')}'"
        )


def test_responses_api_bridge_check_gpt_5_4_tools_plus_reasoning_routes_to_responses():
    """gpt-5.4 with both tools and reasoning_effort should route to Responses API."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort="xhigh",
        )

    assert model == "gpt-5.4"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_gpt_6_astra_tools_with_default_reasoning_routes_to_responses():
    from litellm.main import responses_api_bridge_check

    model_info, model = responses_api_bridge_check(
        model="gpt-6-astra",
        custom_llm_provider="openai",
        tools=[{"type": "function", "function": {"name": "get_capital"}}],
    )

    assert model == "gpt-6-astra"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_gpt_5_5_tools_plus_reasoning_routes_to_responses():
    """gpt-5.5+ with both tools and reasoning_effort should route to Responses API."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.5-pro",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort="xhigh",
        )

    assert model == "gpt-5.5-pro"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_azure_gpt_5_4_tools_plus_reasoning_routes_to_responses():
    """Azure gpt-5.4 with both tools and reasoning_effort should route to Responses API."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="azure",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort="high",
        )

    assert model == "gpt-5.4"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_azure_gpt_5_4_tools_with_default_reasoning_routes_to_responses():
    """
    Azure gpt-5.4 with tools and UNSET reasoning_effort must bridge: OpenAI enables
    reasoning by default for gpt-5.4+, and Chat Completions rejects function tools
    whenever reasoning is on.
    """
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="azure",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort=None,
        )

    assert model == "gpt-5.4"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_gpt_5_4_tools_with_default_reasoning_routes_to_responses():
    """
    gpt-5.4 with tools and UNSET reasoning_effort must bridge: OpenAI enables reasoning
    by default for gpt-5.4+, and Chat Completions rejects function tools whenever
    reasoning is on ("use /v1/responses or set reasoning_effort to 'none'").
    """
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort=None,
        )

    assert model == "gpt-5.4"
    assert model_info.get("mode") == "responses"


@pytest.mark.parametrize("region", ("us", "eu"))
@pytest.mark.parametrize(
    "model_name",
    (
        "codex-mini",
        "gpt-5-codex",
        "gpt-5-pro",
        "gpt-5.1-codex-max",
        "gpt-5.2-codex",
        "gpt-5.2-pro",
        "gpt-5.3-codex",
        "gpt-5.4-pro",
    ),
)
def test_responses_api_bridge_check_azure_regional_responses_only_models_route_to_responses(
    monkeypatch: pytest.MonkeyPatch, region: str, model_name: str
) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    model_info, model = litellm_main.responses_api_bridge_check(
        model=f"{region}/{model_name}",
        custom_llm_provider="azure",
        tools=[{"type": "function", "function": {"name": "get_capital"}}],
        reasoning_effort=None,
    )

    assert model == f"{region}/{model_name}"
    assert model_info.get("mode") == "responses"


@pytest.mark.parametrize(
    "model_name, expected_mode",
    [
        pytest.param("gpt-5.6-sol", "responses", id="above-boundary-bridges"),
        pytest.param("gpt-5.1", None, id="below-boundary-stays-chat"),
    ],
)
def test_responses_api_bridge_check_gpt_5_6_tools_with_default_reasoning_routes_to_responses(
    monkeypatch, model_name, expected_mode
):
    """
    gpt-5.6 must bridge on function tools alone. The bridge used to require an explicit
    reasoning_effort, so a gpt-5.6 call carrying tools and no effort was rejected with
    "Function tools with reasoning_effort are not supported for gpt-5.6-sol in
    /v1/chat/completions".

    Paired with a model below the gpt-5.4 boundary, which must still stay on chat. The
    gate parses the version and drops any suffix, so the family members bridge
    identically and only the boundary distinguishes behaviour.
    """
    import litellm
    from litellm.main import responses_api_bridge_check

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model=model_name,
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort=None,
        )

    assert model == model_name
    assert model_info.get("mode") == expected_mode


def test_responses_api_bridge_check_gpt_5_4_tools_with_reasoning_none_stays_chat():
    """
    Explicit reasoning_effort "none" is OpenAI's documented escape hatch that keeps
    function tools servable on Chat Completions; the bridge must not fire.
    """
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort="none",
        )

    assert model == "gpt-5.4"
    assert model_info.get("mode") != "responses"


def test_responses_api_bridge_check_reasoning_none_with_summary_still_routes_to_responses():
    """A reasoning summary is Responses-only regardless of effort value."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="openai",
            reasoning_effort="none",
            reasoning_summary="detailed",
        )

    assert model == "gpt-5.4"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_gpt_5_4_custom_tools_only_stays_chat():
    """
    Chat Completions serves custom (grammar) tools natively with reasoning on; only
    FUNCTION tools trigger the OpenAI rejection. Custom-only requests must stay on chat
    so responses keep the native custom tool_call shape instead of the bridge's
    function-shaped mapping.
    """
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "custom", "custom": {"name": "ApplyPatch", "description": "V4A patch"}}],
            reasoning_effort=None,
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") != "responses"


def test_responses_api_bridge_check_gpt_5_4_mixed_function_and_custom_tools_routes_to_responses():
    """One function tool in the mix is enough to make chat unservable with reasoning on."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[
                {"type": "custom", "custom": {"name": "ApplyPatch"}},
                {"type": "function", "function": {"name": "shell"}},
            ],
            reasoning_effort=None,
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_gpt_5_4_flat_function_tool_routes_to_responses():
    """Responses-style flat function tool defs still count as function tools."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
            reasoning_effort=None,
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") == "responses"


@pytest.mark.parametrize(
    "custom_llm_provider, model_name, api_base",
    [
        pytest.param("openai", "gpt-5.6", None, id="openai"),
        pytest.param("azure_ai", "gpt-6-astra", "https://myproject.services.ai.azure.com", id="azure-ai-foundry"),
    ],
)
def test_responses_api_bridge_check_function_tool_without_body_stays_chat(
    monkeypatch, custom_llm_provider, model_name, api_base
):
    import litellm
    from litellm.main import responses_api_bridge_check

    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)

    model_info, model = responses_api_bridge_check(
        model=model_name,
        custom_llm_provider=custom_llm_provider,
        tools=[{"type": "function"}],
        reasoning_effort=None,
        api_base=api_base,
    )

    assert model == model_name
    assert model_info.get("mode") != "responses"


def test_responses_api_bridge_check_dict_effort_none_stays_chat():
    """The escape hatch must honor litellm's dict form: {"effort": "none"} means reasoning off."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort={"effort": "none"},
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") != "responses"


def test_responses_api_bridge_check_dict_effort_active_routes_to_responses():
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort={"effort": "low"},
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_dict_effort_none_with_summary_routes_to_responses():
    """A summary inside the dict form is Responses-only even when effort is none."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort={"effort": "none", "summary": "concise"},
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") == "responses"


@pytest.mark.parametrize("blank_api_base", [None, "", "   ", "\t"])
def test_responses_api_bridge_check_blank_api_base_is_default_openai(blank_api_base):
    """
    A blank api_base (None, empty, or whitespace) resolves to the default OpenAI
    endpoint downstream, which enforces the reasoning+tools constraint, so gpt-5.4+
    function-tool requests with unset reasoning_effort must still auto-bridge.
    """
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort=None,
            api_base=blank_api_base,
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_custom_api_base_with_unset_effort_stays_chat():
    """
    Chat-only OpenAI-compatible backends registered under the openai provider with a
    custom api_base and gpt-5.4+ model names serve tools-without-reasoning fine and
    have no /responses route; the unset-effort arm must not reroute them.
    """
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort=None,
            api_base="http://vllm.internal:8000/v1",
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") != "responses"


def test_responses_api_bridge_check_custom_api_base_via_global_with_unset_effort_stays_chat(monkeypatch):
    """
    A custom base set through the litellm.api_base global (not the call arg) is resolved the
    same way the chat handler resolves it, so the unset-effort arm must not reroute a chat-only
    backend to a /responses route it lacks. Regression guard: the gate previously inspected only
    the call-level api_base and bridged these requests.
    """
    import litellm
    from litellm.main import responses_api_bridge_check

    monkeypatch.setattr(litellm, "api_base", "http://vllm.internal:8000/v1")
    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort=None,
            api_base=None,
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") != "responses"


@pytest.mark.parametrize("env_var", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
def test_responses_api_bridge_check_custom_api_base_via_env_with_unset_effort_stays_chat(monkeypatch, env_var):
    """
    A custom base set via OPENAI_BASE_URL/OPENAI_API_BASE env is resolved identically to the chat
    handler, so the unset-effort arm leaves the request on chat instead of bridging it.
    """
    import litellm
    from litellm.main import responses_api_bridge_check

    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv(env_var, "http://vllm.internal:8000/v1")
    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort=None,
            api_base=None,
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") != "responses"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://southcentralus.privatelink.api.openai.com/v1",
        "https://privatelink.corp.api.openai.com/v1",
        "https://api.openai.com:443/v1",
        "https://api.openai.com/v1/",
        "HTTPS://API.OPENAI.COM/v1",
    ],
)
def test_responses_api_bridge_check_openai_backed_custom_api_base_with_unset_effort_routes_to_responses(api_base):
    """
    A custom api_base whose host is api.openai.com or a subdomain of it (a PrivateLink hostname, a
    port-qualified or trailing-slash default) still reaches the real OpenAI backend, which rejects
    function tools with reasoning on Chat Completions, so the unset-effort arm must bridge exactly as
    it does for the literal default URL. Regression guard for GH #39353.
    """
    from litellm.main import responses_api_bridge_check

    model_info, model = responses_api_bridge_check(
        model="gpt-5.6",
        custom_llm_provider="openai",
        tools=[{"type": "function", "function": {"name": "get_capital"}}],
        reasoning_effort=None,
        api_base=api_base,
    )

    assert model == "gpt-5.6"
    assert model_info.get("mode") == "responses"


@pytest.mark.parametrize(
    "api_base",
    [
        "https://api.openai.com.evil.example/v1",
        "https://notapi.openai.com/v1",
        "https://gateway.example/v1?upstream=api.openai.com",
        "https://openai.internal.example/api.openai.com/v1",
    ],
)
def test_responses_api_bridge_check_lookalike_custom_api_base_with_unset_effort_stays_chat(api_base):
    """Only the host decides: api.openai.com appearing elsewhere in the URL is still a foreign backend."""
    from litellm.main import responses_api_bridge_check

    model_info, model = responses_api_bridge_check(
        model="gpt-5.6",
        custom_llm_provider="openai",
        tools=[{"type": "function", "function": {"name": "get_capital"}}],
        reasoning_effort=None,
        api_base=api_base,
    )

    assert model == "gpt-5.6"
    assert model_info.get("mode") != "responses"


def test_responses_api_bridge_check_privatelink_api_base_via_env_with_unset_effort_routes_to_responses(monkeypatch):
    """A PrivateLink base set through OPENAI_BASE_URL resolves the way the chat handler's does and still bridges."""
    import litellm
    from litellm.main import responses_api_bridge_check

    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://southcentralus.privatelink.api.openai.com/v1")
    model_info, model = responses_api_bridge_check(
        model="gpt-5.6",
        custom_llm_provider="openai",
        tools=[{"type": "function", "function": {"name": "get_capital"}}],
        reasoning_effort=None,
        api_base=None,
    )

    assert model == "gpt-5.6"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_custom_api_base_with_explicit_effort_still_routes():
    """Explicit reasoning_effort keeps its pre-existing bridging behavior on any api_base."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.6",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort="high",
            api_base="http://vllm.internal:8000/v1",
        )

    assert model == "gpt-5.6"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_azure_with_api_base_and_unset_effort_routes():
    """Azure OpenAI always sets api_base and does enforce the constraint; keep bridging."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="azure",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort=None,
            api_base="https://myresource.openai.azure.com",
        )

    assert model == "gpt-5.4"
    assert model_info.get("mode") == "responses"


_FOUNDRY_API_BASE: Final = "https://myproject.services.ai.azure.com"
_FOUNDRY_FUNCTION_TOOL: Final = ({"type": "function", "function": {"name": "get_weather"}},)


@pytest.mark.parametrize(
    "model_name, api_base, reasoning_effort",
    [
        pytest.param("gpt-6-astra", _FOUNDRY_API_BASE, None, id="gpt-6-unset-effort"),
        pytest.param("gpt-6-astra", _FOUNDRY_API_BASE, "low", id="gpt-6-explicit-effort"),
        pytest.param("gpt-6-astra", "https://myresource.openai.azure.com", None, id="gpt-6-azure-openai-host"),
        pytest.param("gpt-5.6-sol", _FOUNDRY_API_BASE, "low", id="gpt-5.6-explicit-effort"),
        pytest.param("gpt-5.6-sol", _FOUNDRY_API_BASE, {"effort": "high"}, id="gpt-5.6-explicit-effort-dict"),
    ],
)
def test_responses_api_bridge_check_azure_ai_foundry_rejected_tools_route_to_responses(
    model_name, api_base, reasoning_effort
):
    from litellm.main import responses_api_bridge_check

    model_info, model = responses_api_bridge_check(
        model=model_name,
        custom_llm_provider="azure_ai",
        tools=_FOUNDRY_FUNCTION_TOOL,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
    )

    assert model == model_name
    assert model_info.get("mode") == "responses"


@pytest.mark.parametrize(
    "model_name, api_base, reasoning_effort",
    [
        pytest.param("gpt-6-astra", _FOUNDRY_API_BASE, "none", id="explicit-none-stays-chat"),
        pytest.param("gpt-5.6-sol", _FOUNDRY_API_BASE, None, id="gpt-5.6-unset-effort-stays-chat"),
        pytest.param("gpt-5.6-sol", _FOUNDRY_API_BASE, "none", id="gpt-5.6-explicit-none-stays-chat"),
        pytest.param("gpt-5.5", _FOUNDRY_API_BASE, "high", id="gpt-5.5-explicit-effort-stays-chat"),
        pytest.param("gpt-5.4-mini", _FOUNDRY_API_BASE, None, id="gpt-5.4-mini-unset-effort-stays-chat"),
        pytest.param("gpt-5.4-mini", _FOUNDRY_API_BASE, "low", id="gpt-5.4-mini-explicit-effort-stays-chat"),
        pytest.param("gpt-6-astra", "https://myproject.models.ai.azure.com", None, id="serverless-host-stays-chat"),
        pytest.param("Mistral-large-2411", _FOUNDRY_API_BASE, None, id="non-gpt-5-model-stays-chat"),
        pytest.param("claude-opus-4-1", _FOUNDRY_API_BASE, None, id="claude-on-foundry-stays-chat"),
    ],
)
def test_responses_api_bridge_check_azure_ai_without_foundry_responses_route_stays_chat(
    model_name, api_base, reasoning_effort
):
    from litellm.main import responses_api_bridge_check

    model_info, model = responses_api_bridge_check(
        model=model_name,
        custom_llm_provider="azure_ai",
        tools=_FOUNDRY_FUNCTION_TOOL,
        reasoning_effort=reasoning_effort,
        api_base=api_base,
    )

    assert model == model_name
    assert model_info.get("mode") != "responses"


def test_responses_api_bridge_check_older_gpt_5_tools_without_reasoning_stays_chat():
    """Pre-5.4 GPT-5 names keep the old boundary: tools alone never bridge."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.1",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort=None,
        )

    assert model == "gpt-5.1"
    assert model_info.get("mode") != "responses"


def test_responses_api_bridge_check_gpt_5_4_reasoning_summary_without_tools_routes_to_responses():
    """gpt-5.4+ with reasoning_effort + reasoningSummary but no tools should bridge (AI SDK)."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5.4",
            custom_llm_provider="openai",
            tools=None,
            reasoning_effort="medium",
            reasoning_summary="auto",
        )

    assert model == "gpt-5.4"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_gpt_5_reasoning_summary_routes_to_responses():
    """Bare ``gpt-5`` with reasoning_effort + reasoningSummary should bridge (not 5.4+)."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5",
            custom_llm_provider="openai",
            tools=None,
            reasoning_effort="medium",
            reasoning_summary="auto",
        )

    assert model == "gpt-5"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_gpt_5_tools_without_summary_stays_chat():
    """gpt-5 with tools + reasoning_effort but no summary should stay on chat."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 128000}
        model_info, model = responses_api_bridge_check(
            model="gpt-5",
            custom_llm_provider="openai",
            tools=[{"type": "function", "function": {"name": "get_capital"}}],
            reasoning_effort="medium",
            reasoning_summary=None,
        )

    assert model == "gpt-5"
    assert model_info.get("mode") != "responses"


@patch("litellm.completion_extras.responses_api_bridge.completion")
def test_gpt_5_4_responses_bridge_preserves_reasoning_summary_dict(
    mock_responses_completion,
):
    """When routed to Responses, preserve reasoning_effort summary dict."""
    mock_responses_completion.return_value = MagicMock()

    import litellm

    litellm.completion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_capital",
                    "description": "Get the capital of a country",
                    "parameters": {
                        "type": "object",
                        "properties": {"country": {"type": "string"}},
                    },
                },
            }
        ],
        reasoning_effort={"effort": "xhigh", "summary": "detailed"},
        api_key="fake-key",
    )

    assert mock_responses_completion.called is True
    optional_params = mock_responses_completion.call_args.kwargs["optional_params"]
    assert optional_params["reasoning_effort"] == {
        "effort": "xhigh",
        "summary": "detailed",
    }


@pytest.mark.parametrize("reasoning_effort", ["high", {"effort": "high"}])
def test_responses_bridge_preserves_reasoning_effort_with_drop_params(
    reasoning_effort,
    restore_model_registry,
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    response_body: Final = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1734366691,
        "status": "completed",
        "model": "test-responses-bridge",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Done.", "annotations": []}],
            }
        ],
        "parallel_tool_calls": True,
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "total_tokens": 2,
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "metadata": None,
        "temperature": None,
        "tool_choice": "auto",
        "tools": [],
        "top_p": None,
        "max_output_tokens": None,
        "previous_response_id": None,
        "reasoning": None,
        "truncation": None,
        "user": None,
    }
    response_route: Final = respx_mock.post("https://api.perplexity.ai/v1/responses").respond(json=response_body)
    model: Final = "perplexity/test-responses-bridge"
    litellm.register_model(
        {
            model: {
                "litellm_provider": "perplexity",
                "mode": "responses",
                "supports_reasoning": False,
                "input_cost_per_token": 0.0,
                "output_cost_per_token": 0.0,
            }
        },
        persist_across_reloads=False,
    )

    litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        reasoning_effort=reasoning_effort,
        drop_params=True,
        api_key="fake-key",
        api_base="https://api.perplexity.ai",
    )

    request_body: Final = json.loads(response_route.calls[0].request.content)
    assert request_body["reasoning"] == {"effort": "high"}


_FOUNDRY_RESPONSES_FUNCTION_CALL_BODY: Final = {
    "id": "resp_foundry",
    "object": "response",
    "created_at": 1789852145,
    "status": "completed",
    "model": "gpt-6-astra",
    "output": [
        {
            "id": "fc_1",
            "type": "function_call",
            "status": "completed",
            "arguments": '{"city":"Paris"}',
            "call_id": "call_1",
            "name": "get_weather",
        }
    ],
    "parallel_tool_calls": True,
    "usage": {
        "input_tokens": 53,
        "output_tokens": 18,
        "total_tokens": 71,
        "output_tokens_details": {"reasoning_tokens": 0},
    },
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "metadata": {},
    "temperature": 1.0,
    "tool_choice": "auto",
    "tools": [],
    "top_p": 1.0,
    "max_output_tokens": 200,
    "previous_response_id": None,
    "reasoning": {"effort": "medium", "summary": None},
    "truncation": "disabled",
    "user": None,
}


def test_completion_bridges_azure_ai_foundry_gpt_5_4_plus_function_tools_to_responses(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    responses_route: Final = respx_mock.post(f"{_FOUNDRY_API_BASE}/openai/v1/responses").respond(
        json=_FOUNDRY_RESPONSES_FUNCTION_CALL_BODY
    )

    response: Final = litellm.completion(
        model="azure_ai/gpt-6-astra",
        messages=[{"role": "user", "content": "What is the weather in Paris? Use the tool."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
                },
            }
        ],
        max_tokens=200,
        api_base=_FOUNDRY_API_BASE,
        api_key="fake-foundry-key",
    )

    assert [str(call.request.url) for call in respx_mock.calls] == [f"{_FOUNDRY_API_BASE}/openai/v1/responses"]
    request: Final = responses_route.calls[0].request
    request_body: Final = json.loads(request.content)
    assert request_body["tools"][0]["type"] == "function"
    assert request_body["tools"][0]["name"] == "get_weather"
    assert request.headers["api-key"] == "fake-foundry-key"
    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].function.name == "get_weather"


@pytest.mark.parametrize(
    "model, model_info, expected_model_param, expected_base_model_param",
    [
        ("gemini/gemini-3.1-pro", None, "gemini-3.1-pro", None),
        (
            "gemini/gemini-3.1-pro",
            {"base_model": "gemini-3.1-pro-preview"},
            "gemini-3.1-pro",
            "gemini-3.1-pro-preview",
        ),
    ],
)
def test_completion_optional_params_base_model(
    model: str,
    model_info: dict | None,
    expected_model_param: str,
    expected_base_model_param: str | None,
):
    """``model_info.base_model`` must reach ``get_optional_params`` as ``base_model``
    (an additive capability hint), without overwriting ``model`` with the label.

    Regression for #29618: overwriting ``model`` with a friendly ``base_model``
    label made Bedrock drop ``tools``/``tool_choice`` under ``drop_params``."""
    with patch("litellm.main.get_optional_params") as mock_get_optional_params:
        mock_get_optional_params.return_value = MagicMock()

        import litellm

        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "api_key": "fake-key",
            "mock_response": "Hey, how's it going?",
        }
        if model_info is not None:
            kwargs["model_info"] = model_info

        litellm.completion(**kwargs)

        assert mock_get_optional_params.called is True
        call_kwargs = mock_get_optional_params.call_args.kwargs
        assert call_kwargs["model"] == expected_model_param
        assert call_kwargs["base_model"] == expected_base_model_param


@patch("litellm.completion_extras.responses_api_bridge.completion")
def test_gpt_5_4_responses_bridge_merges_reasoning_summary_kwarg_without_tools(
    mock_responses_completion,
):
    """reasoningSummary without tools should route and merge into reasoning_effort dict."""
    mock_responses_completion.return_value = MagicMock()

    import litellm

    litellm.completion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "ok"}],
        reasoning_effort="medium",
        reasoningSummary="auto",
        api_key="fake-key",
    )

    assert mock_responses_completion.called is True
    optional_params = mock_responses_completion.call_args.kwargs["optional_params"]
    assert optional_params["reasoning_effort"] == {
        "effort": "medium",
        "summary": "auto",
    }
    assert "reasoningSummary" not in optional_params
    assert "reasoning_summary" not in optional_params


@patch("litellm.completion_extras.responses_api_bridge.completion")
def test_responses_bridge_preserves_reasoning_summary_without_effort(
    mock_responses_completion,
):
    """Reasoning summary should survive responses routing even without effort."""
    mock_responses_completion.return_value = MagicMock()

    import litellm

    with patch.object(litellm, "route_all_chat_openai_to_responses", True):
        litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ok"}],
            reasoningSummary="auto",
            api_key="fake-key",
        )

    assert mock_responses_completion.called is True
    optional_params = mock_responses_completion.call_args.kwargs["optional_params"]
    assert optional_params["reasoning_effort"] == {"summary": "auto"}
    assert "reasoningSummary" not in optional_params
    assert "reasoning_summary" not in optional_params


@patch("litellm.completion_extras.responses_api_bridge.completion")
def test_gpt_5_responses_bridge_tools_and_reasoning_summary(
    mock_responses_completion,
):
    """Bare gpt-5 with tools + reasoningSummary should bridge (OpenCode-style)."""
    mock_responses_completion.return_value = MagicMock()

    import litellm

    litellm.completion(
        model="gpt-5",
        messages=[{"role": "user", "content": "ok"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "apply_patch",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="auto",
        reasoning_effort="medium",
        reasoningSummary="auto",
        stream=True,
        api_key="fake-key",
    )

    assert mock_responses_completion.called is True
    optional_params = mock_responses_completion.call_args.kwargs["optional_params"]
    assert optional_params.get("reasoning_effort") == {
        "effort": "medium",
        "summary": "auto",
    }


def test_responses_api_bridge_check_handles_exception():
    """Test that responses_api_bridge_check handles exceptions and still processes responses/ models."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.side_effect = Exception("Model not found")

        model_info, model = responses_api_bridge_check(model="responses/custom-model", custom_llm_provider="custom")

        assert model == "custom-model"
        assert model_info["mode"] == "responses"


def test_responses_api_bridge_check_global_flag_routes_openai():
    """When route_all_chat_openai_to_responses is True, any OpenAI model routes to responses."""
    from litellm.main import responses_api_bridge_check

    with patch.object(litellm, "route_all_chat_openai_to_responses", True):
        model_info, model = responses_api_bridge_check(
            model="gpt-4o",
            custom_llm_provider="openai",
        )

    assert model == "gpt-4o"
    assert model_info.get("mode") == "responses"


def test_responses_api_bridge_check_global_flag_does_not_affect_azure():
    """route_all_chat_openai_to_responses should not affect Azure models."""
    from litellm.main import responses_api_bridge_check

    with patch.object(litellm, "route_all_chat_openai_to_responses", True):
        with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
            mock_get_model_info.return_value = {"max_tokens": 4096}
            model_info, model = responses_api_bridge_check(
                model="gpt-4o",
                custom_llm_provider="azure",
            )

    assert model_info.get("mode") != "responses"


def test_responses_api_bridge_check_global_flag_default_false():
    """By default, route_all_chat_openai_to_responses is False and doesn't affect routing."""
    from litellm.main import responses_api_bridge_check

    with patch.object(litellm, "route_all_chat_openai_to_responses", False):
        with patch("litellm.main.get_model_info_helper") as mock_get_model_info:
            mock_get_model_info.return_value = {"max_tokens": 4096}
            model_info, model = responses_api_bridge_check(
                model="gpt-4o",
                custom_llm_provider="openai",
            )

    assert model_info.get("mode") != "responses"


@pytest.mark.asyncio
async def test_async_mock_delay():
    """Use asyncio await for mock delay on acompletion"""
    import time

    from litellm import acompletion

    start_time = time.time()
    result = await acompletion(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        mock_delay=0.01,
        mock_response="Hello world",
    )
    end_time = time.time()
    delay = end_time - start_time
    assert delay >= 0.01


def test_stream_chunk_builder_keeps_tool_calls_carried_only_by_a_later_choice_of_a_multi_choice_chunk():
    from litellm import stream_chunk_builder
    from litellm.types.utils import (
        ChatCompletionDeltaToolCall,
        Delta,
        Function,
        ModelResponseStream,
        StreamingChoices,
    )

    def chunk(choices: list[StreamingChoices]) -> ModelResponseStream:
        return ModelResponseStream(
            id="chatcmpl-multi-choice",
            created=1751934860,
            model="gpt-4.1-mini",
            object="chat.completion.chunk",
            choices=choices,
        )

    chunks = [
        chunk(
            [
                StreamingChoices(index=0, delta=Delta(role="assistant", content="hello")),
                StreamingChoices(
                    index=1,
                    delta=Delta(
                        role="assistant",
                        tool_calls=[
                            ChatCompletionDeltaToolCall(
                                id="call_1",
                                index=0,
                                type="function",
                                function=Function(name="lookup_fruit", arguments='{"fruit":'),
                            )
                        ],
                    ),
                ),
            ]
        ),
        chunk(
            [
                StreamingChoices(index=0, delta=Delta(content=" world"), finish_reason="stop"),
                StreamingChoices(
                    index=1,
                    delta=Delta(
                        tool_calls=[ChatCompletionDeltaToolCall(index=0, function=Function(arguments='"kiwi"}'))]
                    ),
                    finish_reason="tool_calls",
                ),
            ]
        ),
    ]

    response = stream_chunk_builder(chunks=chunks)

    tool_calls = response.choices[0].message.tool_calls
    assert tool_calls is not None
    assert [(call.id, call.function.name, call.function.arguments) for call in tool_calls] == [
        ("call_1", "lookup_fruit", '{"fruit":"kiwi"}')
    ]


def test_parallel_function_call_stream_reassembles_each_arguments_payload():
    from litellm import stream_chunk_builder
    from litellm.types.utils import ChatCompletionDeltaToolCall, Delta, Function, ModelResponseStream, StreamingChoices

    def chunk(tool_calls: list[ChatCompletionDeltaToolCall], finish_reason: str | None = None) -> ModelResponseStream:
        return ModelResponseStream(
            id="chatcmpl-parallel-tools",
            created=1751934860,
            model="gpt-4.1-mini",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(index=0, delta=Delta(tool_calls=tool_calls), finish_reason=finish_reason)
            ],
        )

    chunks: Final = [
        chunk(
            [
                ChatCompletionDeltaToolCall(
                    id="call_weather",
                    index=0,
                    type="function",
                    function=Function(name="weather", arguments='{"city":"'),
                ),
                ChatCompletionDeltaToolCall(
                    id="call_time",
                    index=1,
                    type="function",
                    function=Function(name="time", arguments='{"zone":"'),
                ),
            ]
        ),
        chunk(
            [
                ChatCompletionDeltaToolCall(index=0, function=Function(arguments='Paris"}')),
                ChatCompletionDeltaToolCall(index=1, function=Function(arguments='UTC"}')),
            ],
            finish_reason="tool_calls",
        ),
    ]

    response: Final = stream_chunk_builder(chunks=chunks)
    tool_calls: Final = response.choices[0].message.tool_calls

    assert tool_calls is not None
    assert [(call.id, call.function.name, call.function.arguments) for call in tool_calls] == [
        ("call_weather", "weather", '{"city":"Paris"}'),
        ("call_time", "time", '{"zone":"UTC"}'),
    ]


def test_stream_chunk_builder_thinking_blocks():
    from litellm import stream_chunk_builder
    from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices

    chunks = [
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        reasoning_content="I need to summar",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "I need to summar",
                                "signature": None,
                            }
                        ],
                        provider_specific_fields={
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "I need to summar",
                                    "signature": None,
                                }
                            ]
                        },
                        content="",
                        role="assistant",
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        reasoning_content="ize the previous agent's thinking process into a",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "ize the previous agent's thinking process into a",
                                "signature": None,
                            }
                        ],
                        provider_specific_fields={
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "ize the previous agent's thinking process into a",
                                    "signature": None,
                                }
                            ]
                        },
                        content="",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        reasoning_content=" short description. Based on the input data provide",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": " short description. Based on the input data provide",
                                "signature": None,
                            }
                        ],
                        provider_specific_fields={
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": " short description. Based on the input data provide",
                                    "signature": None,
                                }
                            ]
                        },
                        content="",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        reasoning_content="d, it seems the agent was planning to refine their search",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "d, it seems the agent was planning to refine their search",
                                "signature": None,
                            }
                        ],
                        provider_specific_fields={
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "d, it seems the agent was planning to refine their search",
                                    "signature": None,
                                }
                            ]
                        },
                        content="",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        reasoning_content=" to focus more on technical aspects of home automation and home",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": " to focus more on technical aspects of home automation and home",
                                "signature": None,
                            }
                        ],
                        provider_specific_fields={
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": " to focus more on technical aspects of home automation and home",
                                    "signature": None,
                                }
                            ]
                        },
                        content="",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        reasoning_content=" energy system management.\n\nI'll create a brief",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": " energy system management.\n\nI'll create a brief",
                                "signature": None,
                            }
                        ],
                        provider_specific_fields={
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": " energy system management.\n\nI'll create a brief",
                                    "signature": None,
                                }
                            ]
                        },
                        content="",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        reasoning_content=" summary of what the agent was doing.",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": " summary of what the agent was doing.",
                                "signature": None,
                            }
                        ],
                        provider_specific_fields={
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": " summary of what the agent was doing.",
                                    "signature": None,
                                }
                            ]
                        },
                        content="",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=0,
                    delta=Delta(
                        reasoning_content="",
                        thinking_blocks=[
                            {
                                "type": "thinking",
                                "thinking": "",
                                "signature": "ErUBCkYIBRgCIkAKBSMkB2+MBF643wiWxlERsGXVdlhbPx9lnTIbygzjFIeZ5uhTV+HNWDon9vQV4hmXvAKwQfwS8vkNFB366l05Egzt2U18IpRrZRyQn1UaDDdYvKHYP8Ps1IbWjSIw8eSYOU9gtqNcwR6D0wY7iOPx2GliDEatLI5rSs96CByoTIoADL2M5bX8KP0jEpbHKh0ccYryigdH/3J8EiFt/BmGUceVASP5l9r22dFWiBgC",
                            }
                        ],
                        provider_specific_fields={
                            "thinking_blocks": [
                                {
                                    "type": "thinking",
                                    "thinking": "",
                                    "signature": "ErUBCkYIBRgCIkAKBSMkB2+MBF643wiWxlERsGXVdlhbPx9lnTIbygzjFIeZ5uhTV+HNWDon9vQV4hmXvAKwQfwS8vkNFB366l05Egzt2U18IpRrZRyQn1UaDDdYvKHYP8Ps1IbWjSIw8eSYOU9gtqNcwR6D0wY7iOPx2GliDEatLI5rSs96CByoTIoADL2M5bX8KP0jEpbHKh0ccYryigdH/3J8EiFt/BmGUceVASP5l9r22dFWiBgC",
                                }
                            ]
                        },
                        content="",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=1,
                    delta=Delta(
                        provider_specific_fields=None,
                        content='{"a',
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=1,
                    delta=Delta(
                        provider_specific_fields=None,
                        content='gent_doing"',
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=1,
                    delta=Delta(
                        provider_specific_fields=None,
                        content=': "Re',
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=1,
                    delta=Delta(
                        provider_specific_fields=None,
                        content="searching",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=1,
                    delta=Delta(
                        provider_specific_fields=None,
                        content=" technic",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=1,
                    delta=Delta(
                        provider_specific_fields=None,
                        content="al aspect",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=1,
                    delta=Delta(
                        provider_specific_fields=None,
                        content="s of home au",
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason=None,
                    index=1,
                    delta=Delta(
                        provider_specific_fields=None,
                        content='tomation"}',
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
            citations=None,
        ),
        ModelResponseStream(
            id="chatcmpl-e8febeb7-cf7d-4947-9417-59ae5e6989f9",
            created=1751934860,
            model="claude-3-7-sonnet-latest",
            object="chat.completion.chunk",
            system_fingerprint=None,
            choices=[
                StreamingChoices(
                    finish_reason="tool_calls",
                    index=0,
                    delta=Delta(
                        provider_specific_fields=None,
                        content=None,
                        role=None,
                        function_call=None,
                        tool_calls=None,
                        audio=None,
                    ),
                    logprobs=None,
                )
            ],
            provider_specific_fields=None,
        ),
    ]

    response = stream_chunk_builder(chunks=chunks)
    print(response)

    assert response is not None
    assert response.choices[0].message.content is not None
    assert response.choices[0].message.thinking_blocks is not None


from litellm.llms.openai.openai import OpenAIChatCompletion
import traceback

user_message = "Write a short poem about the sky"
messages = [{"content": user_message, "role": "user"}]


def throw_retryable_error(*_, **__):
    raise RuntimeError("BOOM")


@pytest.mark.asyncio
async def test_retrying() -> None:
    litellm.num_retries = 10
    with (
        patch.object(
            OpenAIChatCompletion,
            "make_openai_chat_completion_request",
            side_effect=throw_retryable_error,
        ) as mock_request,
        pytest.raises(litellm.InternalServerError, match="LiteLLM Retried: 10 times"),
    ):
        await litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
        )


def test_anthropic_disable_url_suffix_env_var(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://api.example.com")
    default_route: Final = respx_mock.post("https://api.example.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "test response"}],
                "model": "claude-3-sonnet",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )
    )
    default_response: Final = litellm.completion(
        model="anthropic/claude-3-sonnet",
        messages=[{"role": "user", "content": "test"}],
        api_key="test-key",
    )

    assert default_response.choices[0].message.content == "test response"
    assert default_route.called
    assert str(default_route.calls.last.request.url) == "https://api.example.com/v1/messages"

    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://api.example.com/custom/path")
    monkeypatch.setenv("LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX", "true")
    disabled_route: Final = respx_mock.post("https://api.example.com/custom/path").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "test response"}],
                "model": "claude-3-sonnet",
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )
    )
    disabled_response: Final = litellm.completion(
        model="anthropic/claude-3-sonnet",
        messages=[{"role": "user", "content": "test"}],
        api_key="test-key",
    )

    assert disabled_response.choices[0].message.content == "test response"
    assert disabled_route.called
    assert str(disabled_route.calls.last.request.url) == "https://api.example.com/custom/path"


def test_anthropic_text_disable_url_suffix_env_var(
    monkeypatch: pytest.MonkeyPatch, respx_mock: respx.MockRouter
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://api.example.com")
    default_route: Final = respx_mock.post("https://api.example.com/v1/complete").mock(
        return_value=httpx.Response(
            200,
            json={
                "completion": "test response",
                "stop_reason": "stop_sequence",
                "model": "claude-instant-1",
            },
        )
    )
    default_response: Final = litellm.text_completion(
        model="anthropic_text/claude-instant-1",
        prompt="test",
        api_key="test-key",
    )

    assert default_response.choices[0].text == "test response"
    assert default_route.called
    assert str(default_route.calls.last.request.url) == "https://api.example.com/v1/complete"

    monkeypatch.setenv("ANTHROPIC_API_BASE", "https://api.example.com/custom/complete")
    monkeypatch.setenv("LITELLM_ANTHROPIC_DISABLE_URL_SUFFIX", "true")
    disabled_route: Final = respx_mock.post("https://api.example.com/custom/complete").mock(
        return_value=httpx.Response(
            200,
            json={
                "completion": "test response",
                "stop_reason": "stop_sequence",
                "model": "claude-instant-1",
            },
        )
    )
    disabled_response: Final = litellm.text_completion(
        model="anthropic_text/claude-instant-1",
        prompt="test",
        api_key="test-key",
    )

    assert disabled_response.choices[0].text == "test response"
    assert disabled_route.called
    assert str(disabled_route.calls.last.request.url) == "https://api.example.com/custom/complete"


def test_image_edit_merges_headers_and_extra_headers():
    from litellm.images.main import base_llm_http_handler

    combined_headers = {
        "x-test-header-one": "value-1",
        "x-test-header-two": "value-2",
    }

    mock_image_edit_config = MagicMock()
    mock_image_edit_config.get_supported_openai_params.return_value = set()
    mock_image_edit_config.map_openai_params.side_effect = lambda **kwargs: dict(kwargs["image_edit_optional_params"])

    with (
        patch(
            "litellm.images.main.ProviderConfigManager.get_provider_image_edit_config",
            return_value=mock_image_edit_config,
        ) as mock_config,
        patch.object(
            base_llm_http_handler,
            "image_edit_handler",
            return_value="ok",
        ) as mock_handler,
    ):
        response = litellm.image_edit(
            image=MagicMock(name="image"),
            prompt="test",
            model="azure/gpt-image-1",
            headers={"x-test-header-one": "value-1"},
            extra_headers={
                "x-test-header-two": "value-2",
            },
        )

    assert response == "ok"
    mock_config.assert_called_once()

    handler_kwargs = mock_handler.call_args.kwargs
    assert handler_kwargs["extra_headers"] == combined_headers
    assert "extra_headers" not in handler_kwargs["image_edit_optional_request_params"]


@pytest.mark.parametrize("metadata_key", ("metadata", "litellm_metadata"))
@pytest.mark.parametrize("input_tokens", (51234, 0))
def test_mock_completion_usage_reports_admission_input_tokens(metadata_key: str, input_tokens: int):
    response = litellm.completion(
        model="anthropic/claude-sonnet-5",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="ok",
        api_key="mock",
        **{metadata_key: {"user_api_key_budget_reservation": {"reserved_cost": 1.0, "input_tokens": input_tokens}}},
    )

    assert response.usage.prompt_tokens == input_tokens
    assert response.usage.total_tokens == input_tokens + response.usage.completion_tokens


def test_mock_completion_usage_falls_back_to_default_without_admission_count():
    response = litellm.completion(
        model="anthropic/claude-sonnet-5",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="ok",
        api_key="mock",
        metadata={"user_api_key_budget_reservation": {"reserved_cost": 1.0}},
    )

    assert response.usage.prompt_tokens == litellm_main.DEFAULT_MOCK_RESPONSE_PROMPT_TOKEN_COUNT


_AZURE_AI_CUSTOM_PRICED_DEPLOYMENT: Final = {
    "model_name": "azure-ai-custom-priced",
    "litellm_params": {
        "model": "azure_ai/gpt-5.6",
        "api_key": "mock",
        "api_base": "https://example.services.ai.azure.com",
        "mock_response": "ok",
        "input_cost_per_token": 3e-6,
        "output_cost_per_token": 7e-6,
        "cache_read_input_token_cost": 1e-7,
        "cache_creation_input_token_cost": 5e-7,
    },
    "model_info": {"id": "azure-ai-custom-priced-deployment-id"},
}


def _expected_custom_price(response: litellm.ModelResponse) -> float:
    params: Final = _AZURE_AI_CUSTOM_PRICED_DEPLOYMENT["litellm_params"]
    return (
        response.usage.prompt_tokens * params["input_cost_per_token"]
        + response.usage.completion_tokens * params["output_cost_per_token"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", (False, True))
async def test_mock_completion_prices_azure_ai_router_deployment_with_custom_pricing(use_async: bool):
    router: Final = litellm.Router(model_list=[_AZURE_AI_CUSTOM_PRICED_DEPLOYMENT])
    messages: Final = [{"role": "user", "content": "hello"}]

    response: Final = (
        await router.acompletion(model="azure-ai-custom-priced", messages=messages)
        if use_async
        else router.completion(model="azure-ai-custom-priced", messages=messages)
    )

    assert response._hidden_params["response_cost"] == pytest.approx(_expected_custom_price(response))
    assert response._hidden_params["custom_llm_provider"] == "azure_ai"


@pytest.mark.parametrize(
    ("model", "expected_provider"),
    (("anthropic/claude-sonnet-5", "anthropic"), ("no-such-provider-model", None)),
)
def test_mock_completion_infers_provider_when_called_directly_without_one(model: str, expected_provider: str | None):
    response: Final = litellm.mock_completion(
        model=model,
        messages=[{"role": "user", "content": "hello"}],
        mock_response="ok",
    )

    assert response.choices[0].message.content == "ok"
    assert response._hidden_params.get("custom_llm_provider") == expected_provider


def test_mock_request():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litellm.mock_completion(model=model, messages=messages, stream=False)
        print(response)
        print(type(response))
    except Exception:
        traceback.print_exc()


def test_streaming_mock_request():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litellm.mock_completion(model=model, messages=messages, stream=True)
        complete_response = ""
        for chunk in response:
            complete_response += chunk["choices"][0]["delta"]["content"] or ""
        if complete_response == "":
            raise Exception("Empty response received")
    except Exception:
        traceback.print_exc()


@pytest.mark.asyncio()
async def test_async_mock_streaming_request():
    generator = await litellm.acompletion(
        messages=[{"role": "user", "content": "Why is LiteLLM amazing?"}],
        mock_response="LiteLLM is awesome",
        stream=True,
        model="gpt-3.5-turbo",
    )
    complete_response = ""
    async for chunk in generator:
        print(chunk)
        complete_response += chunk["choices"][0]["delta"]["content"] or ""

    assert (
        complete_response == "LiteLLM is awesome"
    ), f"Unexpected response got {complete_response}"


def test_mock_request_n_greater_than_1():
    try:
        model = "gpt-3.5-turbo"
        messages = [{"role": "user", "content": "Hey, I'm a mock request"}]
        response = litellm.mock_completion(model=model, messages=messages, n=5)
        print("response: ", response)

        assert len(response.choices) == 5
        for choice in response.choices:
            assert choice.message.content == "This is a mock request"

    except Exception:
        traceback.print_exc()


@pytest.mark.asyncio()
async def test_async_mock_streaming_request_n_greater_than_1():
    generator = await litellm.acompletion(
        messages=[{"role": "user", "content": "Why is LiteLLM amazing?"}],
        mock_response="LiteLLM is awesome",
        stream=True,
        model="gpt-3.5-turbo",
        n=5,
    )
    complete_response = ""
    async for chunk in generator:
        print(chunk)


_ADMISSION_INPUT_TOKENS: Final = 51234


def _admission_metadata(input_tokens: int) -> dict[str, object]:  # mutable-ok: logging writes into metadata
    return {"user_api_key_budget_reservation": {"reserved_cost": 1.0, "input_tokens": input_tokens}}


_ADMISSION_METADATA: Final = _admission_metadata(_ADMISSION_INPUT_TOKENS)
_MOCK_STREAM_MESSAGES: Final = [{"role": "user", "content": "hello " * 200}]
_STREAM_CHUNK_BUILDER_TOKEN_COUNTER: Final = "litellm.litellm_core_utils.streaming_chunk_builder_utils.token_counter"


def _prompt_token_counter_calls(token_counter: MagicMock) -> list[object]:
    return [call for call in token_counter.call_args_list if call.kwargs.get("messages") is not None]


def _client_usage_chunks(chunks: list[ModelResponseStream]) -> list[Usage]:
    return [chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None]


@pytest.mark.parametrize("n", (None, 2))
def test_mock_completion_stream_usage_reports_admission_input_tokens_without_tokenizer_fallback(n: int | None):
    with patch(_STREAM_CHUNK_BUILDER_TOKEN_COUNTER, wraps=litellm.token_counter) as token_counter:
        chunks: Final = list(
            litellm.completion(
                model="openai/gpt-5.4-mini",
                messages=_MOCK_STREAM_MESSAGES,
                mock_response="ok",
                api_key="mock",
                stream=True,
                n=n,
                stream_options={"include_usage": True},
                metadata=_ADMISSION_METADATA,
            )
        )

    usage_chunks: Final = _client_usage_chunks(chunks)
    assert len(usage_chunks) == 1
    assert usage_chunks[0].prompt_tokens == _ADMISSION_INPUT_TOKENS
    assert usage_chunks[0].completion_tokens == litellm_main.DEFAULT_MOCK_RESPONSE_COMPLETION_TOKEN_COUNT
    assert usage_chunks[0].total_tokens == _ADMISSION_INPUT_TOKENS + usage_chunks[0].completion_tokens
    assert _prompt_token_counter_calls(token_counter) == []
    assert all(chunk.choices for chunk in chunks[:-1])
    assert {chunk.id for chunk in chunks} == {chunks[0].id}


@pytest.mark.asyncio
@pytest.mark.parametrize("n", (None, 2))
async def test_mock_acompletion_stream_usage_reports_admission_input_tokens_without_tokenizer_fallback(
    n: int | None,
):
    with patch(_STREAM_CHUNK_BUILDER_TOKEN_COUNTER, wraps=litellm.token_counter) as token_counter:
        response: Final = await litellm.acompletion(
            model="openai/gpt-5.4-mini",
            messages=_MOCK_STREAM_MESSAGES,
            mock_response="ok",
            api_key="mock",
            stream=True,
            n=n,
            stream_options={"include_usage": True},
            litellm_metadata=_ADMISSION_METADATA,
        )
        chunks: Final = [chunk async for chunk in response]

    usage_chunks: Final = _client_usage_chunks(chunks)
    assert len(usage_chunks) == 1
    assert usage_chunks[0].prompt_tokens == _ADMISSION_INPUT_TOKENS
    assert usage_chunks[0].total_tokens == _ADMISSION_INPUT_TOKENS + usage_chunks[0].completion_tokens
    assert _prompt_token_counter_calls(token_counter) == []
    assert all(chunk.choices for chunk in chunks[:-1])
    assert {chunk.id for chunk in chunks} == {chunks[0].id}


def test_mock_completion_stream_without_include_usage_hides_usage_chunk_but_logs_admission_count():
    with patch(_STREAM_CHUNK_BUILDER_TOKEN_COUNTER, wraps=litellm.token_counter) as token_counter:
        chunks: Final = list(
            litellm.completion(
                model="openai/gpt-5.4-mini",
                messages=_MOCK_STREAM_MESSAGES,
                mock_response="ok",
                api_key="mock",
                stream=True,
                metadata=_ADMISSION_METADATA,
            )
        )

    assert _client_usage_chunks(chunks) == []
    assert all(len(chunk.choices) == 1 for chunk in chunks)
    assert chunks[-1]._hidden_params["usage"].prompt_tokens == _ADMISSION_INPUT_TOKENS
    assert _prompt_token_counter_calls(token_counter) == []


def test_mock_completion_stream_with_empty_stream_options_completes_and_logs_admission_count():
    with patch(_STREAM_CHUNK_BUILDER_TOKEN_COUNTER, wraps=litellm.token_counter) as token_counter:
        chunks: Final = list(
            litellm.completion(
                model="openai/gpt-5.4-mini",
                messages=_MOCK_STREAM_MESSAGES,
                mock_response="ok",
                api_key="mock",
                stream=True,
                stream_options={},
                metadata=_ADMISSION_METADATA,
            )
        )

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "ok"
    assert _client_usage_chunks(chunks) == []
    assert _prompt_token_counter_calls(token_counter) == []


@pytest.mark.asyncio
async def test_mock_acompletion_stream_with_empty_stream_options_completes_and_logs_admission_count():
    with patch(_STREAM_CHUNK_BUILDER_TOKEN_COUNTER, wraps=litellm.token_counter) as token_counter:
        response: Final = await litellm.acompletion(
            model="openai/gpt-5.4-mini",
            messages=_MOCK_STREAM_MESSAGES,
            mock_response="ok",
            api_key="mock",
            stream=True,
            stream_options={},
            litellm_metadata=_ADMISSION_METADATA,
        )
        chunks: Final = [chunk async for chunk in response]

    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "ok"
    assert _client_usage_chunks(chunks) == []
    assert _prompt_token_counter_calls(token_counter) == []


def test_mock_completion_stream_without_admission_count_falls_back_to_tokenizer():
    expected_prompt_tokens: Final = litellm.token_counter(model="openai/gpt-5.4-mini", messages=_MOCK_STREAM_MESSAGES)
    with patch(_STREAM_CHUNK_BUILDER_TOKEN_COUNTER, wraps=litellm.token_counter) as token_counter:
        chunks: Final = list(
            litellm.completion(
                model="openai/gpt-5.4-mini",
                messages=_MOCK_STREAM_MESSAGES,
                mock_response="ok",
                api_key="mock",
                stream=True,
                stream_options={"include_usage": True},
                metadata={"user_api_key_budget_reservation": {"reserved_cost": 1.0}},
            )
        )

    usage_chunks: Final = _client_usage_chunks(chunks)
    assert len(usage_chunks) == 1
    assert usage_chunks[0].prompt_tokens == expected_prompt_tokens
    assert usage_chunks[0].total_tokens == expected_prompt_tokens + usage_chunks[0].completion_tokens
    assert len(_prompt_token_counter_calls(token_counter)) >= 1


@pytest.mark.asyncio
async def test_mock_acompletion_stream_without_admission_count_falls_back_to_tokenizer():
    expected_prompt_tokens: Final = litellm.token_counter(model="openai/gpt-5.4-mini", messages=_MOCK_STREAM_MESSAGES)
    with patch(_STREAM_CHUNK_BUILDER_TOKEN_COUNTER, wraps=litellm.token_counter) as token_counter:
        response: Final = await litellm.acompletion(
            model="openai/gpt-5.4-mini",
            messages=_MOCK_STREAM_MESSAGES,
            mock_response="ok",
            api_key="mock",
            stream=True,
            stream_options={"include_usage": True},
        )
        chunks: Final = [chunk async for chunk in response]

    usage_chunks: Final = _client_usage_chunks(chunks)
    assert len(usage_chunks) == 1
    assert usage_chunks[0].prompt_tokens == expected_prompt_tokens
    assert len(_prompt_token_counter_calls(token_counter)) >= 1


def _usage_triple(usage: Usage) -> tuple[int, int, int]:
    return (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens)


@pytest.mark.parametrize("input_tokens", (_ADMISSION_INPUT_TOKENS, 0))
def test_mock_completion_stream_and_non_stream_report_the_same_admission_usage(input_tokens: int):
    metadata: Final = _admission_metadata(input_tokens)
    non_stream: Final = litellm.completion(
        model="openai/gpt-5.4-mini",
        messages=_MOCK_STREAM_MESSAGES,
        mock_response="ok",
        api_key="mock",
        metadata=metadata,
    )
    with patch(_STREAM_CHUNK_BUILDER_TOKEN_COUNTER, wraps=litellm.token_counter) as token_counter:
        chunks: Final = list(
            litellm.completion(
                model="openai/gpt-5.4-mini",
                messages=_MOCK_STREAM_MESSAGES,
                mock_response="ok",
                api_key="mock",
                stream=True,
                stream_options={"include_usage": True},
                metadata=metadata,
            )
        )

    assert _usage_triple(non_stream.usage) == _usage_triple(_client_usage_chunks(chunks)[0])
    assert non_stream.usage.prompt_tokens == input_tokens
    assert _prompt_token_counter_calls(token_counter) == []


@pytest.mark.asyncio
async def test_mock_acompletion_stream_reports_zero_admission_input_tokens_without_tokenizer_fallback():
    with patch(_STREAM_CHUNK_BUILDER_TOKEN_COUNTER, wraps=litellm.token_counter) as token_counter:
        response: Final = await litellm.acompletion(
            model="openai/gpt-5.4-mini",
            messages=[{"role": "user", "content": ""}],
            mock_response="ok",
            api_key="mock",
            stream=True,
            stream_options={"include_usage": True},
            litellm_metadata=_admission_metadata(0),
        )
        chunks: Final = [chunk async for chunk in response]

    usage_chunks: Final = _client_usage_chunks(chunks)
    assert len(usage_chunks) == 1
    assert _usage_triple(usage_chunks[0]) == (0, usage_chunks[0].completion_tokens, usage_chunks[0].completion_tokens)
    assert _prompt_token_counter_calls(token_counter) == []


def test_mock_text_completion_stream_and_non_stream_report_the_same_zero_admission_usage():
    metadata: Final = _admission_metadata(0)
    non_stream: Final = litellm.text_completion(
        model="openai/gpt-5.4-mini", prompt="", mock_response="ok", api_key="mock", metadata=metadata
    )
    chunks: Final = list(
        litellm.text_completion(
            model="openai/gpt-5.4-mini",
            prompt="",
            mock_response="ok",
            api_key="mock",
            stream=True,
            stream_options={"include_usage": True},
            metadata=metadata,
        )
    )

    stream_usages: Final = tuple(chunk.usage for chunk in chunks if getattr(chunk, "usage", None) is not None)
    assert len(stream_usages) == 1
    assert _usage_triple(non_stream.usage) == _usage_triple(stream_usages[0])
    assert non_stream.usage.prompt_tokens == 0


def test_mock_completion_stream_with_model_response():
    """Test that mock_completion correctly handles stream=True with a ModelResponse as mock_response."""
    from litellm import completion
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    # Create a ModelResponse object
    mock_model_response = ModelResponse(
        id="chatcmpl-test-123",
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="This is a test response",
                    role="assistant",
                ),
            )
        ],
        usage=Usage(
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
        ),
    )

    # Call completion with stream=True and mock_response as ModelResponse
    response = completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        mock_response=mock_model_response,
    )

    # Verify that the response is a stream
    assert response is not None

    # Collect all chunks from the stream
    chunks = []
    for chunk in response:
        chunks.append(chunk)
        print(f"Chunk: {chunk}")

    # Verify we got chunks
    assert len(chunks) > 0

    # Verify the content is streamed correctly
    accumulated_content = ""
    for chunk in chunks:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content

    assert "This is a test response" in accumulated_content or len(chunks) > 0


@pytest.mark.asyncio
async def test_async_mock_completion_stream_with_model_response():
    """Test that async mock_completion correctly handles stream=True with a ModelResponse as mock_response."""
    from litellm import acompletion
    from litellm.types.utils import Choices, Message, ModelResponse, Usage

    # Create a ModelResponse object
    mock_model_response = ModelResponse(
        id="chatcmpl-test-456",
        created=1234567890,
        model="gpt-4o-mini",
        object="chat.completion",
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(
                    content="This is an async test response",
                    role="assistant",
                ),
            )
        ],
        usage=Usage(
            prompt_tokens=15,
            completion_tokens=25,
            total_tokens=40,
        ),
    )

    # Call acompletion with stream=True and mock_response as ModelResponse
    response = await acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello async"}],
        stream=True,
        mock_response=mock_model_response,
    )

    # Verify that the response is a stream
    assert response is not None

    # Collect all chunks from the stream
    chunks = []
    async for chunk in response:
        chunks.append(chunk)
        print(f"Async Chunk: {chunk}")

    # Verify we got chunks
    assert len(chunks) > 0

    # Verify the content is streamed correctly
    accumulated_content = ""
    for chunk in chunks:
        if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
            accumulated_content += chunk.choices[0].delta.content

    assert "This is an async test response" in accumulated_content or len(chunks) > 0


class TestCallTypesOCR:
    """Test that OCR call types are properly defined in CallTypes enum.

    Fixes https://github.com/BerriAI/litellm/issues/17381
    """

    def test_ocr_call_type_exists(self):
        """Test that CallTypes.ocr exists and has correct value."""
        from litellm.types.utils import CallTypes

        assert hasattr(CallTypes, "ocr")
        assert CallTypes.ocr.value == "ocr"

    def test_aocr_call_type_exists(self):
        """Test that CallTypes.aocr exists and has correct value."""
        from litellm.types.utils import CallTypes

        assert hasattr(CallTypes, "aocr")
        assert CallTypes.aocr.value == "aocr"

    def test_ocr_call_type_from_string(self):
        """Test that CallTypes can be constructed from 'ocr' string."""
        from litellm.types.utils import CallTypes

        call_type = CallTypes("ocr")
        assert call_type == CallTypes.ocr

    def test_aocr_call_type_from_string(self):
        """Test that CallTypes can be constructed from 'aocr' string.

        This is the actual use case that was failing - the OCR endpoint
        uses route_type='aocr' and guardrails try to instantiate
        CallTypes('aocr').
        """
        from litellm.types.utils import CallTypes

        call_type = CallTypes("aocr")
        assert call_type == CallTypes.aocr


def test_stream_chunk_builder_text_completion_combines_text_and_usage():
    from litellm.main import stream_chunk_builder_text_completion
    from litellm.types.utils import TextCompletionResponse

    chunks = [
        TextCompletionResponse(
            id="cmpl-1",
            object="text_completion",
            created=1,
            model="gpt-3.5-turbo-instruct",
            choices=[{"text": "Hello", "index": 0, "logprobs": None, "finish_reason": None}],
        ),
        TextCompletionResponse(
            id="cmpl-1",
            object="text_completion",
            created=1,
            model="gpt-3.5-turbo-instruct",
            choices=[{"text": " world", "index": 0, "logprobs": None, "finish_reason": "stop"}],
        ),
    ]

    response = stream_chunk_builder_text_completion(chunks=chunks, messages=[{"role": "user", "content": "say hello"}])

    assert response.choices[0].text == "Hello world"
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.prompt_tokens > 0
    assert response.usage.completion_tokens > 0
    assert response.usage.total_tokens == response.usage.prompt_tokens + response.usage.completion_tokens


def test_completion_forwards_store_and_prompt_cache_key_to_openai():
    """
    Regression test for https://github.com/BerriAI/litellm/issues/33184

    store and prompt_cache_key are documented OpenAI chat completion params that
    were accepted as supported but silently dropped before the provider request
    was built, because they were not named parameters of completion() and
    get_optional_params() the way safety_identifier is.
    """
    from openai import OpenAI

    client = OpenAI(api_key="fake-api-key")

    with patch.object(client.chat.completions.with_raw_response, "create") as mock_client:
        try:
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                store=False,
                prompt_cache_key="test-cache-key",
                client=client,
            )
        except Exception as e:
            print(e)

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs
        assert request_body["store"] is False
        assert request_body["prompt_cache_key"] == "test-cache-key"


@pytest.mark.asyncio
async def test_acompletion_forwards_store_and_prompt_cache_key_to_openai():
    """
    Async variant of the store/prompt_cache_key forwarding regression test for
    https://github.com/BerriAI/litellm/issues/33184
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key")

    with patch.object(client.chat.completions.with_raw_response, "create") as mock_client:
        try:
            await litellm.acompletion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                store=False,
                prompt_cache_key="test-cache-key",
                client=client,
            )
        except Exception as e:
            print(e)

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs
        assert request_body["store"] is False
        assert request_body["prompt_cache_key"] == "test-cache-key"


def test_completion_omits_store_and_prompt_cache_key_when_not_passed():
    """
    When store and prompt_cache_key are not passed, they must not appear in the
    outbound request body (guards against always forwarding None defaults).
    """
    from openai import OpenAI

    client = OpenAI(api_key="fake-api-key")

    with patch.object(client.chat.completions.with_raw_response, "create") as mock_client:
        try:
            litellm.completion(
                model="openai/gpt-4o",
                messages=[{"role": "user", "content": "Hello"}],
                client=client,
            )
        except Exception as e:
            print(e)

        mock_client.assert_called_once()
        request_body = mock_client.call_args.kwargs
        assert "store" not in request_body
        assert "prompt_cache_key" not in request_body


def test_completion_forwards_store_and_prompt_cache_key_to_mcp_gateway():
    """
    Regression test for the MCP gateway early-return in completion(): store and
    prompt_cache_key are named params, so they no longer travel via **kwargs and
    must be forwarded explicitly like safety_identifier and service_tier.
    """
    with patch.object(
        import_module("litellm.responses.mcp.chat_completions_handler"), "acompletion_with_mcp"
    ) as mock_mcp:
        result = litellm.completion(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": "Hello"}],
            tools=[{"type": "mcp", "server_url": "litellm_proxy"}],
            store=False,
            prompt_cache_key="test-cache-key",
        )

        result.close()
        mock_mcp.assert_called_once()
        call_kwargs = mock_mcp.call_args.kwargs
        assert call_kwargs["store"] is False
        assert call_kwargs["prompt_cache_key"] == "test-cache-key"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "aws_credential_kwargs",
    [
        {
            "aws_session_name": "litellm-gcp",
            "aws_role_name": "arn:aws:iam::123456789012:role/litellm-bedrock-role",
            "aws_web_identity_token": "oidc/google/108963886734710037768",
        },
        {
            "aws_access_key_id": "AKIASTATICKEYFORTEST",
            "aws_secret_access_key": "static-secret-key",
            "aws_session_token": "static-session-token",
        },
    ],
    ids=["web_identity", "static_keys"],
)
async def test_acompletion_forwards_aws_credentials_through_responses_bridge(
    respx_mock: respx.MockRouter, monkeypatch, aws_credential_kwargs: dict
):
    from botocore.credentials import Credentials

    from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM

    original_disable_aiohttp = litellm.disable_aiohttp_transport
    try:
        litellm.disable_aiohttp_transport = True
        monkeypatch.setenv("DISABLE_AIOHTTP_TRANSPORT", "True")
        litellm.in_memory_llm_clients_cache.flush_cache()
        monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
        monkeypatch.delenv("BEDROCK_MANTLE_API_KEY", raising=False)

        get_credentials_mock = MagicMock(return_value=Credentials("fake-key", "fake-secret"))
        monkeypatch.setattr(BaseAWSLLM, "get_credentials", get_credentials_mock)

        respx_mock.post("https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses").respond(
            json={
                "id": "resp_123",
                "object": "response",
                "created_at": 1760144904,
                "status": "completed",
                "model": "openai.gpt-5.4",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok", "annotations": []}],
                    }
                ],
            }
        )

        response = await litellm.acompletion(
            model="bedrock_mantle/openai.gpt-5.4",
            messages=[{"role": "user", "content": "hi"}],
            api_base="https://bedrock-mantle.us-east-2.api.aws/v1",
            aws_region_name="us-east-2",
            num_retries=0,
            **aws_credential_kwargs,
        )

        assert response.choices[0].message.content == "ok"
        credential_kwargs = get_credentials_mock.call_args.kwargs
        assert credential_kwargs["aws_region_name"] == "us-east-2"
        for key, value in aws_credential_kwargs.items():
            assert credential_kwargs[key] == value
        authorization = respx_mock.calls.last.request.headers["Authorization"]
        assert authorization.startswith("AWS4-HMAC-SHA256")
        assert "fake-key" in authorization
    finally:
        litellm.disable_aiohttp_transport = original_disable_aiohttp
        litellm.in_memory_llm_clients_cache.flush_cache()


_GEMINI_RESPONSE_BODY = {
    "candidates": [{"content": {"parts": [{"text": "hello"}], "role": "model"}, "finishReason": "STOP"}],
    "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
}


def _gemini_client_returning_a_reply():
    """An injected HTTP client whose post() answers like generativelanguage does."""
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    request = httpx.Request("POST", "https://generativelanguage.googleapis.com/")
    post = MagicMock(return_value=httpx.Response(200, json=_GEMINI_RESPONSE_BODY, request=request))
    return client, post


@pytest.fixture
def restore_model_registry():
    """litellm.model_cost and the provider name sets are module-global.

    register_model merges into the existing entry in place, hence the deep copy.
    """
    model_cost = copy.deepcopy(litellm.model_cost)
    openai_models = set(litellm.open_ai_chat_completion_models)
    yield
    litellm.model_cost.clear()
    litellm.model_cost.update(model_cost)
    litellm.open_ai_chat_completion_models.clear()
    litellm.open_ai_chat_completion_models.update(openai_models)


def test_openai_model_name_does_not_outrank_explicit_provider():
    """`gemini/gpt-4o` goes to Google, not to litellm's OpenAI handler.

    completion() checks `model in litellm.open_ai_chat_completion_models` ahead of
    the gemini branch, so the call used to reach the OpenAI handler carrying
    VertexGeminiConfig, whose transform_request raises NotImplementedError.
    """
    assert "gpt-4o" in litellm.open_ai_chat_completion_models
    client, post = _gemini_client_returning_a_reply()

    with patch.object(client, "post", new=post):
        response = litellm.completion(
            model="gemini/gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-api-key",
            client=client,
        )

    assert "generativelanguage.googleapis.com" in post.call_args.kwargs["url"]
    assert "models/gpt-4o" in post.call_args.kwargs["url"]
    assert response.choices[0].message.content == "hello"


def test_mislabelled_pricing_entry_does_not_reroute_provider(restore_model_registry):
    """register_model is the other way into the same failure.

    An entry claiming litellm_provider "openai" adds its name to
    open_ai_chat_completion_models, so one mislabelled price reroutes every later
    call to that model in the process.
    """
    litellm.register_model(
        {
            "gemini-2.5-pro": {
                "litellm_provider": "openai",
                "mode": "chat",
                "input_cost_per_token": 1e-06,
                "output_cost_per_token": 4e-06,
            }
        }
    )
    assert "gemini-2.5-pro" in litellm.open_ai_chat_completion_models
    client, post = _gemini_client_returning_a_reply()

    with patch.object(client, "post", new=post):
        response = litellm.completion(
            model="gemini/gemini-2.5-pro",
            messages=[{"role": "user", "content": "hello"}],
            api_key="test-api-key",
            client=client,
        )

    assert "generativelanguage.googleapis.com" in post.call_args.kwargs["url"]
    assert response.choices[0].message.content == "hello"


def test_openai_model_without_a_provider_still_routes_to_openai():
    from openai import OpenAI

    client = OpenAI(api_key="fake-key")
    raw_response = client.chat.completions.with_raw_response
    with patch.object(raw_response, "create") as mock_create, contextlib.suppress(Exception):
        litellm.completion(
            model="gpt-4o",
            messages=[{"role": "user", "content": "hello"}],
            client=client,
        )

    mock_create.assert_called()


def _openai_chat_create_kwargs(client, **completion_kwargs):
    with patch.object(client.chat.completions.with_raw_response, "create") as mock_client:
        with contextlib.suppress(Exception):
            litellm.completion(
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                cache_control_injection_points=[{"location": "message", "role": "system"}],
                client=client,
                **completion_kwargs,
            )

        mock_client.assert_called_once()
        return mock_client.call_args.kwargs


@pytest.fixture
def _no_openai_api_base_override(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.setattr(litellm, "api_base", None)


@pytest.mark.usefixtures("_no_openai_api_base_override")
def test_completion_custom_api_base_sends_no_prompt_cache_breakpoint_for_gpt_5_6():
    from openai import OpenAI

    client = OpenAI(api_key="fake-api-key", base_url="http://127.0.0.1:9/v1")
    request_body = _openai_chat_create_kwargs(client, model="gpt-5.6", api_base="http://127.0.0.1:9/v1")

    assert request_body["messages"][0] == {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}}
    assert "prompt_cache_breakpoint" not in json.dumps(request_body["messages"])
    assert "prompt_cache_options" not in json.dumps(request_body)


@pytest.mark.usefixtures("_no_openai_api_base_override")
def test_completion_custom_base_url_sends_no_prompt_cache_breakpoint_for_gpt_5_6():
    from openai import OpenAI

    client = OpenAI(api_key="fake-api-key", base_url="http://127.0.0.1:9/v1")
    request_body = _openai_chat_create_kwargs(client, model="gpt-5.6", base_url="http://127.0.0.1:9/v1")

    assert request_body["messages"][0] == {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}}
    assert "prompt_cache_breakpoint" not in json.dumps(request_body["messages"])
    assert "prompt_cache_options" not in json.dumps(request_body)


@pytest.mark.asyncio
@pytest.mark.usefixtures("_no_openai_api_base_override")
async def test_acompletion_custom_base_url_sends_no_prompt_cache_breakpoint_for_gpt_5_6():
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key="fake-api-key", base_url="http://127.0.0.1:9/v1")
    with patch.object(client.chat.completions.with_raw_response, "create") as mock_create:
        with contextlib.suppress(Exception):
            await litellm.acompletion(
                model="gpt-5.6",
                messages=[{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                cache_control_injection_points=[{"location": "message", "role": "system"}],
                client=client,
                base_url="http://127.0.0.1:9/v1",
            )

        mock_create.assert_called_once()
        request_body = mock_create.call_args.kwargs

    assert request_body["messages"][0] == {"role": "system", "content": "sys", "cache_control": {"type": "ephemeral"}}
    assert "prompt_cache_breakpoint" not in json.dumps(request_body["messages"])
    assert "prompt_cache_options" not in json.dumps(request_body)


@pytest.mark.usefixtures("_no_openai_api_base_override")
def test_completion_default_api_base_sends_prompt_cache_breakpoint_for_gpt_5_6():
    from openai import OpenAI

    client = OpenAI(api_key="fake-api-key")
    request_body = _openai_chat_create_kwargs(client, model="gpt-5.6")

    assert request_body["messages"][0]["content"] == [
        {"type": "text", "text": "sys", "prompt_cache_breakpoint": {"mode": "explicit"}}
    ]
    assert request_body["extra_body"]["prompt_cache_options"] == {"mode": "implicit"}


_SUBSCRIPTION_OAUTH_CREDENTIAL = "Bearer sk-ant-oat01-fake-subscription-token-for-testing-0123456789"


def _scoped_headers_for_oauth_request():
    from litellm.types.utils import ProviderSpecificHeader

    return [
        ProviderSpecificHeader(
            custom_llm_provider="anthropic,bedrock,vertex_ai",
            extra_headers={"anthropic-version": "2023-06-01"},
        ),
        ProviderSpecificHeader(
            custom_llm_provider="anthropic",
            extra_headers={"authorization": _SUBSCRIPTION_OAUTH_CREDENTIAL},
        ),
    ]


def _run_anthropic_hop_with_shared_headers(shared_headers):
    litellm.completion(
        model="anthropic/claude-3-5-sonnet-20240620",
        messages=[{"role": "user", "content": "Say OK"}],
        extra_headers=shared_headers,
        provider_specific_header=_scoped_headers_for_oauth_request(),
        api_key="sk-fake-anthropic-key",
        mock_response="OK",
    )


def test_completion_does_not_mutate_caller_supplied_headers():
    shared_headers = {"x-tenant": "acme"}

    _run_anthropic_hop_with_shared_headers(shared_headers)

    assert shared_headers == {"x-tenant": "acme"}


def test_anthropic_oauth_credential_does_not_persist_into_next_provider_hop():
    shared_headers = {"x-tenant": "acme"}

    _run_anthropic_hop_with_shared_headers(shared_headers)

    leaked = [name for name, value in shared_headers.items() if value == _SUBSCRIPTION_OAUTH_CREDENTIAL]
    assert leaked == []
    assert "anthropic-version" not in shared_headers


STREAM_COST_MODEL = "gpt-4o"
STREAMED_USAGE = {"prompt_tokens": 137, "completion_tokens": 42, "total_tokens": 179}


def _text_chunk(content, finish_reason=None, usage=None):
    chunk = {
        "id": "chatcmpl-stream-cost",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": STREAM_COST_MODEL,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        chunk["usage"] = usage
    return chunk


def _priced_at(prompt_tokens, completion_tokens):
    prices = litellm.model_cost[STREAM_COST_MODEL]
    return prompt_tokens * prices["input_cost_per_token"] + completion_tokens * prices["output_cost_per_token"]


@pytest.fixture
def local_cost_map(monkeypatch):
    """The prices these tests assert are the checked-in ones. Setting the environment
    variable alone does not reload the map, so pin the map itself.

    Prices are read through two separate lru_caches, so pinning ``model_cost`` is not
    enough on its own: an entry warmed against the network-fetched map keeps its old
    prices and billing reads those while the assertions read the pinned map.
    ``_invalidate_model_cost_lowercase_map`` clears both caches, where
    ``get_model_info.cache_clear`` reaches only one. Invalidate on the way in and out
    so entries never leak across tests in either direction."""
    from litellm.utils import _invalidate_model_cost_lowercase_map

    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
    _invalidate_model_cost_lowercase_map()
    yield
    _invalidate_model_cost_lowercase_map()


def test_a_streamed_response_bills_the_usage_the_provider_reported(local_cost_map):
    rebuilt = litellm.stream_chunk_builder(
        chunks=[
            _text_chunk("Hello"),
            _text_chunk(" there"),
            _text_chunk(None, finish_reason="stop", usage=STREAMED_USAGE),
        ],
        messages=[{"role": "user", "content": "hi"}],
    )

    assert rebuilt.choices[0].message.content == "Hello there"
    assert rebuilt.usage.prompt_tokens == STREAMED_USAGE["prompt_tokens"]
    assert rebuilt.usage.completion_tokens == STREAMED_USAGE["completion_tokens"]

    cost = litellm.completion_cost(completion_response=rebuilt, model=STREAM_COST_MODEL)

    assert cost == pytest.approx(_priced_at(137, 42))


def test_streaming_and_not_streaming_bill_the_same_usage_the_same(local_cost_map):
    rebuilt = litellm.stream_chunk_builder(
        chunks=[
            _text_chunk("Hello"),
            _text_chunk(" there"),
            _text_chunk(None, finish_reason="stop", usage=STREAMED_USAGE),
        ],
        messages=[{"role": "user", "content": "hi"}],
    )
    whole = litellm.ModelResponse(
        id="chatcmpl-stream-cost",
        model=STREAM_COST_MODEL,
        object="chat.completion",
        created=1700000000,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello there"},
                "finish_reason": "stop",
            }
        ],
        usage=STREAMED_USAGE,
    )

    assert litellm.completion_cost(completion_response=rebuilt, model=STREAM_COST_MODEL) == pytest.approx(
        litellm.completion_cost(completion_response=whole, model=STREAM_COST_MODEL)
    )


def test_a_stream_that_reported_no_usage_is_still_billed(local_cost_map):
    rebuilt = litellm.stream_chunk_builder(
        chunks=[
            _text_chunk("Hello"),
            _text_chunk(" there"),
            _text_chunk(None, finish_reason="stop"),
        ],
        messages=[{"role": "user", "content": "hi"}],
    )

    assert rebuilt.usage.prompt_tokens > 0
    assert rebuilt.usage.completion_tokens > 0

    cost = litellm.completion_cost(completion_response=rebuilt, model=STREAM_COST_MODEL)

    assert cost > 0
    assert cost == pytest.approx(_priced_at(rebuilt.usage.prompt_tokens, rebuilt.usage.completion_tokens))


@pytest.mark.asyncio
async def test_acompletion_resolves_provider_from_api_base():
    response = await litellm.acompletion(
        model="deepseek-chat",
        api_base="https://api.deepseek.com/v1",
        api_key="fake-key",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="resolved",
    )

    assert response.choices[0].message.content == "resolved"


@dataclass(frozen=True, slots=True)
class _RecordedSpeechSuccess:
    call_type: str | None
    spend_metadata: Mapping[str, object]
    response_cost: float | None
    logged_response_cost: float | None


def _record_speech_success(payload: dict[str, object]) -> _RecordedSpeechSuccess:
    call_type: Final = payload.get("call_type")
    response_cost: Final = payload.get("response_cost")
    logging_payload: Final = payload.get("standard_logging_object")
    logged_cost: Final = logging_payload.get("response_cost") if isinstance(logging_payload, dict) else None
    return _RecordedSpeechSuccess(
        call_type=call_type if isinstance(call_type, str) else None,
        spend_metadata=get_litellm_metadata_from_kwargs(payload),
        response_cost=response_cost if isinstance(response_cost, float) else None,
        logged_response_cost=logged_cost if isinstance(logged_cost, float) else None,
    )


class _SuccessEventRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[_RecordedSpeechSuccess] = []  # mutable-ok: test recorder of success-callback events

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.events.append(_record_speech_success(kwargs))


async def _wait_for_success_event(recorder: _SuccessEventRecorder, call_type: str) -> _RecordedSpeechSuccess:
    for _ in range(100):
        if (event := next((e for e in recorder.events if e.call_type == call_type), None)) is not None:
            return event
        await asyncio.sleep(0.05)
    pytest.fail(f"no {call_type} success event; got {[e.call_type for e in recorder.events]}")


def _gemini_tts_generate_content_response() -> dict[str, object]:
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "audio/L16;codec=pcm;rate=24000",
                                "data": base64.b64encode(b"pcm-audio-bytes").decode(),
                            }
                        }
                    ],
                    "role": "model",
                },
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 5,
            "candidatesTokenCount": 60,
            "totalTokenCount": 65,
            "promptTokensDetails": [{"modality": "TEXT", "tokenCount": 5}],
            "candidatesTokensDetails": [{"modality": "AUDIO", "tokenCount": 60}],
        },
        "modelVersion": "gemini-2.5-flash-preview-tts",
    }


@pytest.mark.asyncio
async def test_aspeech_gemini_bridge_keeps_proxy_metadata_for_spend_tracking(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    recorder: Final = _SuccessEventRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    mock_route: Final = respx_mock.post(
        url__regex=r"https://generativelanguage\.googleapis\.com/v1beta/models/gemini-2\.5-flash-preview-tts:generateContent.*"
    ).mock(return_value=httpx.Response(200, json=_gemini_tts_generate_content_response()))

    await litellm.aspeech(
        model="gemini/gemini-2.5-flash-preview-tts",
        input="spend tracking check",
        voice="Kore",
        api_key="fake-gemini-key",
        metadata={"user_api_key": "hashed-virtual-key", "user_api_key_user_id": "user-1"},
    )

    assert mock_route.called
    assert mock_route.calls.last.request.headers["x-goog-api-key"] == "fake-gemini-key"
    speech_event: Final = await _wait_for_success_event(recorder, call_type="aspeech")
    assert speech_event.spend_metadata["user_api_key"] == "hashed-virtual-key"
    assert speech_event.spend_metadata["user_api_key_user_id"] == "user-1"
    expected_prompt_cost, expected_completion_cost = litellm.cost_per_token(
        model="gemini/gemini-2.5-flash-preview-tts",
        usage_object=Usage(prompt_tokens=5, completion_tokens=60, total_tokens=65),
    )
    expected_cost: Final = expected_prompt_cost + expected_completion_cost
    assert expected_cost > 0
    assert speech_event.response_cost == pytest.approx(expected_cost)
    assert speech_event.logged_response_cost == pytest.approx(expected_cost)


def _stream_builder_text_chunk(model: str, content: str, finish_reason: str | None = None) -> ModelResponseStream:
    return ModelResponseStream(
        id="chatcmpl-cost",
        created=1724900000,
        model=model,
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(finish_reason=finish_reason, index=0, delta=Delta(content=content, role="assistant"))
        ],
    )


def test_stream_chunk_builder_sets_hidden_response_cost_for_known_model():
    chunks: Final = [
        _stream_builder_text_chunk("gpt-4o", "Hello "),
        _stream_builder_text_chunk("gpt-4o", "world.", finish_reason="stop"),
    ]

    response: Final = litellm.stream_chunk_builder(chunks=chunks, messages=[{"role": "user", "content": "hi"}])

    assert response is not None
    prompt_cost, completion_cost = litellm.cost_per_token(model="gpt-4o", usage_object=response.usage)
    expected_cost: Final = prompt_cost + completion_cost
    assert expected_cost > 0
    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)


def test_stream_chunk_builder_unknown_model_leaves_response_cost_unset():
    chunks: Final = [
        _stream_builder_text_chunk("totally-unknown-model-xyz", "Hello "),
        _stream_builder_text_chunk("totally-unknown-model-xyz", "world.", finish_reason="stop"),
    ]

    response: Final = litellm.stream_chunk_builder(chunks=chunks, messages=[{"role": "user", "content": "hi"}])

    assert response is not None
    assert response._hidden_params.get("response_cost") is None
    assert response.choices[0].message.content == "Hello world."


def test_stream_chunk_builder_prices_proxy_alias_via_model_map():
    chunks: Final = [
        _stream_builder_text_chunk("claude-opus-5", "Hello "),
        _stream_builder_text_chunk("claude-opus-5", "world.", finish_reason="stop"),
    ]
    for chunk in chunks:
        chunk._hidden_params = {"custom_llm_provider": "openai"}

    response: Final = litellm.stream_chunk_builder(chunks=chunks, messages=[{"role": "user", "content": "hi"}])

    assert response is not None
    assert response._hidden_params["custom_llm_provider"] == "openai"
    prompt_cost, completion_cost = litellm.cost_per_token(model="claude-opus-5", usage_object=response.usage)
    expected_cost: Final = prompt_cost + completion_cost
    assert expected_cost > 0
    assert response._hidden_params["response_cost"] == pytest.approx(expected_cost)


def _stream_builder_logging_obj(model: str = "gpt-4o", custom_llm_provider: str = "openai") -> LiteLLMLogging:
    logging_obj: Final = LiteLLMLogging(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="test-call-id",
        function_id="test-function-id",
    )
    logging_obj.update_environment_variables(
        model=model,
        user=None,
        optional_params={},
        litellm_params={"custom_llm_provider": custom_llm_provider},
        custom_llm_provider=custom_llm_provider,
    )
    return logging_obj


def test_stream_chunk_builder_stamps_streaming_usage_cost_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "include_cost_in_streaming_usage", False)
    chunks: Final = [
        _stream_builder_text_chunk("gpt-4o", "Hello "),
        _stream_builder_text_chunk("gpt-4o", "world.", finish_reason="stop"),
    ]

    response: Final = litellm.stream_chunk_builder(
        chunks=chunks, messages=[{"role": "user", "content": "hi"}], logging_obj=_stream_builder_logging_obj()
    )

    assert response is not None
    usage_cost: Final = getattr(response.usage, "cost", None)
    assert usage_cost is not None
    assert usage_cost > 0
    assert response._hidden_params["response_cost"] == pytest.approx(usage_cost)


def test_stream_chunk_builder_skips_stamp_when_cost_is_unpriceable():
    import time as time_module

    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLogging

    logging_obj: Final = LiteLLMLogging(
        model="us.anthropic.claude-opus-5",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="completion",
        start_time=time_module.time(),
        litellm_call_id="stream-builder-alias-unpriceable",
        function_id="1",
    )
    logging_obj.model_call_details["custom_llm_provider"] = "bedrock"
    logging_obj.optional_params = {}
    usage_chunk: Final = _stream_builder_text_chunk("bedrock-claude-opus-5", "")
    usage_chunk.usage = Usage(prompt_tokens=40, completion_tokens=5, total_tokens=45)
    chunks: Final = [
        _stream_builder_text_chunk("bedrock-claude-opus-5", "Hello ", finish_reason="stop"),
        usage_chunk,
    ]

    response: Final = litellm.stream_chunk_builder(
        chunks=chunks, messages=[{"role": "user", "content": "hi"}], logging_obj=logging_obj
    )

    assert response is not None
    assert getattr(response.usage, "cost", None) is None
    assert response._hidden_params.get("response_cost") is None


def test_stream_chunk_builder_keeps_provider_reported_usage_cost():
    usage_chunk: Final = _stream_builder_text_chunk("gpt-4o", "")
    usage_chunk.usage = Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.5)
    chunks: Final = [
        _stream_builder_text_chunk("gpt-4o", "Hello "),
        _stream_builder_text_chunk("gpt-4o", "world.", finish_reason="stop"),
        usage_chunk,
    ]

    response: Final = litellm.stream_chunk_builder(
        chunks=chunks, messages=[{"role": "user", "content": "hi"}], logging_obj=_stream_builder_logging_obj()
    )

    assert response is not None
    assert getattr(response.usage, "cost", None) == pytest.approx(0.5)
    assert response._hidden_params["response_cost"] == pytest.approx(0.5)


def test_stream_chunk_builder_prices_alias_from_openai_sdk_usage_chunk():
    from openai.types.completion_usage import CompletionUsage

    usage_chunk: Final = _stream_builder_text_chunk("mantle-claude", "")
    usage_chunk.usage = CompletionUsage(prompt_tokens=20, completion_tokens=60, total_tokens=80, cost=0.000704)
    assert type(usage_chunk.usage) is CompletionUsage
    chunks: Final = [
        _stream_builder_text_chunk("mantle-claude", "Hello "),
        _stream_builder_text_chunk("mantle-claude", "world.", finish_reason="stop"),
        usage_chunk,
    ]

    response: Final = litellm.stream_chunk_builder(chunks=chunks, messages=[{"role": "user", "content": "hi"}])

    assert response is not None
    assert response.usage.prompt_tokens == 20
    assert response.usage.completion_tokens == 60
    assert getattr(response.usage, "cost", None) == pytest.approx(0.000704)
    assert response._hidden_params["response_cost"] == pytest.approx(0.000704)


def test_stream_chunk_builder_leaves_xai_reported_cost_to_the_calculator(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "cost_margin_config", {"xai": 0.5})
    usage_chunk: Final = _stream_builder_text_chunk("grok-4", "")
    usage_chunk.usage = Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7, cost=0.42)
    chunks: Final = [
        _stream_builder_text_chunk("grok-4", "Hello "),
        _stream_builder_text_chunk("grok-4", "world.", finish_reason="stop"),
        usage_chunk,
    ]
    logging_obj: Final = _stream_builder_logging_obj(model="grok-4", custom_llm_provider="xai")

    response: Final = litellm.stream_chunk_builder(
        chunks=chunks, messages=[{"role": "user", "content": "hi"}], logging_obj=logging_obj
    )

    assert response is not None
    assert getattr(response.usage, "cost", None) == pytest.approx(0.42)
    assert response._hidden_params.get("response_cost") is None
    assert logging_obj.response_cost_calculator(result=response) == pytest.approx(0.63)


def test_speech_mistral_dispatches_and_decodes_audio(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-mistral-test")
    audio_bytes: Final = b"ID3-fake-mp3-bytes"
    mock_route: Final = respx_mock.post("https://api.mistral.ai/v1/audio/speech").mock(
        return_value=httpx.Response(200, json={"audio_data": base64.b64encode(audio_bytes).decode()})
    )

    response: Final = litellm.speech(
        model="mistral/voxtral-mini-tts-2603",
        input="hello from litellm",
        voice="en_paul_neutral",
        response_format="wav",
        speed=2,
        instructions="sound cheerful",
    )

    assert mock_route.called
    request_body: Final = json.loads(mock_route.calls.last.request.content)
    assert request_body == {
        "model": "voxtral-mini-tts-2603",
        "input": "hello from litellm",
        "voice_id": "en_paul_neutral",
        "response_format": "wav",
    }
    assert mock_route.calls.last.request.headers["authorization"] == "Bearer sk-mistral-test"
    assert response.content == audio_bytes


def test_speech_mistral_routes_to_configured_api_base(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "sk-mistral-test")
    audio_bytes: Final = b"ID3-gateway-bytes"
    gateway_route: Final = respx_mock.post("https://mistral.gateway.internal/v1/audio/speech").mock(
        return_value=httpx.Response(200, json={"audio_data": base64.b64encode(audio_bytes).decode()})
    )

    response: Final = litellm.speech(
        model="mistral/voxtral-mini-tts-2603",
        input="hello from litellm",
        voice="en_paul_neutral",
        api_base="https://mistral.gateway.internal",
    )

    assert gateway_route.called
    assert response.content == audio_bytes


FOUNDRY_HOST: Final = "https://my-project.services.ai.azure.com"


def test_azure_ai_transcription_on_a_foundry_host_uses_the_azure_openai_deployment_route(
    respx_mock: respx.MockRouter,
):
    route: Final = respx_mock.post(
        url__regex=r"https://my-project\.services\.ai\.azure\.com/openai/deployments/whisper-1/audio/transcriptions\?api-version=.+"
    ).mock(return_value=httpx.Response(200, json={"text": "hello"}))

    response: Final = litellm.transcription(
        model="azure_ai/whisper-1",
        file=("tone.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav"),
        api_base=FOUNDRY_HOST,
        api_key="fake-key",
    )

    assert route.called
    assert response.text == "hello"


def test_azure_ai_speech_on_a_foundry_host_uses_the_azure_openai_deployment_route(
    respx_mock: respx.MockRouter,
):
    route: Final = respx_mock.post(
        url__regex=r"https://my-project\.services\.ai\.azure\.com/openai/deployments/tts-1/audio/speech\?api-version=.+"
    ).mock(return_value=httpx.Response(200, content=b"mp3-bytes"))

    response: Final = litellm.speech(
        model="azure_ai/tts-1",
        input="hello",
        voice="alloy",
        api_base=FOUNDRY_HOST,
        api_key="fake-key",
    )

    assert route.called
    assert response.content == b"mp3-bytes"


GROQ_INTERNAL_BASE: Final = "https://groq.gateway.internal/openai/v1"
GROQ_WAV_FILE: Final = ("tone.wav", b"RIFF\x00\x00\x00\x00WAVE", "audio/wav")


def test_groq_transcription_honors_base_url_alias(respx_mock: respx.MockRouter):
    route: Final = respx_mock.post(f"{GROQ_INTERNAL_BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello"})
    )

    response: Final = litellm.transcription(
        model="groq/whisper-large-v3",
        file=GROQ_WAV_FILE,
        base_url=GROQ_INTERNAL_BASE,
        api_key="fake-key",
    )

    assert route.called
    assert response.text == "hello"


async def test_groq_atranscription_honors_base_url_alias(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(f"{GROQ_INTERNAL_BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello"})
    )

    response: Final = await litellm.atranscription(
        model="groq/whisper-large-v3",
        file=GROQ_WAV_FILE,
        base_url=GROQ_INTERNAL_BASE,
        api_key="fake-key",
    )

    assert route.called
    assert response.text == "hello"


def test_groq_speech_honors_base_url_alias(respx_mock: respx.MockRouter):
    route: Final = respx_mock.post(f"{GROQ_INTERNAL_BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=b"mp3-bytes")
    )

    response: Final = litellm.speech(
        model="groq/playai-tts",
        input="hello",
        voice="Fritz-PlayAI",
        base_url=GROQ_INTERNAL_BASE,
        api_key="fake-key",
    )

    assert route.called
    assert response.content == b"mp3-bytes"


FORWARDED_CLIENT_HEADERS: Final = {"x-forwarded-for": "10.0.0.1", "x-amzn-trace-id": "Root=1-lit7694"}


def _chat_completion_json() -> Mapping[str, object]:
    return {
        "id": "chatcmpl-lit7694",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.4",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chat_completion_sse() -> bytes:
    chunk: Final = {
        "id": "chatcmpl-lit7694",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-5.4",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
    }
    return f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n".encode()


@pytest.mark.parametrize("stream", [False, True])
def test_bridged_responses_with_openai_http_handler_keeps_forwarded_headers_out_of_the_body(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, stream: bool
):
    monkeypatch.setenv("EXPERIMENTAL_OPENAI_BASE_LLM_HTTP_HANDLER", "true")
    route: Final = respx_mock.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=_chat_completion_sse(), headers={"content-type": "text/event-stream"})
        if stream
        else httpx.Response(200, json=_chat_completion_json())
    )

    response: Final = litellm.responses(
        model="openai/gpt-5.4",
        input="Reply with the single word ok",
        stream=stream,
        use_chat_completions_api=True,
        headers=dict(FORWARDED_CLIENT_HEADERS),
        api_key="sk-test",
    )
    if stream:
        list(response)

    assert route.called
    request: Final = route.calls.last.request
    body: Final = json.loads(request.content)
    assert "extra_headers" not in body
    assert body["model"] == "gpt-5.4"
    assert {k: request.headers[k] for k in FORWARDED_CLIENT_HEADERS} == FORWARDED_CLIENT_HEADERS


@pytest.mark.parametrize("http2_on", [True, False])
def test_aiohttp_openai_warns_only_when_http2_enabled(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, http2_on: bool
):
    from litellm.main import base_llm_aiohttp_handler

    monkeypatch.setattr(litellm, "http2", http2_on)
    monkeypatch.delenv("LITELLM_HTTP2", raising=False)

    handler_completion: Final = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(base_llm_aiohttp_handler, "completion", handler_completion)

    with caplog.at_level(logging.WARNING, logger="LiteLLM"):
        litellm.completion(
            model="aiohttp_openai/gpt-4o",
            messages=[{"role": "user", "content": "hi"}],
            api_key="sk-test",
        )

    assert handler_completion.called
    warned: Final = "aiohttp_openai/ always uses aiohttp" in caplog.text
    assert warned is http2_on


@pytest.mark.parametrize("tool_choice", [{"type": "bogus"}, {"name": "lookup_fruit"}, {"type": "file_search"}])
def test_completion_rejects_untranslatable_tool_choice_with_a_400(tool_choice):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.completion(
            model="anthropic/claude-haiku-4-5",
            messages=[{"role": "user", "content": "Which fruit is red?"}],
            tools=[{"type": "function", "function": {"name": "lookup_fruit", "parameters": {"type": "object"}}}],
            tool_choice=tool_choice,
            api_key="sk-unused",
        )
    assert exc_info.value.status_code == 400
    assert f"tool_choice={tool_choice}" in str(exc_info.value)


@pytest.mark.parametrize("raw", ["sixty-four", 0, -1])
def test_completion_rejects_an_invalid_stream_chunk_size_with_a_400_naming_the_param(raw: object) -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.completion(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream_chunk_size=raw,
            mock_response="unused",
        )
    assert exc_info.value.status_code == 400
    assert exc_info.value.param == "stream_chunk_size"
    assert f"Invalid stream_chunk_size={raw!r}: expected a positive integer of at most 18 digits" in str(exc_info.value)


class _PromptHookRecorder(CustomPromptManagement):
    def __init__(self, on_prompt: MagicMock) -> None:
        super().__init__()
        self.on_prompt: Final = on_prompt

    def get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        prompt_spec: PromptSpec | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[str, list[AllMessageValues], dict]:
        self.on_prompt("sync")
        return model, messages, non_default_params

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: list[AllMessageValues],
        non_default_params: dict,
        prompt_id: str | None,
        prompt_variables: dict | None,
        dynamic_callback_params: StandardCallbackDynamicParams,
        litellm_logging_obj: LiteLLMLogging,
        prompt_spec: PromptSpec | None = None,
        tools: list[dict] | None = None,
        prompt_label: str | None = None,
        prompt_version: int | None = None,
        ignore_prompt_manager_model: bool | None = False,
        ignore_prompt_manager_optional_params: bool | None = False,
    ) -> tuple[str, list[AllMessageValues], dict]:
        self.on_prompt("async")
        return model, messages, non_default_params


async def _call_completion(is_async: bool, **kwargs: object) -> None:
    if is_async:
        await litellm.acompletion(**kwargs)
    else:
        litellm.completion(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async,hook", [(False, "sync"), (True, "async")], ids=["completion", "acompletion"])
async def test_the_prompt_hook_runs_when_stream_chunk_size_is_valid(
    monkeypatch: pytest.MonkeyPatch, is_async: bool, hook: str
) -> None:
    on_prompt: Final = MagicMock()
    monkeypatch.setattr(litellm, "callbacks", [_PromptHookRecorder(on_prompt)])

    await _call_completion(
        is_async,
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        prompt_id="greeting",
        stream_chunk_size=64,
        mock_response="hi",
    )

    on_prompt.assert_any_call(hook)


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True], ids=["completion", "acompletion"])
async def test_an_invalid_stream_chunk_size_is_rejected_before_any_prompt_hook_runs(
    monkeypatch: pytest.MonkeyPatch, is_async: bool
) -> None:
    on_prompt: Final = MagicMock()
    monkeypatch.setattr(litellm, "callbacks", [_PromptHookRecorder(on_prompt)])

    with pytest.raises(litellm.BadRequestError):
        await _call_completion(
            is_async,
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "hi"}],
            prompt_id="greeting",
            stream_chunk_size="sixty-four",
            mock_response="hi",
        )

    on_prompt.assert_not_called()


def _completion_logging_obj(call_id: str) -> LiteLLMLogging:
    return LiteLLMLogging(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="completion",
        start_time=datetime(2026, 1, 1),
        litellm_call_id=call_id,
        function_id=f"{call_id}-function",
    )


def test_completion_carries_the_control_options_into_the_logged_litellm_params() -> None:
    logging_obj: Final = _completion_logging_obj("control-params")
    litellm.completion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream_chunk_size=64,
        mock_response="hi",
        litellm_logging_obj=logging_obj,
    )
    assert stored_control_options(logging_obj.litellm_params) == ControlOptions(stream_chunk_size=64)


def test_completion_ignores_a_caller_supplied_control_options_key() -> None:
    logging_obj: Final = _completion_logging_obj("control-params-injection")
    litellm.completion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        mock_response="hi",
        litellm_logging_obj=logging_obj,
        **{CONTROL_OPTIONS_KEY: {"stream_chunk_size": 1}},
    )
    assert stored_control_options(logging_obj.litellm_params) == ControlOptions()


@pytest.mark.parametrize("drop_params", [True, "true"])
def test_drop_params_drops_an_invalid_stream_chunk_size_instead_of_rejecting_it(drop_params: object) -> None:
    logging_obj: Final = _completion_logging_obj(f"drop-params-{drop_params}")
    litellm.completion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream_chunk_size="sixty-four",
        drop_params=drop_params,
        mock_response="hi",
        litellm_logging_obj=logging_obj,
    )
    assert stored_control_options(logging_obj.litellm_params) == ControlOptions()


def test_drop_params_keeps_a_dropped_stream_chunk_size_out_of_the_provider_request(
    respx_mock: respx.MockRouter,
) -> None:
    api_base: Final = "http://localhost:12346/v1"
    mock_route: Final = respx_mock.post(url__regex=rf"{api_base}/chat/completions.*").mock(
        return_value=httpx.Response(
            status_code=200,
            json={
                "id": "chatcmpl-drop",
                "object": "chat.completion",
                "created": 1712697600,
                "model": "gpt-4.1-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )

    litellm.completion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        api_base=api_base,
        api_key="fake_openai_api_key",
        stream_chunk_size="sixty-four",
        drop_params=True,
    )

    assert mock_route.called
    sent: Final = json.loads(respx_mock.calls[0].request.content)
    assert "stream_chunk_size" not in sent, sent
    assert sent["model"] == "gpt-4.1-mini"


@pytest.mark.asyncio
async def test_global_drop_params_drops_an_invalid_stream_chunk_size_on_acompletion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "drop_params", True)
    logging_obj: Final = _completion_logging_obj("global-drop-params")
    await litellm.acompletion(
        model="openai/gpt-4.1-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream_chunk_size=0,
        mock_response="hi",
        litellm_logging_obj=logging_obj,
    )
    assert stored_control_options(logging_obj.litellm_params) == ControlOptions()


def test_completion_rejects_an_invalid_stream_chunk_size_before_the_mcp_gateway() -> None:
    with pytest.raises(litellm.BadRequestError) as exc_info:
        litellm.completion(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "mcp", "server_label": "gateway", "server_url": "litellm_proxy"}],
            stream_chunk_size="sixty-four",
        )
    assert exc_info.value.param == "stream_chunk_size"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("missing_tenacity", [False, True])
@pytest.mark.parametrize("route", ["bedrock", "bedrock/invoke"])
async def test_bedrock_stream_missing_dependency_remains_actionable_with_retries(
    monkeypatch, use_async, missing_tenacity, route
):
    import builtins

    original_import = builtins.__import__

    def import_without_aws_or_retry_dependencies(name, *args, **kwargs):
        if name.split(".")[0] == "botocore" or (name == "tenacity" and missing_tenacity):
            raise ModuleNotFoundError(name=name.split(".")[0])
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_aws_or_retry_dependencies)
    monkeypatch.setattr(litellm, "num_retries", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock as upstream:
        response = upstream.post(url__regex=r"https://bedrock-test\.invalid/.*").respond(200, content=b"")
        arguments = dict(
            model=f"{route}/anthropic.claude-3-sonnet-20240229-v1:0",
            messages=[{"role": "user", "content": "ping"}],
            api_key="test-bearer",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint="https://bedrock-test.invalid",
            stream=True,
            num_retries=1,
        )
        if use_async:
            with pytest.raises(ImportError, match="pip install boto3"):
                await litellm.acompletion(**arguments)
        else:
            with pytest.raises(ImportError, match="pip install boto3"):
                litellm.completion(**arguments)
        assert response.call_count == 1


def test_drop_params_false_still_rejects_an_invalid_stream_chunk_size() -> None:
    with pytest.raises(litellm.BadRequestError):
        litellm.completion(
            model="openai/gpt-4.1-mini",
            messages=[{"role": "user", "content": "hi"}],
            stream_chunk_size="sixty-four",
            drop_params=False,
            mock_response="hi",
        )


def test_acompletion_params():
    import inspect
    from litellm.types.completion import CompletionRequest

    acompletion_params_odict = inspect.signature(acompletion).parameters
    completion_params_dict = inspect.signature(completion).parameters

    acompletion_params = {
        name: param.annotation for name, param in acompletion_params_odict.items()
    }
    completion_params = {
        name: param.annotation for name, param in completion_params_dict.items()
    }

    keys_acompletion = set(acompletion_params.keys())
    keys_completion = set(completion_params.keys())

    print(keys_acompletion)
    print("\n\n\n")
    print(keys_completion)

    print("diff=", keys_completion - keys_acompletion)

    # Assert that the parameters are the same
    if keys_acompletion != keys_completion:
        pytest.fail(
            "The parameters of the litellm.acompletion function and litellm.completion are not the same. "
            f"Completion has extra keys: {keys_completion - keys_acompletion}"
        )


def _openai_mock_response(*args: object, **kwargs: object) -> MagicMock:
    new_response: Final = MagicMock()
    new_response.headers = {"hello": "world"}
    response_object: Final = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0125",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    pydantic_response: Final = ChatCompletion.model_validate(response_object)
    setattr(pydantic_response.choices[0].message, "role", None)
    new_response.parse.return_value = pydantic_response
    return new_response


def test_null_role_response():
    """
    Test if the api returns 'null' role, 'assistant' role is still returned
    """
    import openai

    openai_client = openai.OpenAI()
    with patch.object(
        openai_client.chat.completions, "create", side_effect=_openai_mock_response
    ) as mock_response:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey! how's it going?"}],
            client=openai_client,
        )
        print(f"response: {response}")

        assert response.id == "chatcmpl-123"

        assert response.choices[0].message.role == "assistant"


def test_parse_xml_params():
    from litellm.litellm_core_utils.prompt_templates.factory import parse_xml_params

    ## SCENARIO 1 ## - W/ ARRAY
    xml_content = """<invoke><tool_name>return_list_of_str</tool_name>\n<parameters>\n<value>\n<item>apple</item>\n<item>banana</item>\n<item>orange</item>\n</value>\n</parameters></invoke>"""
    json_schema = {
        "properties": {
            "value": {
                "items": {"type": "string"},
                "title": "Value",
                "type": "array",
            }
        },
        "required": ["value"],
        "type": "object",
    }
    response = parse_xml_params(xml_content=xml_content, json_schema=json_schema)

    print(f"response: {response}")
    assert response["value"] == ["apple", "banana", "orange"]

    ## SCENARIO 2 ## - W/OUT ARRAY
    xml_content = """<invoke><tool_name>get_current_weather</tool_name>\n<parameters>\n<location>Boston, MA</location>\n<unit>fahrenheit</unit>\n</parameters></invoke>"""
    json_schema = {
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
        "required": ["location"],
    }

    response = parse_xml_params(xml_content=xml_content, json_schema=json_schema)

    print(f"response: {response}")
    assert response["location"] == "Boston, MA"
    assert response["unit"] == "fahrenheit"


def test_completion_perplexity_api():
    try:
        response_object = {
            "id": "a8f37485-026e-45da-81a9-cf0184896840",
            "model": "llama-3-sonar-small-32k-online",
            "created": 1722186391,
            "usage": {"prompt_tokens": 17, "completion_tokens": 65, "total_tokens": 82},
            "citations": [
                "https://www.sciencedirect.com/science/article/pii/S007961232200156X",
                "https://www.britannica.com/event/World-War-II",
                "https://www.loc.gov/classroom-materials/united-states-history-primary-source-timeline/great-depression-and-world-war-ii-1929-1945/world-war-ii/",
                "https://www.nationalww2museum.org/war/topics/end-world-war-ii-1945",
                "https://en.wikipedia.org/wiki/World_War_II",
            ],
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "World War II was won by the Allied powers, which included the United States, the Soviet Union, Great Britain, France, China, and other countries. The war concluded with the surrender of Germany on May 8, 1945, and Japan on September 2, 1945[2][3][4].",
                    },
                    "delta": {"role": "assistant", "content": ""},
                }
            ],
        }

        from openai import OpenAI
        from openai.types.chat.chat_completion import ChatCompletion

        pydantic_obj = ChatCompletion(**response_object)

        def _return_pydantic_obj(*args, **kwargs):
            new_response = MagicMock()
            new_response.headers = {"hello": "world"}

            new_response.parse.return_value = pydantic_obj
            return new_response

        openai_client = OpenAI()

        with patch.object(
            openai_client.chat.completions.with_raw_response,
            "create",
            side_effect=_return_pydantic_obj,
        ) as mock_client:
            # litellm.set_verbose= True
            messages = [
                {"role": "system", "content": "You're a good bot"},
                {
                    "role": "user",
                    "content": "Hey",
                },
                {
                    "role": "user",
                    "content": "Hey",
                },
            ]
            response = completion(
                model="mistral-7b-instruct",
                messages=messages,
                api_base="https://api.perplexity.ai",
                client=openai_client,
            )
            print(response)
            assert hasattr(response, "citations")
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize(
    "provider", ["openai", "lm_studio", "llamafile"]
)  # "vertex_ai", hosted_vllm removed - no longer uses OpenAI client
@pytest.mark.asyncio
async def test_openai_compatible_custom_api_base(provider):
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": "Hello world",
        }
    ]
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.chat.completions, "create", new=MagicMock()
    ) as mock_call:
        try:
            completion(
                model="{provider}/my-vllm-model".format(provider=provider),
                messages=messages,
                response_format={"type": "json_object"},
                client=openai_client,
                api_base="my-custom-api-base",
                hello="world",
            )
        except Exception as e:
            print(e)

        mock_call.assert_called_once()

        print("Call KWARGS - {}".format(mock_call.call_args.kwargs))

        assert "hello" in mock_call.call_args.kwargs["extra_body"]


@pytest.mark.parametrize(
    "provider",
    [
        "openai",
        "llamafile",
    ],
)  # "vertex_ai", hosted_vllm removed - no longer uses OpenAI client
@pytest.mark.asyncio
async def test_openai_compatible_custom_api_video(provider):
    litellm.set_verbose = True
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What do you see in this video?",
                },
                {
                    "type": "video_url",
                    "video_url": {"url": "https://www.youtube.com/watch?v=29_ipKNI8I0"},
                },
            ],
        }
    ]
    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.chat.completions, "create", new=MagicMock()
    ) as mock_call:
        try:
            completion(
                model="{provider}/my-vllm-model".format(provider=provider),
                messages=messages,
                response_format={"type": "json_object"},
                client=openai_client,
                api_base="my-custom-api-base",
            )
        except Exception as e:
            print(e)

        mock_call.assert_called_once()


def test_ollama_image():
    """
    Test that datauri prefixes are removed, JPEG/PNG images are passed
    through, and other image formats are converted to JPEG.  Non-image
    data is untouched.
    """

    import base64

    from PIL import Image

    sent_images = []

    def mock_post(url, **kwargs):
        sent_images.append(json.loads(kwargs["data"])["images"])
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {"response": "a black pixel"}
        return mock_response

    def make_b64image(format):
        image = Image.new(mode="RGB", size=(1, 1))
        image_buffer = io.BytesIO()
        image.save(image_buffer, format)
        return base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    jpeg_image = make_b64image("JPEG")
    webp_image = make_b64image("WEBP")
    png_image = make_b64image("PNG")

    base64_data = base64.b64encode(b"some random data")
    datauri_base64_data = f"data:text/plain;base64,{base64_data}"

    tests = [
        # input                                    expected
        [jpeg_image, jpeg_image],
        [webp_image, None],
        [png_image, png_image],
        [f"data:image/jpeg;base64,{jpeg_image}", jpeg_image],
        [f"data:image/webp;base64,{webp_image}", None],
        [f"data:image/png;base64,{png_image}", png_image],
        [datauri_base64_data, datauri_base64_data],
    ]

    client = HTTPHandler()
    for test in tests:
        sent_images.clear()
        try:
            with patch.object(client, "post", side_effect=mock_post):
                completion(
                    model="ollama/llava",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Whats in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": test[0]},
                                },
                            ],
                        }
                    ],
                    client=client,
                )
                (image_data,) = sent_images[0]
                if not test[1]:
                    # the conversion process may not always generate the same image,
                    # so just check for a JPEG image when a conversion was done.
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    assert image.format == "JPEG"
                else:
                    assert image_data == test[1]
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


def test_completion_hf_model_no_provider():
    with pytest.raises(litellm.BadRequestError, match="LLM Provider NOT provided"):
        completion(
            model="WizardLM/WizardLM-70B-V1.0",
            messages=messages,
            max_tokens=5,
        )


def gemini_mock_post(*args: object, **kwargs: object) -> MagicMock:
    mock_response: Final = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/json"}
    mock_response.json = MagicMock(
        return_value={
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "get_current_weather",
                                    "args": {"location": "Boston, MA"},
                                }
                            }
                        ],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                    "safetyRatings": [
                        {
                            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HARASSMENT",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_HATE_SPEECH",
                            "probability": "NEGLIGIBLE",
                        },
                        {
                            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                            "probability": "NEGLIGIBLE",
                        },
                    ],
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 86,
                "candidatesTokenCount": 19,
                "totalTokenCount": 105,
            },
        }
    )
    return mock_response


@pytest.mark.asyncio
@pytest.mark.usefixtures("fake_provider_credentials")
async def test_completion_functions_param():
    litellm.set_verbose = True
    function1 = [
        {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        }
    ]
    try:
        from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

        messages = [{"role": "user", "content": "What is the weather like in Boston?"}]

        client = AsyncHTTPHandler(concurrent_limit=1)

        with patch.object(client, "post", side_effect=gemini_mock_post) as mock_client:
            response: litellm.ModelResponse = await litellm.acompletion(
                model="gemini/gemini-1.5-pro",
                messages=messages,
                functions=function1,
                client=client,
            )
            print(response)
            # Add any assertions here to check the response
            mock_client.assert_called()
            print(f"mock_client.call_args.kwargs: {mock_client.call_args.kwargs}")
            assert "tools" in mock_client.call_args.kwargs["json"]
            assert (
                "litellm_param_is_function_call"
                not in mock_client.call_args.kwargs["json"]
            )
            assert response.choices[0].message.function_call is not None
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_deepseek_custom_prompt_dict():
    model = "llama/arn:aws:bedrock:us-east-1:1234:imported-model/45d34re"
    litellm.register_prompt_template(
        model=model,
        tokenizer_config={
            "add_bos_token": True,
            "add_eos_token": False,
            "bos_token": {
                "__type": "AddedToken",
                "content": "<｜begin▁of▁sentence｜>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "clean_up_tokenization_spaces": False,
            "eos_token": {
                "__type": "AddedToken",
                "content": "<｜end▁of▁sentence｜>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "legacy": True,
            "model_max_length": 16384,
            "pad_token": {
                "__type": "AddedToken",
                "content": "<｜end▁of▁sentence｜>",
                "lstrip": False,
                "normalized": True,
                "rstrip": False,
                "single_word": False,
            },
            "sp_model_kwargs": {},
            "unk_token": None,
            "tokenizer_class": "LlamaTokenizerFast",
            "chat_template": "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% set ns = namespace(is_first=false, is_tool=false, is_output_first=true, system_prompt='') %}{%- for message in messages %}{%- if message['role'] == 'system' %}{% set ns.system_prompt = message['content'] %}{%- endif %}{%- endfor %}{{bos_token}}{{ns.system_prompt}}{%- for message in messages %}{%- if message['role'] == 'user' %}{%- set ns.is_tool = false -%}{{'<｜User｜>' + message['content']}}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is none %}{%- set ns.is_tool = false -%}{%- for tool in message['tool_calls']%}{%- if not ns.is_first %}{{'<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{%- set ns.is_first = true -%}{%- else %}{{'\\n' + '<｜tool▁call▁begin｜>' + tool['type'] + '<｜tool▁sep｜>' + tool['function']['name'] + '\\n' + '```json' + '\\n' + tool['function']['arguments'] + '\\n' + '```' + '<｜tool▁call▁end｜>'}}{{'<｜tool▁calls▁end｜><｜end▁of▁sentence｜>'}}{%- endif %}{%- endfor %}{%- endif %}{%- if message['role'] == 'assistant' and message['content'] is not none %}{%- if ns.is_tool %}{{'<｜tool▁outputs▁end｜>' + message['content'] + '<｜end▁of▁sentence｜>'}}{%- set ns.is_tool = false -%}{%- else %}{% set content = message['content'] %}{% if '</think>' in content %}{% set content = content.split('</think>')[-1] %}{% endif %}{{'<｜Assistant｜>' + content + '<｜end▁of▁sentence｜>'}}{%- endif %}{%- endif %}{%- if message['role'] == 'tool' %}{%- set ns.is_tool = true -%}{%- if ns.is_output_first %}{{'<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- set ns.is_output_first = false %}{%- else %}{{'\\n<｜tool▁output▁begin｜>' + message['content'] + '<｜tool▁output▁end｜>'}}{%- endif %}{%- endif %}{%- endfor -%}{% if ns.is_tool %}{{'<｜tool▁outputs▁end｜>'}}{% endif %}{% if add_generation_prompt and not ns.is_tool %}{{'<｜Assistant｜><think>\\n'}}{% endif %}",
        },
    )
    assert model in litellm.known_tokenizer_config
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()

    messages = [
        {"role": "system", "content": "You are a good assistant"},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
    ]

    with patch.object(client, "post") as mock_post:
        try:
            completion(
                model="bedrock/" + model,
                messages=messages,
                client=client,
            )
        except Exception as e:
            pass

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert (
            json_data["prompt"].rstrip()
            == """<｜begin▁of▁sentence｜>You are a good assistant<｜User｜>What is the weather in Copenhagen?<｜Assistant｜><think>"""
        )


def test_bedrock_deepseek_known_tokenizer_config(monkeypatch):
    model = (
        "deepseek_r1/arn:aws:bedrock:us-west-2:888602223428:imported-model/bnnr6463ejgf"
    )
    from unittest.mock import Mock

    import httpx

    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    monkeypatch.setenv("AWS_REGION", "us-east-1")

    mock_response = Mock(spec=httpx.Response)
    mock_response.status_code = 200
    mock_response.headers = {
        "x-amzn-bedrock-input-token-count": "20",
        "x-amzn-bedrock-output-token-count": "30",
    }

    # The response format for deepseek_r1
    response_data = {
        "generation": "The weather in Copenhagen is currently sunny with a temperature of 20°C (68°F). The forecast shows clear skies throughout the day with a gentle breeze from the northwest.",
        "stop_reason": "stop",
        "stop_sequence": None,
    }

    mock_response.json.return_value = response_data
    mock_response.text = json.dumps(response_data)

    client = HTTPHandler()

    messages = [
        {"role": "system", "content": "You are a good assistant"},
        {"role": "user", "content": "What is the weather in Copenhagen?"},
    ]

    with patch.object(client, "post", return_value=mock_response) as mock_post:
        completion(
            model="bedrock/" + model,
            messages=messages,
            client=client,
        )

        mock_post.assert_called_once()
        print(mock_post.call_args.kwargs)
        url = mock_post.call_args.kwargs["url"]
        assert "deepseek_r1" not in url
        assert "us-east-1" not in url
        assert "us-west-2" in url
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert (
            json_data["prompt"].rstrip()
            == """<｜begin▁of▁sentence｜>You are a good assistant<｜User｜>What is the weather in Copenhagen?<｜Assistant｜><think>"""
        )


def test_completion_anthropic_hanging():
    litellm.set_verbose = True
    litellm.modify_params = True
    messages = [
        {
            "role": "user",
            "content": "What's the capital of fictional country Ubabababababaaba? Use your tools.",
        },
        {
            "role": "assistant",
            "function_call": {
                "name": "get_capital",
                "arguments": '{"country": "Ubabababababaaba"}',
            },
        },
        {"role": "function", "name": "get_capital", "content": "Kokoko"},
    ]

    converted_messages = anthropic_messages_pt(
        messages, model="claude-3-sonnet-20240229", llm_provider="anthropic"
    )

    assert len(converted_messages) == 3
    for i, msg in enumerate(converted_messages):
        if i < len(converted_messages) - 1:
            assert msg["role"] != converted_messages[i + 1]["role"]


@respx.mock
def test_completion_openai_returns_the_scripted_assistant_message():
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-migration-openai",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "scripted answer"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            },
        )
    )

    response: Final = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-openai-key",
    )

    assert response.choices[0].message.content == "scripted answer"
    assert route.calls.last.request.headers["Authorization"] == "Bearer test-openai-key"


@respx.mock
def test_completion_openai_response_headers(monkeypatch: pytest.MonkeyPatch, openai_api_response):
    def respond(request: httpx.Request) -> httpx.Response:
        request_body: Final = json.loads(request.content)
        headers: Final = {
            "x-ratelimit-remaining-tokens": "99",
            "x-ratelimit-remaining-requests": "9",
        }
        if request_body.get("stream"):
            chunks: Final = (
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": request_body["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": "Hello!"},
                            "finish_reason": None,
                        }
                    ],
                },
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": request_body["model"],
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                },
            )
            body: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
            return httpx.Response(
                200,
                text=body,
                headers={**headers, "content-type": "text/event-stream"},
            )
        return httpx.Response(200, json=openai_api_response, headers=headers)

    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=respond)
    monkeypatch.setattr(litellm, "return_response_headers", True)

    response: Final = litellm.completion(
        model="gpt-6-sol",
        messages=[{"role": "user", "content": "hello"}],
    )

    assert len(route.calls) == 1
    assert response._response_headers["x-ratelimit-remaining-tokens"] == "99"
    assert response._hidden_params["additional_headers"]["x-ratelimit-remaining-requests"] == "9"
    assert response._hidden_params["additional_headers"]["llm_provider-x-ratelimit-remaining-requests"] == "9"
    stream: Final = litellm.completion(
        model="gpt-6-sol",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    chunks: Final = list(stream)
    assert len(route.calls) == 2
    assert stream._response_headers["x-ratelimit-remaining-tokens"] == "99"
    assert stream._hidden_params["additional_headers"]["x-ratelimit-remaining-requests"] == "9"
    assert chunks[0].choices[0].delta.content == "Hello!"

    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            headers={"x-ratelimit-remaining-tokens": "98"},
        )
    )
    embedding: Final = litellm.embedding(model="text-embedding-3-small", input="hello")
    assert embedding._response_headers["x-ratelimit-remaining-tokens"] == "98"


@pytest.mark.asyncio
@respx.mock
async def test_async_completion_openai_response_headers(monkeypatch: pytest.MonkeyPatch, openai_api_response):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    stream_body: Final = (
        "data: "
        + json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-6-sol",
                "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hi"}, "finish_reason": "stop"}],
            }
        )
        + "\n\ndata: [DONE]\n\n"
    )
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=(
            httpx.Response(200, json=openai_api_response, headers={"x-ratelimit-remaining-tokens": "99"}),
            httpx.Response(
                200,
                text=stream_body,
                headers={"x-ratelimit-remaining-tokens": "97", "content-type": "text/event-stream"},
            ),
        )
    )
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            },
            headers={"x-ratelimit-remaining-tokens": "98"},
        )
    )
    monkeypatch.setattr(litellm, "return_response_headers", True)

    response: Final = await litellm.acompletion(
        model="gpt-6-sol",
        messages=[{"role": "user", "content": "hello"}],
    )
    stream: Final = await litellm.acompletion(
        model="gpt-6-sol",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
    )
    stream_text: Final = "".join([chunk.choices[0].delta.content or "" async for chunk in stream])
    embedding: Final = await litellm.aembedding(model="text-embedding-3-small", input="hello")

    assert len(route.calls) == 2
    assert response._response_headers["x-ratelimit-remaining-tokens"] == "99"
    assert stream._response_headers["x-ratelimit-remaining-tokens"] == "97"
    assert stream_text == "Hi"
    assert embedding._response_headers["x-ratelimit-remaining-tokens"] == "98"


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
@respx.mock
async def test_azure_rate_limit_error_carries_retry_after_header(async_mode: bool, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    respx.post("https://example.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings").mock(
        return_value=httpx.Response(
            429,
            json={"error": {"message": "Rate Limit Error!", "code": "429"}},
            headers={"retry-after": "30"},
        )
    )
    request: Final = {
        "model": "azure/text-embedding-3-small",
        "input": "hello",
        "api_base": "https://example.openai.azure.com",
        "api_version": "2024-02-01",
        "api_key": "azure-test-key",
        "max_retries": 0,
    }

    with pytest.raises(litellm.RateLimitError) as error:
        await (litellm.aembedding(**request) if async_mode else asyncio.to_thread(litellm.embedding, **request))

    assert error.value.litellm_response_headers["retry-after"] == "30"


@respx.mock
def test_completion_preserves_empty_openai_message_content():
    response_data: Final = {
        "id": "chatcmpl-empty",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-6-sol",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 0, "total_tokens": 1},
    }
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response: Final = litellm.completion(
        model="gpt-6-sol",
        messages=[{"role": "user", "content": ""}],
    )

    assert len(route.calls) == 1
    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": ""}],
    }
    assert response.choices[0].message.content == ""


@respx.mock
def test_openai_completion_parses_scripted_chat_response(openai_api_response):
    response_data: Final = {**openai_api_response, "model": "gpt-6-sol"}
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=response_data)
    )
    messages: Final = [{"role": "user", "content": "hello"}]

    response: Final = litellm.completion(model="gpt-6-sol", messages=messages)

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {"model": "gpt-6-sol", "messages": messages}
    assert response.choices[0].message.content == response_data["choices"][0]["message"]["content"]
    assert response.usage.total_tokens == response_data["usage"]["total_tokens"]


@respx.mock
def test_completion_logprobs_parses_top_logprobs():
    response_data: Final = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-6-sol",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "hello"},
                "logprobs": {
                    "content": [
                        {
                            "token": "hello",
                            "logprob": -0.1,
                            "bytes": [104, 101, 108, 108, 111],
                            "top_logprobs": [
                                {"token": "hello", "logprob": -0.1, "bytes": [104, 101, 108, 108, 111]},
                                {"token": "hi", "logprob": -0.2, "bytes": [104, 105]},
                                {"token": "hey", "logprob": -0.3, "bytes": [104, 101, 121]},
                            ],
                        }
                    ]
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response: Final = litellm.completion(
        model="gpt-6-sol",
        messages=[{"role": "user", "content": "hello"}],
        logprobs=True,
        top_logprobs=3,
        reasoning_effort="none",
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "hello"}],
        "logprobs": True,
        "top_logprobs": 3,
        "reasoning_effort": "none",
    }
    assert [
        item.token for item in response.choices[0].logprobs.content[0].top_logprobs
    ] == ["hello", "hi", "hey"]


@respx.mock
def test_completion_logprobs_stream_parses_top_logprobs():
    chunks: Final = (
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-6-sol",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hello"},
                    "finish_reason": None,
                    "logprobs": {
                        "content": [
                            {
                                "token": "hello",
                                "logprob": -0.1,
                                "bytes": [104, 101, 108, 108, 111],
                                "top_logprobs": [
                                    {"token": "hello", "logprob": -0.1, "bytes": [104, 101, 108, 108, 111]},
                                    {"token": "hi", "logprob": -0.2, "bytes": [104, 105]},
                                    {"token": "hey", "logprob": -0.3, "bytes": [104, 101, 121]},
                                ],
                            }
                        ]
                    },
                }
            ],
        },
        {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-6-sol",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop", "logprobs": {"content": []}}],
        },
    )
    stream_body: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, text=stream_body, headers={"content-type": "text/event-stream"})
    )

    streamed_response: Final = litellm.completion(
        model="gpt-6-sol",
        messages=[{"role": "user", "content": "hello"}],
        logprobs=True,
        top_logprobs=3,
        reasoning_effort="none",
        stream=True,
    )
    response_chunks: Final = list(streamed_response)

    assert len(route.calls) == 1
    stream_request_body: Final = json.loads(route.calls[0].request.content)
    assert stream_request_body == {
        "model": "gpt-6-sol",
        "messages": [{"role": "user", "content": "hello"}],
        "logprobs": True,
        "top_logprobs": 3,
        "reasoning_effort": "none",
        "stream_options": {"include_usage": True},
        "stream": True,
    }
    assert [
        item.token for item in response_chunks[0].choices[0].logprobs.content[0].top_logprobs
    ] == ["hello", "hi", "hey"]


@respx.mock
def test_completion_fireworks_ai_parses_response():
    response_data: Final = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "accounts/fireworks/models/deepseek-v4-pro-0813",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello there!"},
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
    }
    route: Final = respx.post("https://api.fireworks.ai/inference/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=response_data)
    )
    messages: Final = [{"role": "user", "content": "hello"}]

    response: Final = litellm.completion(
        model="fireworks_ai/accounts/fireworks/models/deepseek-v4-pro-0813",
        messages=messages,
        api_key="fireworks-test-key",
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "accounts/fireworks/models/deepseek-v4-pro-0813",
        "messages": messages,
    }
    assert response.choices[0].message.content == "Hello there!"
    assert response.usage.total_tokens == 12


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
@respx.mock
async def test_bedrock_converse_parses_scripted_response(
    async_mode: bool, monkeypatch: pytest.MonkeyPatch, fake_provider_credentials: None
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    route: Final = respx.post(
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/us.anthropic.claude-sonnet-5-5/converse"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "Hello from Bedrock."}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 8, "outputTokens": 5, "totalTokens": 13},
                "metrics": {"latencyMs": 12},
            },
        )
    )
    request: Final = {
        "model": "bedrock/converse/us.anthropic.claude-sonnet-5-5",
        "messages": [{"role": "user", "content": "hello"}],
        "api_key": "test-bedrock-token",
        "aws_region_name": "us-west-2",
    }
    response: Final = (
        await litellm.acompletion(**request) if async_mode else litellm.completion(**request)
    )

    assert route.calls[0].request.headers["authorization"] == "Bearer test-bedrock-token"
    assert json.loads(route.calls[0].request.content) == {
        "messages": [{"role": "user", "content": [{"text": "hello"}]}],
        "inferenceConfig": {},
    }
    assert response.choices[0].message.content == "Hello from Bedrock."
    assert response.usage.prompt_tokens == 8
    assert response.usage.completion_tokens == 5


@respx.mock
def test_bedrock_converse_parses_scripted_tool_call_arguments(
    fake_provider_credentials: None,
):
    route: Final = respx.post(
        "https://bedrock-runtime.us-west-2.amazonaws.com/model/us.anthropic.claude-haiku-4-5-20251001-v1:0/converse"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "toolUse": {
                                    "toolUseId": "call_trade",
                                    "name": "trade",
                                    "input": {
                                        "orders": [
                                            {
                                                "action": "buy",
                                                "asset": "BTC",
                                                "amount": 0.1,
                                            }
                                        ]
                                    },
                                }
                            }
                        ],
                    }
                },
                "stopReason": "tool_use",
                "usage": {"inputTokens": 8, "outputTokens": 5, "totalTokens": 13},
            },
        )
    )

    response: Final = litellm.completion(
        model="bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0",
        messages=[{"role": "user", "content": "Buy 0.1 BTC"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "trade",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "orders": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "action": {"type": "string"},
                                        "asset": {"type": "string"},
                                        "amount": {"type": "number"},
                                    },
                                    "required": ["action", "asset", "amount"],
                                },
                            }
                        },
                        "required": ["orders"],
                    },
                },
            }
        ],
        tool_choice={"type": "function", "function": {"name": "trade"}},
        api_key="test-bedrock-token",
        aws_region_name="us-west-2",
    )

    tool_call: Final = response.choices[0].message.tool_calls[0]
    assert route.call_count == 1
    assert tool_call.function.name == "trade"
    assert json.loads(tool_call.function.arguments) == {
        "orders": [{"action": "buy", "asset": "BTC", "amount": 0.1}]
    }


@respx.mock
def test_completion_azure_ad_token_is_forwarded(openai_api_response):
    route: Final = respx.post(
        "https://example.openai.azure.com/openai/deployments/gpt-6.1-sol/chat/completions"
    ).mock(return_value=httpx.Response(200, json=openai_api_response))

    litellm.completion(
        model="azure/gpt-6.1-sol",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://example.openai.azure.com",
        api_version="2023-07-01-preview",
        azure_ad_token="my-special-token",
    )

    assert route.calls[0].request.headers["Authorization"] == "Bearer my-special-token"


@respx.mock
def test_completion_azure_extra_headers_are_forwarded(openai_api_response):
    route: Final = respx.post(
        "https://example.openai.azure.com/openai/deployments/gpt-6.1-sol/chat/completions"
    ).mock(return_value=httpx.Response(200, json=openai_api_response))

    litellm.completion(
        model="azure/gpt-6.1-sol",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://example.openai.azure.com",
        api_version="2023-07-01-preview",
        api_key="azure-test-key",
        extra_headers={
            "Authorization": "my-bad-key",
            "Ocp-Apim-Subscription-Key": "test-subscription-key",
        },
    )

    request: Final = route.calls[0].request
    assert request.headers["Authorization"] == "my-bad-key"
    assert request.headers["Ocp-Apim-Subscription-Key"] == "test-subscription-key"


@respx.mock
def test_completion_azure_api_key_argument_is_forwarded(openai_api_response):
    route: Final = respx.post(
        "https://example.openai.azure.com/openai/deployments/gpt-6.1-sol/chat/completions"
    ).mock(return_value=httpx.Response(200, json=openai_api_response))

    response: Final = litellm.completion(
        model="azure/gpt-6.1-sol",
        messages=[{"role": "user", "content": "hello"}],
        api_base="https://example.openai.azure.com",
        api_version="2023-07-01-preview",
        api_key="azure-test-key",
    )

    assert route.calls[0].request.headers["api-key"] == "azure-test-key"
    assert response._hidden_params["custom_llm_provider"] == "azure"


@respx.mock
def test_completion_forwards_openai_optional_parameters(openai_api_response):
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=openai_api_response)
    )
    messages: Final = [{"role": "user", "content": "return json"}]

    litellm.completion(
        model="gpt-6-sol",
        messages=messages,
        temperature=0.25,
        top_p=0.5,
        seed=12,
        max_tokens=10,
        user="test-user",
        response_format={"type": "json_object"},
        reasoning_effort="none",
        logit_bias=None,
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "gpt-6-sol",
        "messages": messages,
        "temperature": 0.25,
        "top_p": 0.5,
        "seed": 12,
        "max_completion_tokens": 10,
        "user": "test-user",
        "response_format": {"type": "json_object"},
        "reasoning_effort": "none",
    }


@respx.mock
def test_completion_drops_unsupported_o3_mini_temperature(openai_api_response):
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=openai_api_response)
    )

    litellm.completion(
        model="o3-mini",
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        drop_params=True,
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "o3-mini",
        "messages": [{"role": "user", "content": "hello"}],
    }


@pytest.mark.asyncio
@respx.mock
async def test_anthropic_tool_result_after_tool_call_is_translated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    litellm.in_memory_llm_clients_cache.flush_cache()
    response_data: Final = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_test",
                "name": "submitFruit",
                "input": {"name": "Apple"},
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {"input_tokens": 16, "output_tokens": 8},
    }
    route: Final = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=response_data)
    )
    messages: Final = [
        {"role": "system", "content": "Use the submitFruit function for a fruit."},
        {"role": "user", "content": "I like apples"},
        {
            "role": "assistant",
            "content": "<thinking>Use the submitFruit function.</thinking>",
            "tool_calls": [
                {
                    "id": "toolu_test",
                    "type": "function",
                    "function": {"name": "submitFruit", "arguments": '{"name": "Apple"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "toolu_test", "content": '{"success":true}'},
    ]
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "submitFruit",
                "description": "Submits a fruit",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }
    ]

    response: Final = await litellm.acompletion(
        model="anthropic/claude-sonnet-5-5",
        messages=messages,
        tools=tools,
        max_tokens=128,
        api_key="anthropic-test-key",
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body["messages"][1]["content"] == [
        {"type": "text", "text": "<thinking>Use the submitFruit function.</thinking>"},
        {
            "type": "tool_use",
            "id": "toolu_test",
            "name": "submitFruit",
            "input": {"name": "Apple"},
        },
    ]
    assert request_body["messages"][2]["content"][0]["type"] == "tool_result"
    assert request_body["messages"][2]["content"][0]["tool_use_id"] == "toolu_test"
    assert response.choices[0].message.tool_calls[0].function.name == "submitFruit"
    assert json.loads(response.choices[0].message.tool_calls[0].function.arguments) == {"name": "Apple"}


@respx.mock
def test_gemini_completion_translates_messages_and_parses_candidate():
    response_data: Final = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Hello there!"}], "role": "model"},
                "finishReason": "STOP",
                "index": 0,
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 2,
            "candidatesTokenCount": 2,
            "totalTokenCount": 4,
        },
        "modelVersion": "gemini-3.8-flash",
    }
    route: Final = respx.post(
        "https://generativelanguage.googleapis.com/v1alpha/models/gemini-3.8-flash:generateContent"
    ).mock(return_value=httpx.Response(200, json=response_data))

    safety_settings: Final = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
    response: Final = litellm.completion(
        model="gemini/gemini-3.8-flash",
        messages=[
            {"role": "system", "content": "Be a good bot!"},
            {"role": "user", "content": "Hey, how's it going?"},
        ],
        safety_settings=safety_settings,
        api_key="test-gemini-key",
    )

    request: Final = route.calls[0].request
    request_body: Final = json.loads(request.content)
    assert request.headers["x-goog-api-key"] == "test-gemini-key"
    assert request_body["system_instruction"] == {"parts": [{"text": "Be a good bot!"}]}
    assert request_body["contents"] == [{"parts": [{"text": "Hey, how's it going?"}], "role": "user"}]
    assert request_body["safetySettings"] == safety_settings
    assert response.choices[0].message.content == "Hello there!"
    assert response.usage.total_tokens == 4


@respx.mock
def test_qwen_text_completion_parses_text_and_logprobs():
    response_data: Final = {
        "id": "cmpl-test",
        "object": "text_completion",
        "created": 1,
        "model": "gpt-6-sol",
        "choices": [
            {
                "text": "hello",
                "index": 0,
                "logprobs": {
                    "tokens": ["hello"],
                    "token_logprobs": [-0.1],
                    "top_logprobs": [{"hello": -0.1}],
                    "text_offset": [0],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    route: Final = respx.post("https://api.openai.com/v1/completions").mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response: Final = litellm.completion(
        model="text-completion-openai/gpt-6-sol",
        messages=[{"role": "user", "content": "hello"}],
        logprobs=1,
    )

    assert len(route.calls) == 1
    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {"model": "gpt-6-sol", "prompt": "hello", "logprobs": 1}
    assert response.choices[0].message.content == "hello"
    assert response.choices[0].logprobs.token_logprobs == [-0.1]


@respx.mock
def test_replicate_completion_applies_registered_prompt_template(monkeypatch: pytest.MonkeyPatch):
    model: Final = "replicate/meta/llama-3-70b-instruct"
    prompt: Final = "You are a good assistant[INST] What is 2 + 2? [/INST]Now answer as best you can:"
    create_prediction: Final = respx.post(
        "https://api.replicate.com/v1/models/meta/llama-3-70b-instruct/predictions"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "prediction-test",
                "urls": {
                    "get": "https://api.replicate.com/v1/predictions/prediction-test",
                    "cancel": "https://api.replicate.com/v1/predictions/prediction-test/cancel",
                },
            },
        )
    )
    respx.get("https://api.replicate.com/v1/predictions/prediction-test").mock(
        return_value=httpx.Response(200, json={"status": "succeeded", "output": ["4"]})
    )
    monkeypatch.setattr(litellm, "custom_prompt_dict", dict(litellm.custom_prompt_dict))
    litellm.register_prompt_template(
        model=model,
        initial_prompt_value="You are a good assistant",
        final_prompt_value="Now answer as best you can:",
        roles={
            "user": {"pre_message": "[INST] ", "post_message": " [/INST]"},
        },
    )

    response: Final = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": "What is 2 + 2?"}],
        api_key="replicate-test-key",
    )

    request_body: Final = json.loads(create_prediction.calls[0].request.content)
    assert request_body["input"] == {"prompt": prompt}
    assert response.choices[0].message.content == "4"


@respx.mock
def test_petals_completion_sends_model_and_messages_to_custom_base():
    route: Final = respx.post("https://api.petals.dev/").mock(
        return_value=httpx.Response(200, json={"outputs": "Hello!"})
    )
    messages: Final = [{"role": "user", "content": "hello"}]

    response: Final = litellm.completion(
        model="petals-team/StableBeluga2",
        messages=messages,
        api_base="https://api.petals.dev",
        max_tokens=7,
    )

    request: Final = route.calls[0].request
    assert str(request.url) == "https://api.petals.dev/"
    assert request.content == b"model=petals-team%2FStableBeluga2&inputs=hello&max_new_tokens=7"
    assert response.choices[0].message.content == "Hello!"


@respx.mock
def test_mistral_tool_use_round_trip():
    first_response: Final = {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 1,
        "model": "mistral-medium-latest",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{"location":"Boston","unit":"fahrenheit"}',
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    }
    final_response: Final = {
        "id": "chatcmpl-final",
        "object": "chat.completion",
        "created": 1,
        "model": "mistral-large-latest",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "A sunny day."},
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 4, "total_tokens": 34},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        request_body: Final = json.loads(request.content)
        return httpx.Response(
            200,
            json=first_response if request_body["model"] == "mistral-medium-latest" else final_response,
        )

    route: Final = respx.post("https://api.mistral.ai/v1/chat/completions").mock(side_effect=respond)
    messages: Final = [{"role": "user", "content": "What's the weather in Boston?"}]
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"},
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    tool_response: Final = litellm.completion(
        model="mistral/mistral-medium-latest",
        messages=messages,
        tools=tools,
        tool_choice="auto",
        api_key="test-mistral-key",
    )
    tool_call: Final = tool_response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "get_current_weather"
    assert tool_call.function.arguments == '{"location":"Boston","unit":"fahrenheit"}'
    followup_messages: Final = [
        *messages,
        tool_response.choices[0].message.model_dump(exclude_none=True),
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": '{"temperature":"72","unit":"fahrenheit"}',
        },
    ]
    final: Final = litellm.completion(
        model="mistral/mistral-large-latest",
        messages=followup_messages,
        tools=tools,
        tool_choice="auto",
        api_key="test-mistral-key",
    )

    assert len(route.calls) == 2
    second_request: Final = json.loads(route.calls[1].request.content)
    assert second_request["messages"][-1]["role"] == "tool"
    assert second_request["messages"][-1]["tool_call_id"] == "call_weather"
    assert final.choices[0].message.content == "A sunny day."


@respx.mock
def test_mistral_text_content_array_is_sent_as_text():
    response_data: Final = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "mistral-large-latest",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello."},
            }
        ],
        "usage": {"prompt_tokens": 8, "completion_tokens": 2, "total_tokens": 10},
    }
    route: Final = respx.post("https://api.mistral.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=response_data)
    )

    response: Final = litellm.completion(
        model="mistral/mistral-large-latest",
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hey, how's it going?"}],
            }
        ],
        api_key="test-mistral-key",
        input_cost_per_token=0.0000008,
        output_cost_per_token=0.0000032,
    )

    request_body: Final = json.loads(route.calls[0].request.content)
    assert request_body == {
        "model": "mistral-large-latest",
        "messages": [{"role": "user", "content": "Hey, how's it going?"}],
    }
    assert response._hidden_params["response_cost"] == pytest.approx(8 * 0.0000008 + 2 * 0.0000032)


def test_mock_request_returns_requested_response():
    response: Final = litellm.mock_completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        mock_response="mocked answer",
    )

    assert response.choices[0].message.content == "mocked answer"


def test_mock_request_with_mock_timeout_raises_litellm_timeout():
    with pytest.raises(litellm.Timeout) as error:
        litellm.completion(
            model="gpt-5.6",
            messages=[{"role": "user", "content": "hello"}],
            timeout=0.01,
            mock_timeout=True,
            num_retries=0,
        )

    assert error.value.model == "gpt-5.6"


def test_router_mock_request_with_mock_timeout_raises_litellm_timeout():
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5.6",
                "litellm_params": {
                    "model": "gpt-5.6",
                    "api_key": "test-key",
                },
            }
        ],
        num_retries=0,
    )

    with pytest.raises(litellm.Timeout) as error:
        router.completion(
            model="gpt-5.6",
            messages=[{"role": "user", "content": "hello"}],
            timeout=0.01,
            mock_timeout=True,
        )

    assert error.value.model == "gpt-5.6"


@respx.mock
def test_router_mock_request_fallback_uses_fallback_model(openai_api_response):
    route: Final = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={**openai_api_response, "model": "gpt-5.5"})
    )
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-5.6",
                "litellm_params": {"model": "gpt-5.6", "api_key": "test-key"},
            },
            {
                "model_name": "gpt-5.5",
                "litellm_params": {"model": "gpt-5.5", "api_key": "test-key"},
            },
        ],
        fallbacks=[{"gpt-5.6": ["gpt-5.5"]}],
        num_retries=0,
    )

    response: Final = router.completion(
        model="gpt-5.6",
        messages=[{"role": "user", "content": "hello"}],
        timeout=0.01,
        mock_timeout=True,
    )

    assert [json.loads(call.request.content)["model"] for call in route.calls] == ["gpt-5.5"]
    assert response.model == "gpt-5.5"


@pytest.mark.parametrize("drop_params", [True, False])
@respx.mock
def test_completion_deep_infra(drop_params):
    litellm.set_verbose = False
    model_name: Final = "deepinfra/meta-llama/Llama-2-70b-chat-hf"
    model: Final = "meta-llama/Llama-2-70b-chat-hf"
    endpoint: Final = "https://api.deepinfra.com/v1/openai/chat/completions"
    tools: Final = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages: Final = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ]
    response_body: Final = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "It's sunny."},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }
    route: Final = respx.post(endpoint).mock(
        return_value=httpx.Response(200, json=response_body)
    )

    if not drop_params:
        with pytest.raises(litellm.exceptions.UnsupportedParamsError):
            completion(
                model=model_name,
                messages=messages,
                temperature=0,
                max_tokens=10,
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "get_current_weather"}},
                drop_params=False,
                api_key="fake-api-key",
            )
        assert route.calls == []
        return

    response: Final = completion(
        model=model_name,
        messages=messages,
        temperature=0,
        max_tokens=10,
        tools=tools,
        tool_choice={
            "type": "function",
            "function": {"name": "get_current_weather"},
        },
        drop_params=drop_params,
        api_key="fake-api-key",
    )

    assert route.call_count == 1
    request_body: Final = json.loads(route.calls.last.request.read())
    assert request_body == {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 10,
        "tools": tools,
    }
    assert response.choices[0].message.content == "It's sunny."
    assert response.choices[0].finish_reason == "stop"


@respx.mock
def test_completion_deep_infra_mistral():
    model_name: Final = "deepinfra/mistralai/Mistral-7B-Instruct-v0.1"
    model: Final = "mistralai/Mistral-7B-Instruct-v0.1"
    messages: Final = [{"role": "user", "content": "Say hello."}]
    endpoint: Final = "https://api.deepinfra.com/v1/openai/chat/completions"
    response_body: Final = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": model,
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "Hello!"},
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }
    route: Final = respx.post(endpoint).mock(
        return_value=httpx.Response(200, json=response_body)
    )

    response: Final = completion(
        model=model_name,
        messages=messages,
        temperature=0.01,
        max_tokens=10,
        api_key="fake-api-key",
    )

    assert route.call_count == 1
    request_body: Final = json.loads(route.calls.last.request.read())
    assert request_body == {
        "model": model,
        "messages": messages,
        "temperature": 0.01,
        "max_tokens": 10,
    }
    assert response.choices[0].message.content == "Hello!"
    assert response.choices[0].finish_reason == "stop"


@pytest.mark.parametrize(
    "provider, model, project, region_name, token",
    [
        ("azure", "chatgpt-v-3", None, None, "test-token"),
        ("vertex_ai", "anthropic-claude-3", "adroit-crow-1", "us-east1", None),
        ("watsonx", "ibm/granite", "96946574", "dallas", "1234"),
        ("bedrock", "anthropic.claude-3", None, "us-east-1", None),
    ],
)
def test_unified_auth_params(provider, model, project, region_name, token):
    """
    Check if params = ["project", "region_name", "token"]
    are correctly translated for = ["azure", "vertex_ai", "watsonx", "aws"]

    tests get_optional_params
    """
    data = {
        "project": project,
        "region_name": region_name,
        "token": token,
        "custom_llm_provider": provider,
        "model": model,
    }

    translated_optional_params = litellm.utils.get_optional_params(**data)

    if provider == "azure":
        special_auth_params = (
            litellm.AzureOpenAIConfig().get_mapped_special_auth_params()
        )
    elif provider == "bedrock":
        special_auth_params = (
            litellm.AmazonBedrockGlobalConfig().get_mapped_special_auth_params()
        )
    elif provider == "vertex_ai":
        special_auth_params = litellm.VertexAIConfig().get_mapped_special_auth_params()
    elif provider == "watsonx":
        special_auth_params = (
            litellm.IBMWatsonXAIConfig().get_mapped_special_auth_params()
        )

    for param, value in special_auth_params.items():
        assert param in data
        assert value in translated_optional_params


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("sync_mode", [False, True])
@pytest.mark.asyncio
async def test_dynamic_azure_params(stream, sync_mode):
    """
    If dynamic params are given, which are different from the initialized client, use a new client
    """
    from openai import AsyncAzureOpenAI, AzureOpenAI

    if sync_mode:
        client = AzureOpenAI(
            api_key="my-test-key",
            base_url="my-test-base",
            api_version="my-test-version",
        )
        mock_client = MagicMock(return_value="Hello world!")
    else:
        client = AsyncAzureOpenAI(
            api_key="my-test-key",
            base_url="my-test-base",
            api_version="my-test-version",
        )
        mock_client = AsyncMock(return_value="Hello world!")

    ## CHECK IF CLIENT IS USED (NO PARAM CHANGE)
    with patch.object(
        client.chat.completions.with_raw_response, "create", new=mock_client
    ) as mock_client:
        try:
            # client.chat.completions.with_raw_response.create = mock_client
            if sync_mode:
                _ = completion(
                    model="azure/chatgpt-v2",
                    messages=[{"role": "user", "content": "Hello world"}],
                    client=client,
                    stream=stream,
                )
            else:
                _ = await litellm.acompletion(
                    model="azure/chatgpt-v2",
                    messages=[{"role": "user", "content": "Hello world"}],
                    client=client,
                    stream=stream,
                )
        except Exception:
            pass

        mock_client.assert_called()

    ## recreate mock client
    if sync_mode:
        new_mock_client = MagicMock(return_value="Hello world!")
    else:
        new_mock_client = AsyncMock(return_value="Hello world!")

    ## CHECK IF NEW CLIENT IS USED (PARAM CHANGE)
    with patch.object(
        client.chat.completions.with_raw_response, "create", new=new_mock_client
    ) as new_mock_client:
        try:
            if sync_mode:
                _ = completion(
                    model="azure/chatgpt-v2",
                    messages=[{"role": "user", "content": "Hello world"}],
                    client=client,
                    api_version="my-new-version",
                    stream=stream,
                )
            else:
                _ = await litellm.acompletion(
                    model="azure/chatgpt-v2",
                    messages=[{"role": "user", "content": "Hello world"}],
                    client=client,
                    api_version="my-new-version",
                    stream=stream,
                )
        except Exception:
            pass

        try:
            new_mock_client.assert_called()
        except Exception as e:
            raise e


def _openai_hallucinated_tool_call_mock_response(
    *args: object,
    **kwargs: object,
) -> MagicMock:
    new_response: Final = MagicMock()
    new_response.headers = {"hello": "world"}
    response_object: Final = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-3.5-turbo-0125",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"tool_uses":[{"recipient_name":"product_title","parameters":{"content":"Story Scribe"}},{"recipient_name":"one_liner","parameters":{"content":"Transform interview transcripts into actionable user stories"}}]}',
                                "name": "multi_tool_use.parallel",
                            },
                            "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s",
                            "type": "function",
                        }
                    ],
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
    }
    pydantic_response: Final = ChatCompletion.model_validate(response_object)
    setattr(pydantic_response.choices[0].message, "role", None)
    new_response.parse.return_value = pydantic_response
    return new_response


def test_openai_hallucinated_tool_call():
    """
    Patch for this issue: https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653

    Handle openai invalid tool calling response.

    OpenAI assistant will sometimes return an invalid tool calling response, which needs to be parsed

    -           "arguments": "{\"tool_uses\":[{\"recipient_name\":\"product_title\",\"parameters\":{\"content\":\"Story Scribe\"}},{\"recipient_name\":\"one_liner\",\"parameters\":{\"content\":\"Transform interview transcripts into actionable user stories\"}}]}",

    To extract actual tool calls:

    1. Parse arguments JSON object
    2. Iterate over tool_uses array to call functions:
        - get function name from recipient_name value
        - parameters will be JSON object for function arguments
    """
    import openai

    openai_client = openai.OpenAI()
    with patch.object(
        openai_client.chat.completions,
        "create",
        side_effect=_openai_hallucinated_tool_call_mock_response,
    ) as mock_response:
        response = litellm.completion(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hey! how's it going?"}],
            client=openai_client,
        )
        print(f"response: {response}")

        response_dict = response.model_dump()

        tool_calls = response_dict["choices"][0]["message"]["tool_calls"]

        print(f"tool_calls: {tool_calls}")

        for idx, tc in enumerate(tool_calls):
            if idx == 0:
                print(f"tc in test_openai_hallucinated_tool_call: {tc}")
                assert tc == {
                    "function": {
                        "arguments": '{"content": "Story Scribe"}',
                        "name": "product_title",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_0",
                    "type": "function",
                }
            elif idx == 1:
                assert tc == {
                    "function": {
                        "arguments": '{"content": "Transform interview transcripts into actionable user stories"}',
                        "name": "one_liner",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_1",
                    "type": "function",
                }


@pytest.mark.parametrize(
    "function_name, expect_modification",
    [
        ("multi_tool_use.parallel", True),
        ("my-fake-function", False),
    ],
)
def test_openai_hallucinated_tool_call_util(function_name, expect_modification):
    """
    Patch for this issue: https://community.openai.com/t/model-tries-to-call-unknown-function-multi-tool-use-parallel/490653

    Handle openai invalid tool calling response.

    OpenAI assistant will sometimes return an invalid tool calling response, which needs to be parsed

    -           "arguments": "{\"tool_uses\":[{\"recipient_name\":\"product_title\",\"parameters\":{\"content\":\"Story Scribe\"}},{\"recipient_name\":\"one_liner\",\"parameters\":{\"content\":\"Transform interview transcripts into actionable user stories\"}}]}",

    To extract actual tool calls:

    1. Parse arguments JSON object
    2. Iterate over tool_uses array to call functions:
        - get function name from recipient_name value
        - parameters will be JSON object for function arguments
    """
    from litellm.types.utils import ChatCompletionMessageToolCall
    from litellm.utils import _handle_invalid_parallel_tool_calls

    response = _handle_invalid_parallel_tool_calls(
        tool_calls=[
            ChatCompletionMessageToolCall(
                **{
                    "function": {
                        "arguments": '{"tool_uses":[{"recipient_name":"product_title","parameters":{"content":"Story Scribe"}},{"recipient_name":"one_liner","parameters":{"content":"Transform interview transcripts into actionable user stories"}}]}',
                        "name": function_name,
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s",
                    "type": "function",
                }
            )
        ]
    )

    print(f"response: {response}")

    if expect_modification:
        for idx, tc in enumerate(response):
            if idx == 0:
                assert tc.model_dump() == {
                    "function": {
                        "arguments": '{"content": "Story Scribe"}',
                        "name": "product_title",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_0",
                    "type": "function",
                }
            elif idx == 1:
                assert tc.model_dump() == {
                    "function": {
                        "arguments": '{"content": "Transform interview transcripts into actionable user stories"}',
                        "name": "one_liner",
                    },
                    "id": "call_IzGXwVa5OfBd9XcCJOkt2q0s_1",
                    "type": "function",
                }
    else:
        assert len(response) == 1
        assert response[0].function.name == function_name


def test_completion_novita_ai():
    litellm.set_verbose = True
    messages = [
        {"role": "system", "content": "You're a good bot"},
        {
            "role": "user",
            "content": "Hey",
        },
    ]

    from openai import OpenAI

    openai_client = OpenAI(api_key="fake-key")

    with patch.object(
        openai_client.chat.completions.with_raw_response, "create"
    ) as mock_call:
        mock_call.return_value.headers = {}
        mock_call.return_value.parse.return_value = litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "Hello"}}]
        )
        try:
            response = completion(
                model="novita/meta-llama/llama-3.3-70b-instruct",
                messages=messages,
                client=openai_client,
                api_base="https://api.novita.ai/v3/openai",
            )

            mock_call.assert_called_once()
            assert response.choices[0].message.content == "Hello"

            # Verify model is passed correctly
            assert (
                mock_call.call_args.kwargs["model"]
                == "meta-llama/llama-3.3-70b-instruct"
            )
            # Verify messages are passed correctly
            assert mock_call.call_args.kwargs["messages"] == messages

        except Exception as e:
            pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("api_key", ["my-bad-api-key"])
def test_completion_novita_ai_dynamic_params(api_key):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]

        from openai import OpenAI

        openai_client = OpenAI(api_key="fake-key")

        with patch.object(
            openai_client.chat.completions,
            "create",
            side_effect=Exception("Invalid API key"),
        ) as mock_call:
            with pytest.raises(Exception, match="Invalid API key") as exc_info:
                completion(
                    model="novita/meta-llama/llama-3.3-70b-instruct",
                    messages=messages,
                    api_key=api_key,
                    client=openai_client,
                    api_base="https://api.novita.ai/v3/openai",
                )
            e = exc_info.value
            assert "Invalid API key" in str(e)

            mock_call.assert_called_once()
    except Exception as e:
        pytest.fail(f"Unexpected error: {e}")


@pytest.mark.parametrize(
    "enable_preview_features",
    [True, False],
)
def test_completion_openai_metadata(monkeypatch, enable_preview_features):
    from openai import OpenAI

    client = OpenAI()

    litellm.set_verbose = True

    monkeypatch.setattr(litellm, "enable_preview_features", enable_preview_features)
    with patch.object(
        client.chat.completions.with_raw_response, "create", return_value=MagicMock()
    ) as mock_completion:
        try:
            resp = litellm.completion(
                model="openai/gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello world"}],
                metadata={"my-test-key": "my-test-value"},
                client=client,
            )
        except Exception as e:
            print(f"Error: {e}")

        mock_completion.assert_called_once()
        if enable_preview_features:
            assert mock_completion.call_args.kwargs["metadata"] == {
                "my-test-key": "my-test-value"
            }
        else:
            assert "metadata" not in mock_completion.call_args.kwargs


AZURE_TTS_BASE: Final = "https://tts.example.azure.com"
SPEECH_INPUT: Final = "the quick brown fox jumped over the lazy dogs"


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_speech_azure_returns_binary_audio(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, sync_mode: bool
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post(
        url__regex=rf"{AZURE_TTS_BASE}/openai/deployments/tts/audio/speech\?api-version=.+"
    ).mock(return_value=httpx.Response(200, content=b"ID3-fake-mp3"))

    speech_kwargs: Final = {
        "model": "azure/tts",
        "input": SPEECH_INPUT,
        "voice": "alloy",
        "api_base": AZURE_TTS_BASE,
        "api_key": "fake-key",
        "max_retries": 1,
        "timeout": 60,
    }
    response: Final = litellm.speech(**speech_kwargs) if sync_mode else await litellm.aspeech(**speech_kwargs)

    assert route.call_count == 1
    assert route.calls[0].request.headers["api-key"] == "fake-key"
    assert json.loads(route.calls[0].request.content) == {"model": "tts", "input": SPEECH_INPUT, "voice": "alloy"}
    assert isinstance(response, HttpxBinaryResponseContent)
    assert response.content == b"ID3-fake-mp3"


@pytest.mark.parametrize("sync_mode", [True, False])
async def test_speech_openai_returns_binary_audio(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, sync_mode: bool
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/audio/speech").mock(
        return_value=httpx.Response(200, content=b"ID3-fake-mp3")
    )

    speech_kwargs: Final = {
        "model": "openai/tts-1",
        "input": SPEECH_INPUT,
        "voice": "alloy",
        "api_key": "fake-key",
        "max_retries": 1,
        "timeout": 60,
    }
    response: Final = litellm.speech(**speech_kwargs) if sync_mode else await litellm.aspeech(**speech_kwargs)

    assert route.call_count == 1
    assert route.calls[0].request.headers["authorization"] == "Bearer fake-key"
    assert json.loads(route.calls[0].request.content) == {"model": "tts-1", "input": SPEECH_INPUT, "voice": "alloy"}
    assert isinstance(response, HttpxBinaryResponseContent)
    assert response.content == b"ID3-fake-mp3"


class _SignallingCache(Cache):
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        super().__init__()
        self.loop: Final = loop
        self.written: Final = asyncio.Event()

    async def async_add_cache(
        self, result: object, dynamic_cache_object: BaseCache | None = None, **kwargs: object
    ) -> None:
        await super().async_add_cache(result, dynamic_cache_object=dynamic_cache_object, **kwargs)
        self.loop.call_soon_threadsafe(self.written.set)


GETTYSBURG_WAV: Final = ("gettysburg.wav", b"RIFF\x00\x00\x00\x00WAVE-gettysburg", "audio/wav")
EAGLE_WAV: Final = ("eagle.wav", b"RIFF\x00\x00\x00\x00WAVE-eagle", "audio/wav")


async def test_transcription_caching_hit_same_file_miss_different_file(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    cache: Final = _SignallingCache(asyncio.get_running_loop())
    monkeypatch.setattr(litellm, "cache", cache)
    route: Final = respx_mock.post("https://api.openai.com/v1/audio/transcriptions").mock(
        side_effect=[
            httpx.Response(200, json={"text": "gettysburg transcript"}),
            httpx.Response(200, json={"text": "eagle transcript"}),
        ]
    )

    response_1: Final = await litellm.atranscription(model="openai/whisper-1", file=GETTYSBURG_WAV, api_key="fake-key")
    await asyncio.wait_for(cache.written.wait(), 30)

    response_2: Final = await litellm.atranscription(model="openai/whisper-1", file=GETTYSBURG_WAV, api_key="fake-key")
    assert response_2._hidden_params["cache_hit"] is True
    assert response_2.text == response_1.text == "gettysburg transcript"

    response_3: Final = await litellm.atranscription(model="openai/whisper-1", file=EAGLE_WAV, api_key="fake-key")
    assert response_3._hidden_params.get("cache_hit") is not True
    assert response_3.text == "eagle transcript"
    assert route.call_count == 2


async def test_whisper_log_pre_call_fires_once(respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    respx_mock.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello"})
    )

    class _PreCallRecorder(CustomLogger):
        def __init__(self) -> None:
            super().__init__()
            self.models: tuple[str, ...] = ()

        def log_pre_api_call(self, model: str, messages: object, kwargs: Mapping[str, object]) -> None:
            self.models = (*self.models, model)

    recorder: Final = _PreCallRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])

    await litellm.atranscription(model="openai/whisper-1", file=GETTYSBURG_WAV, api_key="fake-key")

    assert recorder.models == ("whisper-1",)


@pytest.mark.parametrize("model", ["gpt-4o-mini-transcribe", "gpt-4o-transcribe", "whisper-1"])
async def test_transcription_model_names_pass_through(
    respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch, model: str
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route: Final = respx_mock.post("https://api.openai.com/v1/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello"})
    )

    response: Final = await litellm.atranscription(
        model=f"openai/{model}",
        file=GETTYSBURG_WAV,
        api_key="fake-key",
        response_format="json",
    )

    assert response._hidden_params["model"] == model
    assert response._hidden_params["custom_llm_provider"] == "openai"
    assert response.text == "hello"
    assert route.call_count == 1
    assert f'name="model"\r\n\r\n{model}\r\n'.encode() in route.calls[0].request.content


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("model", ["sagemaker/test-endpoint", "sagemaker_chat/test-endpoint"])
async def test_sagemaker_missing_dependency_remains_actionable_with_retries(monkeypatch, use_async, model):
    import sys

    monkeypatch.setattr(litellm, "num_retries", None)
    for dependency in ("botocore", "boto3", "tenacity"):
        monkeypatch.setitem(sys.modules, dependency, None)
    if use_async:
        with pytest.raises(ModuleNotFoundError, match="pip install boto3") as caught:
            await litellm.acompletion(model=model, messages=[{"role": "user", "content": "ping"}], num_retries=1)
    else:
        with pytest.raises(ModuleNotFoundError, match="pip install boto3") as caught:
            litellm.completion(model=model, messages=[{"role": "user", "content": "ping"}], num_retries=1)
    assert caught.value.name == "botocore"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_polly_missing_dependency_remains_actionable_with_retries(monkeypatch, use_async):
    import sys

    monkeypatch.setattr(litellm, "num_retries", None)
    for dependency in ("botocore", "boto3", "tenacity"):
        monkeypatch.setitem(sys.modules, dependency, None)
    if use_async:
        with pytest.raises(ModuleNotFoundError, match="pip install boto3") as caught:
            await litellm.aspeech(model="aws_polly/standard", input="ping", voice="Joanna", num_retries=1)
    else:
        with pytest.raises(ModuleNotFoundError, match="pip install boto3") as caught:
            litellm.speech(model="aws_polly/standard", input="ping", voice="Joanna", num_retries=1)
    assert caught.value.name == "botocore"


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_tenacity", [False, True])
@pytest.mark.parametrize("use_async", [False, True])
async def test_mantle_responses_missing_dependency_is_not_retried(monkeypatch, missing_tenacity, use_async):
    import builtins

    original_import = builtins.__import__
    attempts = []

    def import_without_aws(name, *args, **kwargs):
        if name == "botocore":
            attempts.append(name)
            raise ModuleNotFoundError(name="botocore")
        if name == "tenacity" and missing_tenacity:
            attempts.append(name)
            raise ModuleNotFoundError(name="tenacity")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_aws)
    monkeypatch.setattr(litellm, "num_retries", None)
    for name in ("AWS_BEARER_TOKEN_BEDROCK", "BEDROCK_MANTLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    if use_async:
        with pytest.raises(ModuleNotFoundError, match="pip install boto3"):
            await litellm.aresponses(model="bedrock_mantle/openai.gpt-oss-120b", input="ping", num_retries=1)
    else:
        with pytest.raises(ModuleNotFoundError, match="pip install boto3"):
            litellm.responses(model="bedrock_mantle/openai.gpt-oss-120b", input="ping", num_retries=1)
    assert attempts == ["botocore"]


@pytest.mark.asyncio
async def test_async_responses_still_retries_provider_server_errors(monkeypatch):
    monkeypatch.setattr(litellm, "num_retries", None)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    with respx.mock as upstream:
        response = upstream.post("https://openai-test.invalid/v1/responses").mock(side_effect=[
            httpx.Response(500, json={"error": {"message": "temporary provider failure", "type": "server_error"}}),
            httpx.Response(200, json={
                "id": "resp-retry", "object": "response", "created_at": 1, "status": "completed",
                "model": "test-model", "output": [],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
            }),
        ])
        result = await litellm.aresponses(
            model="openai/test-model", input="ping", api_key="test-key",
            api_base="https://openai-test.invalid/v1", num_retries=1, max_retries=0,
        )
        assert result.status == "completed"
        assert response.call_count == 2


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_completion_with_retries(sync_mode):
    """
    If completion_with_retries is called with num_retries=3, and max_retries=0, then litellm.completion should receive num_retries , max_retries=0
    """
    if sync_mode:
        target_function = "completion"
    else:
        target_function = "acompletion"

    with patch.object(litellm, target_function) as mock_completion:
        if sync_mode:
            completion_with_retries(
                model="gpt-3.5-turbo",
                messages=[{"gm": "vibe", "role": "user"}],
                num_retries=3,
                original_function=mock_completion,
            )
        else:
            await acompletion_with_retries(
                model="gpt-3.5-turbo",
                messages=[{"gm": "vibe", "role": "user"}],
                num_retries=3,
                original_function=mock_completion,
            )
        mock_completion.assert_called_once()
        assert mock_completion.call_args.kwargs["num_retries"] == 0
        assert mock_completion.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_responses_with_retries(sync_mode):
    """
    Test that responses() and aresponses() properly handle num_retries parameter.
    If responses_with_retries is called with num_retries=3, and max_retries=0,
    then litellm.responses should receive num_retries=0, max_retries=0
    """
    if sync_mode:
        target_function = "responses"
        retry_function = responses_with_retries
    else:
        target_function = "aresponses"
        retry_function = aresponses_with_retries

    with patch(
        "litellm.responses.main.responses" if sync_mode else "litellm.responses.main.aresponses"
    ) as mock_responses:
        if sync_mode:
            mock_responses.return_value = MagicMock()
            retry_function(
                model="gpt-4o",
                input="Hello, what's the weather?",
                num_retries=3,
                original_function=mock_responses,
            )
        else:
            mock_responses.return_value = AsyncMock()
            await retry_function(
                model="gpt-4o",
                input="Hello, what's the weather?",
                num_retries=3,
                original_function=mock_responses,
            )

        mock_responses.assert_called_once()
        assert mock_responses.call_args.kwargs["num_retries"] == 0
        assert mock_responses.call_args.kwargs["max_retries"] == 0


def test_azure_embedding_exceptions():
    with pytest.raises(Exception, match="Mock error") as exc_info:
        litellm.embedding(
            model="azure/text-embedding-ada-002",
            input="hello",
            mock_response="error",
        )
    assert str(exc_info.value) == "Mock error"


@respx.mock
def test_openai_chat_stream_routes_legacy_function_calls_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "return_response_headers", True)
    api_base: Final = "https://openai-unit.test/v1"
    messages: Final = ({"role": "user", "content": "What is the weather in SF?"},)
    functions: Final = (
        {
            "name": "get_current_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    )
    answer_events: Final = (
        {
            "id": "chatcmpl-answer",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "The answer is "},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-answer",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "42."},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-answer",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-answer",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    )
    function_events: Final = (
        {
            "id": "chatcmpl-function",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "function_call": {
                            "name": "get_current_weather",
                            "arguments": '{"location":',
                        },
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-function",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"function_call": {"arguments": '"San Francisco"}'}},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-function",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "function_call"}
            ],
        },
    )

    def response_body(events: tuple[dict[str, object], ...]) -> str:
        return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"

    scripted_responses: Final = iter(
        (
            httpx.Response(
                200,
                text=response_body(answer_events),
                headers={
                    "content-type": "text/event-stream",
                    "x-ratelimit-remaining-requests": "17",
                },
            ),
            httpx.Response(
                200,
                text=response_body(function_events),
                headers={"content-type": "text/event-stream"},
            ),
        )
    )
    route: Final = respx.post(f"{api_base}/chat/completions").mock(
        side_effect=lambda _: next(scripted_responses)
    )
    stream: Final = litellm.completion(
        model="gpt-4o-mini",
        api_key="sk-test",
        api_base=api_base,
        messages=messages,
        stream=True,
        stream_options={"include_usage": True},
    )
    answer_chunks: Final = tuple(stream)

    assert "".join(
        chunk.choices[0].delta.content
        for chunk in answer_chunks
        if chunk.choices and chunk.choices[0].delta.content
    ) == "The answer is 42."
    assert tuple(
        chunk.choices[0].finish_reason
        for chunk in answer_chunks
        if chunk.choices and chunk.choices[0].finish_reason is not None
    ) == ("stop",)
    assert answer_chunks[-2].choices[0].finish_reason == "stop"
    assert tuple(getattr(chunk, "usage", None) for chunk in answer_chunks[:-1]) == (
        None,
        None,
        None,
    )
    assert answer_chunks[-1].usage is not None
    assert answer_chunks[-1].usage.prompt_tokens == 4
    assert answer_chunks[-1].usage.completion_tokens == 2
    assert answer_chunks[-1].usage.total_tokens == 6
    assert stream._hidden_params["api_base"] == api_base
    assert (
        answer_chunks[0]._hidden_params["additional_headers"][
            "llm_provider-x-ratelimit-remaining-requests"
        ]
        == "17"
    )

    function_stream: Final = litellm.completion(
        model="gpt-4o-mini",
        api_key="sk-test",
        api_base=api_base,
        messages=messages,
        functions=functions,
        function_call={"name": "get_current_weather"},
        stream=True,
    )
    function_chunks: Final = tuple(function_stream)
    assembled: Final = litellm.stream_chunk_builder(
        chunks=function_chunks, messages=messages
    )
    function_call: Final = assembled.choices[0].message.function_call

    assert function_call is not None
    assert function_call.name == "get_current_weather"
    assert json.loads(function_call.arguments) == {"location": "San Francisco"}
    assert tuple(
        chunk.choices[0].finish_reason
        for chunk in function_chunks
        if chunk.choices and chunk.choices[0].finish_reason is not None
    ) == ("function_call",)
    assert route.call_count == 2
    assert json.loads(route.calls[0].request.content)["stream_options"] == {
        "include_usage": True
    }
    assert json.loads(route.calls[1].request.content)["function_call"] == {
        "name": "get_current_weather"
    }
    assert json.loads(route.calls[1].request.content)["functions"] == list(functions)


@respx.mock
async def test_async_openai_stream_options_include_usage_on_the_last_chunk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    events: Final = (
        {
            "id": "chatcmpl-async-usage",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "async reply"},
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-async-usage",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-async-usage",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    )
    body: Final = "".join(
        f"data: {json.dumps(event)}\n\n" for event in events
    ) + "data: [DONE]\n\n"
    route: Final = respx.post(
        "https://api.openai.com/v1/chat/completions"
    ).mock(
        return_value=httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )
    )
    stream: Final = await litellm.acompletion(
        model="gpt-4o-mini",
        api_key="sk-test",
        messages=[{"role": "user", "content": "say hello"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks: Final = tuple([chunk async for chunk in stream])

    assert "".join(
        chunk.choices[0].delta.content
        for chunk in chunks
        if chunk.choices and chunk.choices[0].delta.content
    ) == "async reply"
    assert tuple(getattr(chunk, "usage", None) for chunk in chunks[:-1]) == (
        None,
        None,
    )
    assert chunks[-1].usage.prompt_tokens == 3
    assert chunks[-1].usage.completion_tokens == 2
    assert chunks[-1].usage.total_tokens == 5
    assert json.loads(route.calls[0].request.content)["stream_options"] == {
        "include_usage": True
    }


@respx.mock
async def test_parallel_openai_streams_keep_prompt_responses_separate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)

    def respond(request: httpx.Request) -> httpx.Response:
        prompt: Final = json.loads(request.content)["messages"][0]["content"]
        response_text: Final = {
            "request one": "response one",
            "request two": "response two",
        }[prompt]
        events: Final = (
            {
                "id": "chatcmpl-parallel",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": response_text},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-parallel",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        )
        body: Final = "".join(
            f"data: {json.dumps(event)}\n\n" for event in events
        ) + "data: [DONE]\n\n"
        return httpx.Response(
            200, text=body, headers={"content-type": "text/event-stream"}
        )

    route: Final = respx.post(
        "https://api.openai.com/v1/chat/completions"
    ).mock(side_effect=respond)

    async def collect(prompt: str) -> tuple[str, tuple[str, ...]]:
        stream = await litellm.acompletion(
            model="gpt-4o-mini",
            api_key="sk-test",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        chunks: Final = tuple([chunk async for chunk in stream])
        response_text: Final = "".join(
            chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
        )
        finish_reasons: Final = tuple(
            chunk.choices[0].finish_reason
            for chunk in chunks
            if chunk.choices and chunk.choices[0].finish_reason is not None
        )
        return response_text, finish_reasons

    responses: Final = tuple(
        await asyncio.gather(collect("request one"), collect("request two"))
    )

    assert tuple(response for response, _ in responses) == (
        "response one",
        "response two",
    )
    assert tuple(finish_reasons for _, finish_reasons in responses) == (
        ("stop",),
        ("stop",),
    )
    assert route.call_count == 2


@respx.mock
def test_openai_streaming_tool_call_arguments_are_valid_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    chunks: Final = (
        {
            "id": "chatcmpl-tools",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_weather",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{"location":"Boston",',
                                },
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-tools",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "index": 0,
                                "function": {"arguments": '"unit":"fahrenheit"}'},
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-tools",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
        },
    )
    body: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )
    )
    messages: Final = [{"role": "user", "content": "What is the weather in Boston?"}]
    stream: Final = litellm.completion(
        model="gpt-4o-mini",
        messages=messages,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_current_weather",
                    "description": "Get the weather",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"},
                            "unit": {"type": "string"},
                        },
                        "required": ["location", "unit"],
                    },
                },
            }
        ],
        stream=True,
        api_key="test-openai-key",
    )
    stream_chunks: Final = tuple(stream)
    assembled: Final = litellm.stream_chunk_builder(chunks=list(stream_chunks), messages=messages)
    assert assembled is not None
    assert assembled.choices[0].message.tool_calls[0].function.name == "get_current_weather"
    assert json.loads(assembled.choices[0].message.tool_calls[0].function.arguments) == {
        "location": "Boston",
        "unit": "fahrenheit",
    }


@respx.mock
def test_streaming_openai_usage_includes_test_owned_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    model_name: Final = "stream-cost-test-model"
    monkeypatch.setitem(
        litellm.model_cost,
        model_name,
        {
            "input_cost_per_token": 0.001,
            "output_cost_per_token": 0.002,
            "litellm_provider": "openai",
            "mode": "chat",
        },
    )
    chunks: Final = (
        {
            "id": "chatcmpl-cost",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model_name,
            "choices": [
                {"index": 0, "delta": {"role": "assistant", "content": "priced"}, "finish_reason": None}
            ],
        },
        {
            "id": "chatcmpl-cost",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        },
        {
            "id": "chatcmpl-cost",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model_name,
            "choices": [],
            "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
        },
    )
    body: Final = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            text=body,
            headers={"content-type": "text/event-stream"},
        )
    )
    stream: Final = litellm.completion(
        model=f"openai/{model_name}",
        messages=[{"role": "user", "content": "price the response"}],
        stream=True,
        stream_options={"include_usage": True},
        api_key="test-openai-key",
    )
    stream_chunks: Final = tuple(stream)
    usage_chunks: Final = tuple(
        getattr(chunk, "usage", None)
        for chunk in stream_chunks
        if getattr(chunk, "usage", None) is not None
    )
    assert len(usage_chunks) == 1
    assert usage_chunks[0].cost == pytest.approx(4 * 0.001 + 2 * 0.002)

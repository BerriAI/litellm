import json
import os
import sys

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import urllib.parse
from unittest.mock import MagicMock, patch

import litellm


@pytest.fixture(autouse=True)
def add_api_keys_to_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-1234567890")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-api03-1234567890")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "my-fake-aws-access-key-id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "my-fake-aws-secret-access-key")
    monkeypatch.setenv("AWS_REGION", "us-east-1")


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
        client.chat.completions.with_raw_response, "create", mock_raw_response
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


@pytest.mark.parametrize(
    "model",
    [
        "gemini/gemini-1.5-flash",
        "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        "bedrock/invoke/anthropic.claude-3-5-sonnet-20240620-v1:0",
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

    args = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                            "format": "image/png",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
    }
    with patch.object(client, "post", new=MagicMock()) as mock_client:
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
        assert "png" in json_str
        assert "jpeg" not in json_str


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
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                            "format": "image/png",
                        },
                    },
                    {"type": "text", "text": "Describe this image"},
                ],
            }
        ],
    }
    with patch.object(
        client.chat.completions.with_raw_response, "create"
    ) as mock_client:
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


def test_bedrock_latency_optimized_inference():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    client = HTTPHandler()
    with patch.object(client, "post") as mock_post:
        try:
            response = litellm.completion(
                model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                performanceConfig={"latency": "optimized"},
                client=client,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()
        json_data = json.loads(mock_post.call_args.kwargs["data"])
        assert json_data["performanceConfig"]["latency"] == "optimized"


def test_custom_provider_with_extra_headers():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    with patch.object(
        litellm.llms.custom_httpx.http_handler.HTTPHandler, "post"
    ) as mock_post:
        response = litellm.completion(
            model="custom/custom",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            headers={"X-Custom-Header": "custom-value"},
            api_base="https://example.com/api/v1",
        )

        mock_post.assert_called_once()
        assert mock_post.call_args[1]["headers"]["X-Custom-Header"] == "custom-value"


def test_custom_provider_with_extra_body():
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    with patch.object(
        litellm.llms.custom_httpx.http_handler.HTTPHandler, "post"
    ) as mock_post:
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
    with patch.object(
        litellm.llms.custom_httpx.http_handler.HTTPHandler, "post"
    ) as mock_post:
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
async def test_extra_body_with_fallback(
    respx_mock: respx.MockRouter, set_openrouter_api_key
):
    """
    test regression for https://github.com/BerriAI/litellm/issues/8425.

    This was perhaps a wider issue with the acompletion function not passing kwargs such as extra_body correctly when fallbacks are specified.
    """

    # since this uses respx, we need to set use_aiohttp_transport to False
    litellm.disable_aiohttp_transport = True
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

    respx_mock.post("https://openrouter.ai/api/v1/chat/completions").respond(
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
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    response = await litellm.acompletion(
        model=model,
        messages=messages,
        extra_body=extra_body,
        fallbacks=fallbacks,
        api_key="fake-openrouter-api-key",
    )

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

    # Verify the response
    assert response is not None
    assert response.choices[0].message.content == "Hello from mocked response!"


@pytest.mark.parametrize("env_base", ["OPENAI_BASE_URL", "OPENAI_API_BASE"])
@pytest.mark.asyncio
async def test_openai_env_base(
    respx_mock: respx.MockRouter, env_base, openai_api_response, monkeypatch
):
    "This tests OpenAI env variables are honored, including legacy OPENAI_API_BASE"
    litellm.disable_aiohttp_transport = True

    expected_base_url = "http://localhost:12345/v1"

    # Assign the environment variable based on env_base, and use a fake API key.
    monkeypatch.setenv(env_base, expected_base_url)
    monkeypatch.setenv("OPENAI_API_KEY", "fake_openai_api_key")

    model = "gpt-4o"
    messages = [{"role": "user", "content": "Hello, how are you?"}]

    respx_mock.post(f"{expected_base_url}/chat/completions").respond(
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
            "usage": {"prompt_tokens": 9, "completion_tokens": 12, "total_tokens": 21},
        }
    )

    response = await litellm.acompletion(model=model, messages=messages)

    # verify we had a response
    assert response.choices[0].message.content == "Hello from mocked response!"


def build_database_url(username, password, host, dbname):
    username_enc = urllib.parse.quote_plus(username)
    password_enc = urllib.parse.quote_plus(password)
    dbname_enc = urllib.parse.quote_plus(dbname)
    return f"postgresql://{username_enc}:{password_enc}@{host}/{dbname_enc}"


def test_build_database_url():
    url = build_database_url("user@name", "p@ss:word", "localhost", "db/name")
    assert url == "postgresql://user%40name:p%40ss%3Aword@localhost/db%2Fname"


def test_bedrock_llama():
    litellm._turn_on_debug()
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


def test_responses_api_bridge_check_strips_responses_prefix():
    """Test that responses_api_bridge_check strips 'responses/' prefix and sets mode."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main._get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.return_value = {"max_tokens": 4096}

        model_info, model = responses_api_bridge_check(
            model="responses/gpt-4-responses",
            custom_llm_provider="openai",
        )

        assert model == "gpt-4-responses"
        assert model_info["mode"] == "responses"


def test_responses_api_bridge_check_handles_exception():
    """Test that responses_api_bridge_check handles exceptions and still processes responses/ models."""
    from litellm.main import responses_api_bridge_check

    with patch("litellm.main._get_model_info_helper") as mock_get_model_info:
        mock_get_model_info.side_effect = Exception("Model not found")

        model_info, model = responses_api_bridge_check(
            model="responses/custom-model", custom_llm_provider="custom"
        )

        assert model == "custom-model"
        assert model_info["mode"] == "responses"


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

import json

from collections.abc import Mapping
from typing import Final
from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest

import litellm
from litellm import Message
from litellm.llms.bedrock.common_utils import get_bedrock_chat_config
from litellm.llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation import (
    AmazonMoonshotConfig,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.utils import ModelResponse
from litellm.utils import CustomStreamWrapper

AWS_AUTH_PARAMS = {
    "aws_access_key_id": "AKIAEXAMPLE",
    "aws_secret_access_key": "secret",
    "aws_session_token": "token",
    "aws_region_name": "us-west-2",
    "aws_session_name": "session",
    "aws_role_name": "arn:aws:iam::000000000000:role/example",
    "aws_web_identity_token": "web-identity",
    "aws_sts_endpoint": "https://sts.us-west-2.amazonaws.com",
    "aws_bedrock_runtime_endpoint": "https://bedrock-runtime.us-west-2.amazonaws.com",
    "aws_external_id": "external",
    "aws_session_tags": [{"Key": "team", "Value": "genai"}],
}


def test_transform_request_never_resolves_aws_credentials():
    """A broken credential chain must not stop the request body from being built."""
    config = AmazonMoonshotConfig()

    transformed = config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"aws_profile_name": "litellm-profile-that-does-not-exist", "max_tokens": 16},
        litellm_params={},
        headers={},
    )

    assert transformed["model"] == "moonshot.kimi-k2-thinking"
    assert transformed["max_tokens"] == 16
    assert "aws_profile_name" not in transformed


@pytest.mark.parametrize("aws_param", sorted(AWS_AUTH_PARAMS))
def test_transform_request_keeps_aws_params_out_of_the_body(aws_param: str):
    config = AmazonMoonshotConfig()

    transformed = config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={aws_param: AWS_AUTH_PARAMS[aws_param]},
        litellm_params={},
        headers={},
    )

    assert aws_param not in transformed


def test_transform_request_leaves_the_caller_aws_params_in_place_for_signing():
    """sign_request reads the aws_* keys off optional_params after transform_request runs."""
    config = AmazonMoonshotConfig()
    optional_params = dict(AWS_AUTH_PARAMS)

    config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert optional_params == AWS_AUTH_PARAMS


def _make_moonshot_response(content: str = "Hi!") -> httpx.Response:
    body: Final = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "moonshot.kimi-k2-thinking",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    return httpx.Response(
        status_code=200,
        headers={"Content-Type": "application/json"},
        json=body,
        request=httpx.Request("POST", "https://bedrock-runtime.us-west-2.amazonaws.com"),
    )


def _invoke_with_mocked_post(
    *,
    messages: list[dict[str, object] | Message],
    extra_kwargs: Mapping[str, object] | None = None,
    response_content: str = "Hi!",
) -> tuple[Mock, ModelResponse]:
    """Run a sync litellm.completion() with HTTPHandler.post patched to
    return a canned moonshot response. Returns (mock_post, response)."""
    client: Final = HTTPHandler()
    mock_response: Final = _make_moonshot_response(content=response_content)
    mock_post: Final = Mock(return_value=mock_response)
    with patch.object(client, "post", new=mock_post):
        response: Final = litellm.completion(
            model="bedrock/invoke/moonshot.kimi-k2-thinking",
            messages=messages,
            stream=False,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-west-2",
            client=client,
            **(extra_kwargs or {}),
        )
    return mock_post, response


def test_developer_role_translation() -> None:
    """Verify LiteLLM maps the ``developer`` role to ``system`` on the
    outgoing Bedrock invoke request, without hitting the network."""
    mock_post, response = _invoke_with_mocked_post(
        messages=[
            {"role": "developer", "content": "Be a good bot!"},
            {"role": "user", "content": "Hello, how are you?"},
        ],
    )
    mock_post.assert_called_once()
    body: Final = json.loads(mock_post.call_args.kwargs["data"])
    assert body["messages"][0]["role"] == "system"
    assert body["messages"][0]["content"] == "Be a good bot!"
    assert body["messages"][1]["role"] == "user"
    assert response.choices[0].message.content is not None


def test_message_with_name() -> None:
    """Verify a user message carrying a ``name`` field is serialized into
    the outgoing Bedrock invoke request without breaking the call."""
    mock_post, response = _invoke_with_mocked_post(
        messages=[{"role": "user", "content": "Hello", "name": "test_name"}],
    )
    mock_post.assert_called_once()
    body: Final = json.loads(mock_post.call_args.kwargs["data"])
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "Hello"
    assert response is not None


def test_content_list_handling() -> None:
    mock_post, response = _invoke_with_mocked_post(
        messages=[
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello, how are you?"}],
            }
        ],
    )
    mock_post.assert_called_once()
    assert response.choices[0].message.content is not None


def test_pydantic_model_input() -> None:
    """Verify a completion call with a pydantic ``Message`` as input does
    not raise and produces a parseable response."""

    mock_post, response = _invoke_with_mocked_post(
        messages=[Message(content="Hello, how are you?", role="user")],
    )
    mock_post.assert_called_once()
    assert response is not None


@pytest.mark.parametrize("response_format", [{"type": "text"}])
def test_response_format_type_text_with_tool_calls_no_tool_choice(
    response_format: dict[str, str],
) -> None:
    """Verify response_format + tools + drop_params sends a valid request
    and produces a response object."""
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
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    mock_post, response = _invoke_with_mocked_post(
        messages=[{"role": "user", "content": "What's the weather like in Boston today?"}],
        extra_kwargs={
            "response_format": response_format,
            "tools": tools,
            "drop_params": True,
        },
    )
    mock_post.assert_called_once()
    body: Final = json.loads(mock_post.call_args.kwargs["data"])
    assert "tools" in body
    assert body["tools"][0]["function"]["name"] == "get_current_weather"
    assert response is not None


def test_streaming() -> None:
    client: Final = HTTPHandler()
    stream_response: Final = Mock(
        status_code=200,
        headers=httpx.Headers(),
        iter_bytes=Mock(return_value=iter(())),
    )
    mock_post: Final = Mock(return_value=stream_response)
    with patch.object(client, "post", new=mock_post):
        response: Final = litellm.completion(
            model="bedrock/invoke/moonshot.kimi-k2-thinking",
            messages=[
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, how are you?"}],
                }
            ],
            stream=True,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-west-2",
            client=client,
        )

    assert isinstance(response, CustomStreamWrapper)
    mock_post.assert_called_once()
    call_args: Final = mock_post.call_args
    assert call_args is not None
    assert call_args.args[0].endswith("/invoke-with-response-stream")
    body: Final = json.loads(call_args.kwargs["data"])
    assert body["messages"][0]["role"] == "user"


async def test_completion_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    mock_response: Final = _make_moonshot_response()
    client: Final = AsyncHTTPHandler()
    mock_post: Final = AsyncMock(return_value=mock_response)
    with patch.object(client, "post", new=mock_post):
        response: Final = await litellm.acompletion(
            model="bedrock/invoke/moonshot.kimi-k2-thinking",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            stream=False,
            aws_access_key_id="fake",
            aws_secret_access_key="fake",
            aws_region_name="us-west-2",
            client=client,
        )

    assert response._hidden_params["response_cost"] > 0


def test_provider_detection_invoke() -> None:
    """Test that Bedrock Moonshot invoke models are correctly detected."""
    config: Final = get_bedrock_chat_config("bedrock/invoke/moonshot.kimi-k2-thinking")
    assert config is not None
    assert config.__class__.__name__ == "AmazonMoonshotConfig"


def test_provider_detection_converse() -> None:
    """Test that Bedrock Moonshot converse models are correctly detected."""
    config: Final = get_bedrock_chat_config("bedrock/moonshot.kimi-k2-thinking")
    assert config is not None


def test_config_initialization() -> None:
    """Test that AmazonMoonshotConfig initializes correctly."""
    config: Final = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")
    assert config is not None
    assert config.custom_llm_provider == "bedrock"


def test_supported_params() -> None:
    """Test that supported OpenAI params are correctly defined."""
    config: Final = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")
    supported_params: Final = config.get_supported_openai_params("moonshot.kimi-k2-thinking")

    assert "temperature" in supported_params
    assert "max_tokens" in supported_params
    assert "top_p" in supported_params
    assert "stream" in supported_params
    assert "tools" in supported_params
    assert "tool_choice" in supported_params

    assert "stop" not in supported_params

    assert "functions" not in supported_params


def test_transform_request_strips_model_prefix() -> None:
    """Test that model ID prefixes are correctly stripped in transform_request."""

    config: Final = AmazonMoonshotConfig()

    messages: Final = [{"role": "user", "content": "Hello"}]

    transformed: Final = config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert transformed["model"] == "moonshot.kimi-k2-thinking"


def test_reasoning_content_extraction() -> None:
    """Test that reasoning content is extracted from <reasoning> tags."""

    config: Final = AmazonMoonshotConfig()

    content_with_reasoning: Final = "<reasoning>This is my thought process</reasoning>This is the answer"
    reasoning_with_tags, content_with_tags = config._extract_reasoning_from_content(content_with_reasoning)

    assert reasoning_with_tags == "This is my thought process"
    assert content_with_tags == "This is the answer"
    assert "<reasoning>" not in content_with_tags

    content_without_reasoning: Final = "This is just a regular answer"
    reasoning_without_tags, content_without_tags = config._extract_reasoning_from_content(content_without_reasoning)

    assert reasoning_without_tags is None
    assert content_without_tags == "This is just a regular answer"


def test_tool_calling_supported() -> None:
    """Test that tool calling is supported for Kimi K2 Thinking model."""
    config: Final = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")
    supported_params: Final = config.get_supported_openai_params("moonshot.kimi-k2-thinking")

    assert "tools" in supported_params
    assert "tool_choice" in supported_params


def test_tool_call_request_format() -> None:
    """Test that tool call requests are formatted correctly."""

    config: Final = AmazonMoonshotConfig()

    messages: Final = [{"role": "user", "content": "What's the weather in San Francisco?"}]

    optional_params: Final = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
    }

    transformed: Final = config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert transformed["model"] == "moonshot.kimi-k2-thinking"

    assert "tools" in transformed
    assert len(transformed["tools"]) == 1
    assert transformed["tools"][0]["function"]["name"] == "get_weather"


def test_stop_sequences_not_supported() -> None:
    """Test that stop sequences are correctly excluded from supported params."""
    config: Final = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")
    supported_params: Final = config.get_supported_openai_params("moonshot.kimi-k2-thinking")

    assert "stop" not in supported_params


def test_temperature_range() -> None:
    """Test that temperature parameter is handled correctly."""
    config: Final = get_bedrock_chat_config("invoke/moonshot.kimi-k2-thinking")

    assert config is not None
    supported_params: Final = config.get_supported_openai_params("moonshot.kimi-k2-thinking")
    assert "temperature" in supported_params


def test_transform_request_basic() -> None:
    """Test basic request transformation."""

    config: Final = AmazonMoonshotConfig()

    messages: Final = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    optional_params: Final = {"temperature": 0.7, "max_tokens": 100}

    transformed: Final = config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert transformed["model"] == "moonshot.kimi-k2-thinking"

    assert "messages" in transformed
    assert len(transformed["messages"]) >= 1

    assert transformed["temperature"] == 0.7
    assert transformed["max_tokens"] == 100


def test_transform_request_with_system_message() -> None:
    """Test request transformation with system message."""

    config: Final = AmazonMoonshotConfig()

    messages: Final = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]

    transformed: Final = config.transform_request(
        model="moonshot.kimi-k2-thinking",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )

    assert "messages" in transformed

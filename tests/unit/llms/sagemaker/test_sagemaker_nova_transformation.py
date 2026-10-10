"""
Unit tests for SageMaker Nova transformation config.
"""

import binascii
import json
import struct
from typing import Final

import pytest

from litellm.llms.sagemaker.common_utils import AWSEventStreamDecoder
from litellm.llms.sagemaker.nova.transformation import SagemakerNovaConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse
from litellm.utils import convert_to_model_response_object


def _event_frame(payload: bytes) -> bytes:
    headers: Final = {
        ":event-type": "PayloadPart",
        ":content-type": "application/json",
        ":message-type": "event",
    }
    encoded_headers: Final = b"".join(
        struct.pack("B", len(name.encode()))
        + name.encode()
        + struct.pack("B", 7)
        + struct.pack(">H", len(value.encode()))
        + value.encode()
        for name, value in headers.items()
    )
    prelude: Final = struct.pack(">I", 16 + len(encoded_headers) + len(payload)) + struct.pack(
        ">I", len(encoded_headers)
    )
    prefix: Final = prelude + struct.pack(">I", binascii.crc32(prelude) & 0xFFFFFFFF)
    message: Final = prefix + encoded_headers + payload
    return message + struct.pack(">I", binascii.crc32(message) & 0xFFFFFFFF)


def _nova_stream_frame(content: str, finish_reason: str | None) -> bytes:
    chunk: Final = {
        "id": "chatcmpl-nova-stream",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "nova-endpoint",
        "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
    }
    return _event_frame(f"data: {json.dumps(chunk)}\n\n".encode())


class TestSagemakerNovaConfig:
    def setup_method(self):
        self.config = SagemakerNovaConfig()

    def test_should_support_stream_param_in_request_body(self):
        """Nova requires stream: true in the request body."""
        assert self.config.supports_stream_param_in_request_body is True

    def test_should_include_nova_specific_params(self):
        """Nova-specific params should be in the supported params list."""
        params = self.config.get_supported_openai_params(model="my-nova-endpoint")
        assert "top_k" in params
        assert "reasoning_effort" in params
        assert "allowed_token_ids" in params
        assert "truncate_prompt_tokens" in params

    def test_should_include_standard_openai_params(self):
        """Standard OpenAI params from parent should still be present."""
        params = self.config.get_supported_openai_params(model="my-nova-endpoint")
        assert "temperature" in params
        assert "max_tokens" in params
        assert "top_p" in params
        assert "stream" in params
        assert "logprobs" in params
        assert "top_logprobs" in params
        assert "stream_options" in params

    def test_should_map_nova_params_to_request(self):
        """Nova-specific params should pass through to optional_params."""
        optional_params = self.config.map_openai_params(
            non_default_params={
                "top_k": 40,
                "reasoning_effort": "low",
                "temperature": 0.7,
            },
            optional_params={},
            model="my-nova-endpoint",
            drop_params=False,
        )
        assert optional_params["top_k"] == 40
        assert optional_params["reasoning_effort"] == "low"
        assert optional_params["temperature"] == 0.7

    def test_should_generate_correct_url_non_streaming(self):
        """Non-streaming URL should use /invocations."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="my-nova-endpoint",
            optional_params={"aws_region_name": "us-east-1"},
            litellm_params={},
            stream=False,
        )
        assert (
            url
            == "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-nova-endpoint/invocations"
        )

    def test_should_generate_correct_url_streaming(self):
        """Streaming URL should use /invocations-response-stream."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="my-nova-endpoint",
            optional_params={"aws_region_name": "us-east-1"},
            litellm_params={},
            stream=True,
        )
        assert (
            url
            == "https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/my-nova-endpoint/invocations-response-stream"
        )

    def test_should_have_custom_stream_wrapper(self):
        """Nova should use custom stream wrapper (AWS EventStream)."""
        assert self.config.has_custom_stream_wrapper is True


class TestSagemakerNovaResponseParsing:
    """Test that Nova's OpenAI-compatible responses are correctly parsed."""

    def test_should_parse_non_streaming_response(self):
        """Nova non-streaming response should be parsed into ModelResponse."""
        nova_response = {
            "id": "chatcmpl-123e4567-e89b-12d3-a456-426614174000",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "nova-micro-custom",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help?",
                        "refusal": None,
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 9,
                "completion_tokens": 12,
                "total_tokens": 21,
            },
        }
        result = convert_to_model_response_object(
            response_object=nova_response,
            model_response_object=ModelResponse(),
        )
        assert result.id == "chatcmpl-123e4567-e89b-12d3-a456-426614174000"
        assert result.choices[0].message.content == "Hello! How can I help?"
        assert result.choices[0].finish_reason == "stop"
        assert result.usage.prompt_tokens == 9
        assert result.usage.completion_tokens == 12
        assert result.usage.total_tokens == 21

    def test_should_parse_response_with_reasoning_content(self):
        """Nova reasoning_content should be extracted correctly."""
        nova_response = {
            "id": "chatcmpl-reasoning-test",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "nova-2-lite-custom",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "The answer is 4.",
                        "reasoning_content": "Let me think: 2+2=4",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 20,
                "total_tokens": 35,
            },
        }
        result = convert_to_model_response_object(
            response_object=nova_response,
            model_response_object=ModelResponse(),
        )
        assert result.choices[0].message.content == "The answer is 4."
        assert result.choices[0].message.reasoning_content == "Let me think: 2+2=4"

    def test_should_parse_response_with_logprobs(self):
        """Nova logprobs should be preserved in response."""
        nova_response = {
            "id": "chatcmpl-logprobs-test",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "nova-micro-custom",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello",
                    },
                    "logprobs": {
                        "content": [
                            {
                                "token": "Hello",
                                "logprob": -0.5,
                                "top_logprobs": [
                                    {"token": "Hello", "logprob": -0.5},
                                    {"token": "Hi", "logprob": -1.2},
                                ],
                            }
                        ]
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 1,
                "total_tokens": 6,
            },
        }
        result = convert_to_model_response_object(
            response_object=nova_response,
            model_response_object=ModelResponse(),
        )
        assert result.choices[0].logprobs is not None
        assert result.choices[0].logprobs["content"][0]["token"] == "Hello"

    def test_should_parse_response_with_cached_tokens(self):
        """Nova prompt_tokens_details with cached_tokens should be parsed."""
        nova_response = {
            "id": "chatcmpl-cached-test",
            "object": "chat.completion",
            "created": 1677652288,
            "model": "nova-micro-custom",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hi",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 1,
                "total_tokens": 21,
                "prompt_tokens_details": {"cached_tokens": 10},
            },
        }
        result = convert_to_model_response_object(
            response_object=nova_response,
            model_response_object=ModelResponse(),
        )
        assert result.usage.prompt_tokens_details.cached_tokens == 10


class TestSagemakerChatBackwardsCompatibility:
    """Verify that changes to SagemakerChatConfig don't break existing sagemaker_chat callers."""

    def setup_method(self):
        from litellm.llms.sagemaker.chat.transformation import SagemakerChatConfig

        self.config = SagemakerChatConfig()

    def test_should_not_support_stream_param_in_request_body(self):
        """sagemaker_chat should NOT send stream in request body (unchanged behavior)."""
        assert self.config.supports_stream_param_in_request_body is False

    def test_should_generate_correct_urls(self):
        """sagemaker_chat URLs should be unchanged."""
        url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="my-hf-endpoint",
            optional_params={"aws_region_name": "us-west-2"},
            litellm_params={},
            stream=False,
        )
        assert (
            url
            == "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-hf-endpoint/invocations"
        )

        stream_url = self.config.get_complete_url(
            api_base=None,
            api_key=None,
            model="my-hf-endpoint",
            optional_params={"aws_region_name": "us-west-2"},
            litellm_params={},
            stream=True,
        )
        assert (
            stream_url
            == "https://runtime.sagemaker.us-west-2.amazonaws.com/endpoints/my-hf-endpoint/invocations-response-stream"
        )

    def test_should_still_have_custom_stream_wrapper(self):
        """sagemaker_chat should still use custom stream wrapper."""
        assert self.config.has_custom_stream_wrapper is True

    def test_should_not_include_nova_specific_params(self):
        """sagemaker_chat should NOT have Nova-specific params."""
        params = self.config.get_supported_openai_params(model="my-hf-endpoint")
        assert "top_k" not in params
        assert "reasoning_effort" not in params
        assert "allowed_token_ids" not in params
        assert "truncate_prompt_tokens" not in params

    def test_should_preserve_standard_openai_params(self):
        """sagemaker_chat should still support standard OpenAI params."""
        params = self.config.get_supported_openai_params(model="my-hf-endpoint")
        assert "temperature" in params
        assert "max_tokens" in params
        assert "top_p" in params
        assert "stream" in params

    def test_sync_stream_wrapper_uses_correct_provider_string(self):
        """
        Verify that when get_sync_custom_stream_wrapper is called with
        custom_llm_provider="sagemaker_chat", the CustomStreamWrapper
        receives "sagemaker_chat" (not something else).
        """
        from unittest.mock import patch, MagicMock

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_bytes.return_value = iter([])
        mock_client.post.return_value = mock_response

        with patch(
            "litellm.llms.sagemaker.chat.transformation.CustomStreamWrapper"
        ) as mock_csw:
            mock_csw.return_value = MagicMock()
            self.config.get_sync_custom_stream_wrapper(
                litellm_params={},
                model="my-hf-endpoint",
                custom_llm_provider="sagemaker_chat",
                logging_obj=MagicMock(),
                api_base="https://example.com",
                headers={},
                data={},
                messages=[],
                client=mock_client,
            )
            mock_csw.assert_called_once()
            call_kwargs = mock_csw.call_args[1]
            assert call_kwargs["custom_llm_provider"] == "sagemaker_chat"

    def test_async_stream_wrapper_uses_correct_provider_string(self):
        """
        Verify that when get_async_custom_stream_wrapper is called with
        custom_llm_provider="sagemaker_chat", the CustomStreamWrapper
        receives "sagemaker_chat".
        """
        import asyncio
        from unittest.mock import patch, MagicMock, AsyncMock

        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def empty_aiter():
            return
            yield  # make it an async generator

        mock_response.aiter_bytes.return_value = empty_aiter()
        mock_client.post.return_value = mock_response

        with patch(
            "litellm.llms.sagemaker.chat.transformation.CustomStreamWrapper"
        ) as mock_csw:
            mock_csw.return_value = MagicMock()
            asyncio.run(
                self.config.get_async_custom_stream_wrapper(
                    litellm_params={},
                    model="my-hf-endpoint",
                    custom_llm_provider="sagemaker_chat",
                    logging_obj=MagicMock(),
                    api_base="https://example.com",
                    headers={},
                    data={},
                    messages=[],
                    client=mock_client,
                )
            )
            mock_csw.assert_called_once()
            call_kwargs = mock_csw.call_args[1]
            assert call_kwargs["custom_llm_provider"] == "sagemaker_chat"

    def test_async_stream_wrapper_llm_provider_enum_resolves(self):
        """
        Verify LlmProviders(custom_llm_provider) resolves correctly for
        "sagemaker_chat" and doesn't fall through to the ValueError fallback.
        """
        from litellm.types.utils import LlmProviders

        provider = LlmProviders("sagemaker_chat")
        assert provider == LlmProviders.SAGEMAKER_CHAT


class TestSagemakerNovaTransformRequest:
    """Test Nova-specific request transformation."""

    def setup_method(self):
        self.config = SagemakerNovaConfig()

    def test_should_not_include_model_in_request_body(self):
        """Nova SageMaker endpoints reject 'model' in the request body."""
        request = self.config.transform_request(
            model="my-nova-endpoint",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={"temperature": 0.7},
            litellm_params={},
            headers={},
        )
        assert "model" not in request
        assert "messages" in request
        assert request["temperature"] == 0.7

    def test_should_preserve_system_and_multi_turn_messages_in_request(self):
        messages: Final[list[AllMessageValues]] = [
            {"role": "system", "content": "Remember the user's name"},
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "I will remember"},
            {"role": "user", "content": "What is my name?"},
        ]
        request = self.config.transform_request(
            model="my-nova-endpoint",
            messages=messages,
            optional_params={},
            litellm_params={},
            headers={},
        )

        assert request["messages"] == messages

    def test_should_include_all_nova_params_in_request(self):
        """Nova-specific params should appear in the request body."""
        request = self.config.transform_request(
            model="my-nova-endpoint",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={
                "top_k": 40,
                "max_tokens": 512,
                "reasoning_effort": "low",
            },
            litellm_params={},
            headers={},
        )
        assert "model" not in request
        assert request["top_k"] == 40
        assert request["max_tokens"] == 512
        assert request["reasoning_effort"] == "low"

    def test_should_parse_streamed_nova_chunks_and_final_usage(self):
        config: Final = SagemakerNovaConfig()
        chunks: Final = tuple(
            chunk
            for chunk in AWSEventStreamDecoder(model="nova-endpoint", is_messages_api=True).iter_bytes(
                iter(
                    [
                        _nova_stream_frame("Nova ", None),
                        _nova_stream_frame("stream", "stop"),
                    ]
                )
            )
            if chunk is not None
        )

        assert config.supports_stream_param_in_request_body
        assert tuple(chunk.choices[0].delta.content for chunk in chunks) == ("Nova ", "stream")
        assert chunks[-1].choices[0].finish_reason == "stop"
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.total_tokens == 3

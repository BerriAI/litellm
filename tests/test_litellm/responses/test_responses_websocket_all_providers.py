"""
Unit tests to verify that all providers support Responses API WebSocket mode.

Tests that:
1. All providers with ResponsesAPIConfig support websocket mode
2. Providers with native websocket support use direct connection
3. Providers without native websocket support use ManagedResponsesWebSocketHandler
"""

import json
from unittest.mock import MagicMock

import pytest

from litellm.llms.azure.responses.transformation import AzureOpenAIResponsesAPIConfig
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.databricks.responses.transformation import (
    DatabricksResponsesAPIConfig,
)
from litellm.llms.github_copilot.responses.transformation import (
    GithubCopilotResponsesAPIConfig,
)
from litellm.llms.hosted_vllm.responses.transformation import (
    HostedVLLMResponsesAPIConfig,
)
from litellm.llms.litellm_proxy.responses.transformation import (
    LiteLLMProxyResponsesAPIConfig,
)
from litellm.llms.manus.responses.transformation import ManusResponsesAPIConfig
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.llms.openrouter.responses.transformation import (
    OpenRouterResponsesAPIConfig,
)
from litellm.llms.perplexity.responses.transformation import PerplexityResponsesConfig
from litellm.llms.volcengine.responses.transformation import (
    VolcEngineResponsesAPIConfig,
)
from litellm.llms.xai.responses.transformation import XAIResponsesAPIConfig


class TestResponsesAPIWebSocketSupport:
    """Test that all providers have websocket support configured correctly"""

    def test_openai_supports_native_websocket(self):
        """OpenAI should support native websocket"""
        config = OpenAIResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is True
        ), "OpenAI should support native websocket"

    def test_azure_supports_native_websocket(self):
        """Azure should support native websocket"""
        config = AzureOpenAIResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is True
        ), "Azure should support native websocket"

    def test_azure_websocket_url_uses_v1_path(self):
        """Azure WebSocket URL must use /openai/v1/responses (no api-version)"""
        config = AzureOpenAIResponsesAPIConfig()
        url = config.get_websocket_url(
            api_base="https://myresource.cognitiveservices.azure.com",
            litellm_params={"api_version": "2025-04-01-preview"},
        )
        assert url == "wss://myresource.cognitiveservices.azure.com/openai/v1/responses"
        assert "api-version" not in url

    def test_azure_websocket_url_strips_existing_path(self):
        """api_base that already contains /openai/responses must be cleaned"""
        config = AzureOpenAIResponsesAPIConfig()
        url = config.get_websocket_url(
            api_base="https://myresource.cognitiveservices.azure.com/openai/responses",
            litellm_params={},
        )
        assert url == "wss://myresource.cognitiveservices.azure.com/openai/v1/responses"

    def test_azure_websocket_url_strips_query_params(self):
        config = AzureOpenAIResponsesAPIConfig()
        url = config.get_websocket_url(
            api_base="https://myresource.cognitiveservices.azure.com/openai/responses?api-version=2024-05-01-preview",
            litellm_params={},
        )
        assert url == "wss://myresource.cognitiveservices.azure.com/openai/v1/responses"

    def test_azure_websocket_url_requires_api_base(self):
        config = AzureOpenAIResponsesAPIConfig()
        with pytest.raises(ValueError):
            config.get_websocket_url(api_base=None, litellm_params={})

    def test_azure_model_not_in_websocket_url(self):
        """Azure sends the model in the body, so it must not be appended to the URL"""
        assert AzureOpenAIResponsesAPIConfig().model_in_websocket_url() is False

    def test_openai_default_websocket_url_converts_scheme(self):
        """The base get_websocket_url default converts the HTTP endpoint to wss://"""
        config = OpenAIResponsesAPIConfig()
        url = config.get_websocket_url(
            api_base="https://api.openai.com/v1", litellm_params={}
        )
        assert url == "wss://api.openai.com/v1/responses"

    def test_openai_model_in_websocket_url_default(self):
        assert OpenAIResponsesAPIConfig().model_in_websocket_url() is True

    @pytest.mark.asyncio
    async def test_openai_websocket_forwards_explicit_api_key(self, monkeypatch):
        from unittest.mock import AsyncMock

        import litellm
        from litellm.responses import main as responses_main

        websocket_handler = AsyncMock()
        monkeypatch.setattr(litellm, "api_key", None)
        monkeypatch.setattr(litellm, "openai_key", None)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            responses_main.base_llm_http_handler,
            "async_responses_websocket",
            websocket_handler,
        )

        await responses_main._aresponses_websocket.__wrapped__(
            model="gpt-4o",
            websocket=MagicMock(),
            api_key="explicit-api-key",
            litellm_logging_obj=MagicMock(),
        )

        assert websocket_handler.await_args.kwargs["api_key"] == "explicit-api-key"

    def test_xai_uses_managed_websocket(self):
        """XAI should use managed websocket handler"""
        config = XAIResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "XAI should use managed websocket handler"

    def test_github_copilot_uses_managed_websocket(self):
        """GitHub Copilot should use managed websocket handler"""
        config = GithubCopilotResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "GitHub Copilot should use managed websocket handler"

    def test_chatgpt_uses_managed_websocket(self):
        """ChatGPT should use managed websocket handler"""
        config = ChatGPTResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "ChatGPT should use managed websocket handler"

    def test_litellm_proxy_uses_managed_websocket(self):
        """LiteLLM Proxy should use managed websocket handler"""
        config = LiteLLMProxyResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "LiteLLM Proxy should use managed websocket handler"

    def test_volcengine_uses_managed_websocket(self):
        """VolcEngine should use managed websocket handler"""
        config = VolcEngineResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "VolcEngine should use managed websocket handler"

    def test_manus_uses_managed_websocket(self):
        """Manus should use managed websocket handler"""
        config = ManusResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Manus should use managed websocket handler"

    def test_perplexity_uses_managed_websocket(self):
        """Perplexity should use managed websocket handler"""
        config = PerplexityResponsesConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Perplexity should use managed websocket handler"

    def test_databricks_uses_managed_websocket(self):
        """Databricks should use managed websocket handler"""
        config = DatabricksResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Databricks should use managed websocket handler"

    def test_openrouter_uses_managed_websocket(self):
        """OpenRouter should use managed websocket handler"""
        config = OpenRouterResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "OpenRouter should use managed websocket handler"

    def test_hosted_vllm_uses_managed_websocket(self):
        """Hosted vLLM should use managed websocket handler"""
        config = HostedVLLMResponsesAPIConfig()
        assert (
            config.supports_native_websocket() is False
        ), "Hosted vLLM should use managed websocket handler"


class TestManagedWebSocketHandlerIntegration:
    """Test that ManagedResponsesWebSocketHandler is properly integrated"""

    @pytest.mark.asyncio
    async def test_managed_handler_instantiation(self):
        """Test that ManagedResponsesWebSocketHandler can be instantiated"""
        from unittest.mock import MagicMock

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        mock_websocket = MagicMock()
        mock_logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=True,
            call_type="aresponses",
            start_time=0,
            litellm_call_id="test-id",
            function_id="test-func",
        )

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="test-model",
            logging_obj=mock_logging_obj,
            user_api_key_dict=None,
            litellm_metadata={},
            api_key="test-key",
            api_base="https://api.example.com",
            timeout=30.0,
            custom_llm_provider="test_provider",
        )

        assert handler.model == "test-model"
        assert handler.api_key == "test-key"
        assert handler.api_base == "https://api.example.com"
        assert handler.timeout == 30.0
        assert handler.custom_llm_provider == "test_provider"

    @pytest.mark.asyncio
    async def test_frame_alias_resolves_to_connection_model(self, monkeypatch):
        """
        A response.create frame that repeats the public model alias must reach
        litellm.aresponses with the router-resolved deployment model, not the
        raw alias (which fails in get_llm_provider). Regression for codex
        WebSocket sessions against managed providers like bedrock_mantle.
        """
        import json
        from unittest.mock import AsyncMock, MagicMock

        import litellm
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        captured: dict = {}

        async def fake_aresponses(*args, **kwargs):
            captured["model"] = kwargs.get("model")

            async def _empty():
                return
                yield

            return _empty()

        monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="bedrock_mantle/openai.gpt-5.5",
            logging_obj=Logging(
                model="bedrock_mantle/openai.gpt-5.5",
                messages=[],
                stream=True,
                call_type="aresponses",
                start_time=0,
                litellm_call_id="test-id",
                function_id="test-func",
            ),
            litellm_metadata={"model_group": "gpt-5.5-mantle"},
        )

        frame = json.dumps(
            {
                "type": "response.create",
                "model": "gpt-5.5-mantle",
                "input": [],
            }
        )
        await handler._process_response_create(frame)

        assert captured["model"] == "bedrock_mantle/openai.gpt-5.5"

    @pytest.mark.asyncio
    async def test_warmup_frame_skips_provider_and_sends_synthetic_ack(
        self, monkeypatch
    ):
        """
        A generate=false warmup frame (codex prewarm) carries empty input that
        managed HTTP providers reject. It must not call the provider, and should
        emit synthetic response.created/completed events so Codex can proceed.
        """
        import json
        from unittest.mock import AsyncMock, MagicMock

        import litellm
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        called = False

        async def fail_aresponses(*args, **kwargs):
            nonlocal called
            called = True
            raise AssertionError("provider must not be called for a warmup frame")

        monkeypatch.setattr(litellm, "aresponses", fail_aresponses)

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="bedrock_mantle/openai.gpt-5.5",
            logging_obj=Logging(
                model="bedrock_mantle/openai.gpt-5.5",
                messages=[],
                stream=True,
                call_type="aresponses",
                start_time=0,
                litellm_call_id="test-id",
                function_id="test-func",
            ),
            litellm_metadata={"model_group": "gpt-5.5-mantle"},
        )

        frame = json.dumps(
            {
                "type": "response.create",
                "model": "gpt-5.5-mantle",
                "generate": False,
                "input": [],
            }
        )
        await handler._process_response_create(frame)

        assert called is False
        assert mock_websocket.send_text.call_count == 2
        events = [
            json.loads(call.args[0]) for call in mock_websocket.send_text.call_args_list
        ]
        assert events[0]["type"] == "response.created"
        assert events[0]["response"]["status"] == "in_progress"
        assert events[1]["type"] == "response.completed"
        assert events[1]["response"]["status"] == "completed"
        assert events[1]["response"]["output"] == []
        assert events[1]["response"]["model"] == "gpt-5.5-mantle"

    @pytest.mark.asyncio
    async def test_warmup_previous_response_id_not_forwarded_to_provider(
        self, monkeypatch
    ):
        import json
        from unittest.mock import AsyncMock, MagicMock

        import litellm
        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        captured: dict = {}

        async def fake_aresponses(*args, **kwargs):
            captured.update(kwargs)

            async def _empty():
                return
                yield

            return _empty()

        monkeypatch.setattr(litellm, "aresponses", fake_aresponses)

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="bedrock_mantle/openai.gpt-5.5",
            logging_obj=Logging(
                model="bedrock_mantle/openai.gpt-5.5",
                messages=[],
                stream=True,
                call_type="aresponses",
                start_time=0,
                litellm_call_id="test-id",
                function_id="test-func",
            ),
            litellm_metadata={"model_group": "gpt-5.5-mantle"},
        )

        await handler._process_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "gpt-5.5-mantle",
                    "generate": False,
                    "input": [],
                }
            )
        )
        warmup_id = json.loads(mock_websocket.send_text.call_args_list[1].args[0])[
            "response"
        ]["id"]

        await handler._process_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "model": "gpt-5.5-mantle",
                    "previous_response_id": warmup_id,
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": "Hi"}],
                        }
                    ],
                }
            )
        )

        assert "previous_response_id" not in captured


class TestChunkTransformation:
    """Test chunk serialization and transformation for WebSocket streaming"""

    def test_serialize_chunk_with_dict(self):
        """Test serialization of dict chunks"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.created",
            "response": {"id": "resp_456", "status": "in_progress"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.created" in serialized
        assert "resp_456" in serialized

    def test_serialize_chunk_handles_invalid_json(self):
        """Test that chunks with circular references are handled"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        # Create object with circular reference
        obj = {"a": 1}
        obj["self"] = obj  # type: ignore

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(obj)
        assert serialized is None

    def test_extract_output_messages_with_text_content(self):
        """Test extraction of output messages with text content"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Hello world"}],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"][0]["text"] == "Hello world"

    def test_extract_output_messages_with_multiple_content_parts(self):
        """Test extraction with multiple content parts"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Part 1. "},
                            {"type": "output_text", "text": "Part 2."},
                        ],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Part 1. Part 2."

    def test_extract_output_messages_with_function_calls(self):
        """Test that function calls are preserved"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["type"] == "function_call"
        assert messages[0]["id"] == "call_123"
        assert messages[0]["name"] == "get_weather"

    def test_extract_output_messages_filters_empty_text(self):
        """Test that messages with empty text are filtered out"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": ""}],
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Valid text"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Valid text"

    def test_extract_output_messages_handles_non_dict_items(self):
        """Test that non-dict items are skipped"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    "invalid_string",
                    None,
                    123,
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Valid"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Valid"

    def test_input_to_messages_with_string(self):
        """Test conversion of string input to messages"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        messages = ManagedResponsesWebSocketHandler._input_to_messages("Hello world")
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][0]["text"] == "Hello world"

    def test_input_to_messages_with_list(self):
        """Test conversion of list input to messages"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Question"}],
            }
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert messages[0]["type"] == "message"
        assert messages[0]["content"][0]["text"] == "Question"

    def test_input_to_messages_filters_non_dict_items(self):
        """Test that non-dict items in list input are filtered"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            "invalid_string",
            None,
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "Valid"}],
            },
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Valid"

    def test_input_to_messages_handles_empty_input(self):
        """Test that empty input returns empty list"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        assert ManagedResponsesWebSocketHandler._input_to_messages(None) == []
        assert ManagedResponsesWebSocketHandler._input_to_messages([]) == []
        assert ManagedResponsesWebSocketHandler._input_to_messages({}) == []


class TestWebSocketEventTypes:
    """Test that all WebSocket event types are properly handled with dict-based chunks"""

    def test_serialize_response_created_event_dict(self):
        """Test serialization of response.created event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.created",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "object": "response",
                "status": "in_progress",
                "created_at": 1234567890,
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.created" in serialized
        assert "resp_123" in serialized

    def test_serialize_response_in_progress_event_dict(self):
        """Test serialization of response.in_progress event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {"type": "response.in_progress", "response_id": "resp_123"}

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.in_progress" in serialized

    def test_serialize_output_item_added_event_dict(self):
        """Test serialization of response.output_item.added event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_item.added",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_item.added" in serialized
        assert "msg_456" in serialized

    def test_serialize_output_text_delta_event_dict(self):
        """Test serialization of response.output_text.delta event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_text.delta",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "delta": "Hello",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_text.delta" in serialized
        assert "Hello" in serialized

    def test_serialize_output_text_done_event_dict(self):
        """Test serialization of response.output_text.done event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_text.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "text": "Hello world",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_text.done" in serialized
        assert "Hello world" in serialized

    def test_serialize_content_part_done_event_dict(self):
        """Test serialization of response.content_part.done event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.content_part.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "output_text", "text": "Complete text"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.content_part.done" in serialized

    def test_serialize_output_item_done_event_dict(self):
        """Test serialization of response.output_item.done event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.output_item.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "item": {"type": "message", "role": "assistant", "status": "completed"},
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.output_item.done" in serialized
        assert "msg_456" in serialized

    def test_serialize_response_completed_event_dict(self):
        """Test serialization of response.completed event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.completed",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "Done"}],
                    }
                ],
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.completed" in serialized
        assert "resp_123" in serialized

    def test_serialize_response_failed_event_dict(self):
        """Test serialization of response.failed event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.failed",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "status": "failed",
                "status_details": {"error": {"message": "Rate limit exceeded"}},
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.failed" in serialized
        assert "Rate limit exceeded" in serialized

    def test_serialize_response_incomplete_event_dict(self):
        """Test serialization of response.incomplete event as dict"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.incomplete",
            "response_id": "resp_123",
            "response": {
                "id": "resp_123",
                "status": "incomplete",
                "status_details": {"reason": "max_output_tokens"},
            },
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.incomplete" in serialized
        assert "max_output_tokens" in serialized


class TestMultiTurnSessionHistory:
    """Test multi-turn conversation handling via session history"""

    def test_extract_output_messages_preserves_multiple_messages(self):
        """Test that multiple output messages are all preserved"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "First message"}],
                    },
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": "{}",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Second message"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 3
        assert messages[0]["content"][0]["text"] == "First message"
        assert messages[1]["type"] == "function_call"
        assert messages[2]["content"][0]["text"] == "Second message"

    def test_input_to_messages_with_mixed_content_types(self):
        """Test input conversion with mixed content types"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Question"},
                    {"type": "input_image", "image_url": "https://example.com/img.png"},
                ],
            }
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][1]["type"] == "input_image"

    def test_extract_output_messages_with_mixed_text_types(self):
        """Test that both 'output_text' and 'text' types are extracted"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Part 1"},
                            {"type": "text", "text": "Part 2"},
                        ],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Part 1Part 2"

    def test_extract_response_id_from_completed_event(self):
        """Test extraction of response ID from completed event"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {"id": "resp_abc123", "status": "completed"},
        }

        response_id = ManagedResponsesWebSocketHandler._extract_response_id(
            completed_event
        )
        assert response_id == "resp_abc123"

    def test_extract_response_id_handles_missing_response(self):
        """Test that missing response dict returns None"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {"type": "response.completed"}

        response_id = ManagedResponsesWebSocketHandler._extract_response_id(
            completed_event
        )
        assert response_id is None


class TestWebSocketErrorHandling:
    """Test error handling in WebSocket mode"""

    @pytest.mark.asyncio
    async def test_managed_handler_handles_invalid_json(self):
        """Test that invalid JSON in response.create is handled gracefully"""
        from unittest.mock import AsyncMock, MagicMock

        from litellm.litellm_core_utils.litellm_logging import Logging
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        mock_websocket = MagicMock()
        mock_websocket.send_text = AsyncMock()
        mock_websocket.recv = AsyncMock(return_value="invalid json {{{")

        mock_logging_obj = Logging(
            model="test-model",
            messages=[],
            stream=True,
            call_type="aresponses",
            start_time=0,
            litellm_call_id="test-id",
            function_id="test-func",
        )

        handler = ManagedResponsesWebSocketHandler(
            websocket=mock_websocket,
            model="test-model",
            logging_obj=mock_logging_obj,
        )

        # Process invalid JSON
        await handler._process_response_create("invalid json {{{")

        # Should have sent an error event
        mock_websocket.send_text.assert_called_once()
        error_event = mock_websocket.send_text.call_args[0][0]
        assert "error" in error_event
        assert "Invalid JSON" in error_event


class TestNativeWebSocketGuardrails:
    @pytest.mark.asyncio
    async def test_response_create_injects_authorized_model(self):
        import json
        from unittest.mock import MagicMock

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        handler = ResponsesWebSocketStreaming(
            websocket=MagicMock(),
            backend_ws=MagicMock(),
            logging_obj=MagicMock(),
            authorized_model="authorized-deployment",
        )

        flat_message = await handler._mask_response_create(
            json.dumps({"type": "response.create", "input": "hi"})
        )
        nested_message = await handler._mask_response_create(
            json.dumps({"type": "response.create", "response": {"input": "hi"}})
        )

        assert json.loads(flat_message)["model"] == "authorized-deployment"
        assert (
            json.loads(nested_message)["response"]["model"] == "authorized-deployment"
        )

    @pytest.mark.asyncio
    async def test_completed_event_with_null_response_passes_through(self):
        from unittest.mock import MagicMock

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        class Guardrail:
            def get_presidio_settings_from_request_data(self, request_data):
                return None

            def _unmask_pii_text(self, text, pii_tokens):
                return text

        event = '{"type":"response.completed","response":null}'
        guardrail = Guardrail()
        handler = ResponsesWebSocketStreaming(
            websocket=MagicMock(),
            backend_ws=MagicMock(),
            logging_obj=MagicMock(),
            request_data={"metadata": {"pii_tokens": {"<TOKEN_1>": "secret"}}},
            guardrail_callbacks=[guardrail],
            output_guardrail_callbacks=[guardrail],
        )

        assert handler._unmask_response_event(event) == event
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_output_masking_suppresses_delta_without_calling_presidio(self):
        import json
        from unittest.mock import AsyncMock, MagicMock

        import websockets.exceptions

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        class RecordingGuardrail:
            def __init__(self):
                self.check_pii_calls = []

            def get_presidio_settings_from_request_data(self, request_data):
                return None

            def _unmask_pii_text(self, text, pii_tokens):
                return text

            async def check_pii(
                self, text, output_parse_pii, presidio_config, request_data
            ):
                self.check_pii_calls.append(text)
                return text

        class FakeBackendWS:
            def __init__(self, events):
                self._events = list(events)

            async def recv(self, decode=False):
                if self._events:
                    return self._events.pop(0)
                raise websockets.exceptions.ConnectionClosed(None, None)

        guardrail = RecordingGuardrail()
        client_ws = MagicMock()
        client_ws.send_text = AsyncMock()
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        delta_event = json.dumps(
            {"type": "response.output_text.delta", "delta": "alice@example.com"}
        )
        completed_event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": "alice@example.com"}
                            ]
                        }
                    ]
                },
            }
        )

        handler = ResponsesWebSocketStreaming(
            websocket=client_ws,
            backend_ws=FakeBackendWS([delta_event, completed_event]),
            logging_obj=logging_obj,
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # The delta event must be suppressed without ever invoking Presidio,
        # so check_pii is called exactly once (for the completed event only).
        assert guardrail.check_pii_calls == ["alice@example.com"]
        client_ws.send_text.assert_called_once()
        sent_payload = client_ws.send_text.call_args[0][0]
        assert json.loads(sent_payload)["type"] == "response.completed"

    @pytest.mark.asyncio
    async def test_output_masking_suppresses_text_bearing_done_events(self):
        import json
        from unittest.mock import AsyncMock, MagicMock

        import websockets.exceptions

        from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

        class MaskingGuardrail:
            def __init__(self):
                self.check_pii_calls = []

            def get_presidio_settings_from_request_data(self, request_data):
                return None

            def _unmask_pii_text(self, text, pii_tokens):
                return text

            async def check_pii(
                self, text, output_parse_pii, presidio_config, request_data
            ):
                self.check_pii_calls.append(text)
                return text.replace("alice@example.com", "<EMAIL_ADDRESS>")

        class FakeBackendWS:
            def __init__(self, events):
                self._events = list(events)

            async def recv(self, decode=False):
                if self._events:
                    return self._events.pop(0)
                raise websockets.exceptions.ConnectionClosed(None, None)

        guardrail = MaskingGuardrail()
        client_ws = MagicMock()
        client_ws.send_text = AsyncMock()
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        done_events = [
            json.dumps(
                {"type": "response.output_text.done", "text": "alice@example.com"}
            ),
            json.dumps(
                {
                    "type": "response.content_part.done",
                    "part": {"type": "output_text", "text": "alice@example.com"},
                }
            ),
            json.dumps(
                {
                    "type": "response.output_item.done",
                    "item": {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "alice@example.com"}
                        ],
                    },
                }
            ),
        ]
        completed_event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": "alice@example.com"}
                            ]
                        }
                    ]
                },
            }
        )

        handler = ResponsesWebSocketStreaming(
            websocket=client_ws,
            backend_ws=FakeBackendWS(done_events + [completed_event]),
            logging_obj=logging_obj,
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # Text-bearing done events carry the full output before response.completed
        # arrives; they must be suppressed so unmasked PII never reaches the
        # client, and Presidio is only invoked for response.completed.
        assert guardrail.check_pii_calls == ["alice@example.com"]
        client_ws.send_text.assert_called_once()
        sent_payload = client_ws.send_text.call_args[0][0]
        assert json.loads(sent_payload)["type"] == "response.completed"
        assert "alice@example.com" not in sent_payload
        assert "<EMAIL_ADDRESS>" in sent_payload


class _FakeWSGuardrail:
    """Presidio-like guardrail double for the WebSocket masking hooks.

    ``check_pii`` replaces each known PII string with its token. When
    ``output_parse_pii`` is True (input masking) the token->original map is
    persisted into ``request_data["metadata"]["pii_tokens"]`` so the response
    path can reverse it. ``_unmask_pii_text`` performs that reversal.
    """

    def __init__(self, mask_map=None):
        self.mask_map = mask_map or {"alice@example.com": "<EMAIL_ADDRESS_1>"}
        self.output_parse_pii = True
        self.apply_to_output = True

    def get_presidio_settings_from_request_data(self, request_data):
        return None

    async def check_pii(self, text, output_parse_pii, presidio_config, request_data):
        masked = text
        tokens = {}
        for original, token in self.mask_map.items():
            if original in masked:
                masked = masked.replace(original, token)
                tokens[token] = original
        if output_parse_pii and tokens:
            metadata = request_data.setdefault("metadata", {})
            metadata.setdefault("pii_tokens", {}).update(tokens)
        return masked

    def _unmask_pii_text(self, text, pii_tokens):
        for token, original in pii_tokens.items():
            text = text.replace(token, original)
        return text


def _make_streaming(**kwargs):
    from unittest.mock import MagicMock

    from litellm.responses.streaming_iterator import ResponsesWebSocketStreaming

    kwargs.setdefault("websocket", MagicMock())
    kwargs.setdefault("backend_ws", MagicMock())
    kwargs.setdefault("logging_obj", MagicMock())
    return ResponsesWebSocketStreaming(**kwargs)


class TestNativeWebSocketGuardrailMasking:
    """Exercises the input/output PII masking hooks on ResponsesWebSocketStreaming."""

    @pytest.mark.asyncio
    async def test_mask_response_create_flat_string_input(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={},
            guardrail_callbacks=[guardrail],
            authorized_model="auth-model",
        )

        masked = await handler._mask_response_create(
            json.dumps(
                {"type": "response.create", "input": "email alice@example.com now"}
            )
        )
        obj = json.loads(masked)

        assert obj["model"] == "auth-model"
        assert obj["input"] == "email <EMAIL_ADDRESS_1> now"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_mask_response_create_list_content_string(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": "ping alice@example.com",
                        }
                    ],
                }
            )
        )
        obj = json.loads(masked)

        assert obj["input"][0]["content"] == "ping <EMAIL_ADDRESS_1>"

    @pytest.mark.asyncio
    async def test_mask_response_create_input_text_blocks(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        {
                            "type": "message",
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": "alice@example.com"},
                                {"type": "input_image", "image_url": "http://x"},
                            ],
                        }
                    ],
                }
            )
        )
        obj = json.loads(masked)
        blocks = obj["input"][0]["content"]

        assert blocks[0]["text"] == "<EMAIL_ADDRESS_1>"
        assert blocks[1]["image_url"] == "http://x"

    @pytest.mark.asyncio
    async def test_mask_response_create_function_call_output_string(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": "tool returned alice@example.com",
                        }
                    ],
                }
            )
        )
        obj = json.loads(masked)

        assert obj["input"][0]["output"] == "tool returned <EMAIL_ADDRESS_1>"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_mask_response_create_function_call_output_blocks(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        {
                            "type": "function_call_output",
                            "call_id": "call_1",
                            "output": [
                                {"type": "output_text", "text": "alice@example.com"},
                                {"type": "input_image", "image_url": "http://x"},
                            ],
                        }
                    ],
                }
            )
        )
        obj = json.loads(masked)
        blocks = obj["input"][0]["output"]

        assert blocks[0]["text"] == "<EMAIL_ADDRESS_1>"
        assert blocks[1]["image_url"] == "http://x"

    @pytest.mark.asyncio
    async def test_mask_response_create_nested_shape(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={},
            guardrail_callbacks=[guardrail],
            authorized_model="auth-model",
        )

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {"input": "alice@example.com", "model": "spoofed"},
                }
            )
        )
        obj = json.loads(masked)

        assert obj["response"]["model"] == "auth-model"
        assert obj["response"]["input"] == "<EMAIL_ADDRESS_1>"

    @pytest.mark.asyncio
    async def test_mask_response_create_flat_instructions(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": "hi",
                    "instructions": "reply to alice@example.com",
                }
            )
        )
        obj = json.loads(masked)

        assert obj["instructions"] == "reply to <EMAIL_ADDRESS_1>"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_mask_response_create_nested_instructions(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "response": {
                        "input": "hi",
                        "instructions": "email alice@example.com",
                    },
                }
            )
        )
        obj = json.loads(masked)

        assert obj["response"]["instructions"] == "email <EMAIL_ADDRESS_1>"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_mask_response_create_non_create_unchanged(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={},
            guardrail_callbacks=[guardrail],
            authorized_model="auth-model",
        )

        message = json.dumps({"type": "response.cancel", "input": "alice@example.com"})
        assert await handler._mask_response_create(message) == message

    @pytest.mark.asyncio
    async def test_mask_response_create_invalid_json_unchanged(self):
        handler = _make_streaming(
            request_data={}, guardrail_callbacks=[_FakeWSGuardrail()]
        )
        assert await handler._mask_response_create("not json {{{") == "not json {{{"

    @pytest.mark.asyncio
    async def test_mask_response_create_model_only_without_guardrails(self):
        handler = _make_streaming(request_data={}, authorized_model="auth-model")

        masked = await handler._mask_response_create(
            json.dumps({"type": "response.create", "input": "alice@example.com"})
        )
        obj = json.loads(masked)

        assert obj["model"] == "auth-model"
        assert obj["input"] == "alice@example.com"

    @pytest.mark.asyncio
    async def test_mask_response_create_no_op_without_model_or_guardrails(self):
        handler = _make_streaming(request_data={})
        message = json.dumps({"type": "response.create", "input": "alice@example.com"})
        assert await handler._mask_response_create(message) == message

    @pytest.mark.asyncio
    async def test_mask_response_create_list_with_non_dict_item(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])

        masked = await handler._mask_response_create(
            json.dumps(
                {
                    "type": "response.create",
                    "input": [
                        "not-a-dict",
                        {
                            "type": "message",
                            "role": "user",
                            "content": "alice@example.com",
                        },
                    ],
                }
            )
        )
        obj = json.loads(masked)
        assert obj["input"][0] == "not-a-dict"
        assert obj["input"][1]["content"] == "<EMAIL_ADDRESS_1>"

    def test_enforce_authorized_model_no_authorized_model(self):
        handler = _make_streaming(request_data={})
        assert handler._enforce_authorized_model({"model": "anything"}) is False

    def test_enforce_authorized_model_nested_with_top_level_model(self):
        handler = _make_streaming(request_data={}, authorized_model="auth-model")
        msg = {"response": {"model": "spoofed"}, "model": "also-spoofed"}
        assert handler._enforce_authorized_model(msg) is True
        assert msg["response"]["model"] == "auth-model"
        assert msg["model"] == "auth-model"

    @pytest.mark.asyncio
    async def test_unmask_response_event_completed(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={
                "metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "alice@example.com"}}
            },
            guardrail_callbacks=[guardrail],
        )

        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "content": [
                                {"type": "output_text", "text": "to <EMAIL_ADDRESS_1>"}
                            ]
                        }
                    ]
                },
            }
        )
        unmasked = json.loads(handler._unmask_response_event(event))
        assert (
            unmasked["response"]["output"][0]["content"][0]["text"]
            == "to alice@example.com"
        )

    @pytest.mark.asyncio
    async def test_unmask_response_event_delta(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={
                "metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "alice@example.com"}}
            },
            guardrail_callbacks=[guardrail],
        )

        event = json.dumps(
            {"type": "response.output_text.delta", "delta": "<EMAIL_ADDRESS_1>"}
        )
        unmasked = json.loads(handler._unmask_response_event(event))
        assert unmasked["delta"] == "alice@example.com"

    def test_unmask_response_event_no_tokens_unchanged(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(request_data={}, guardrail_callbacks=[guardrail])
        event = json.dumps(
            {"type": "response.output_text.delta", "delta": "<EMAIL_ADDRESS_1>"}
        )
        assert handler._unmask_response_event(event) == event

    def test_unmask_response_event_no_guardrails_unchanged(self):
        handler = _make_streaming(
            request_data={"metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "x"}}}
        )
        event = json.dumps({"type": "response.completed", "response": {}})
        assert handler._unmask_response_event(event) == event

    def test_unmask_response_event_invalid_json_unchanged(self):
        handler = _make_streaming(
            request_data={"metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "x"}}},
            guardrail_callbacks=[_FakeWSGuardrail()],
        )
        assert handler._unmask_response_event("not json {{{") == "not json {{{"

    def test_unmask_response_event_non_dict_response_unchanged(self):
        handler = _make_streaming(
            request_data={"metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "x"}}},
            guardrail_callbacks=[_FakeWSGuardrail()],
        )
        event = json.dumps({"type": "response.completed", "response": ["bad-shape"]})
        assert handler._unmask_response_event(event) == event

    def test_unmask_response_event_malformed_output_items_unchanged(self):
        handler = _make_streaming(
            request_data={
                "metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "alice@example.com"}}
            },
            guardrail_callbacks=[_FakeWSGuardrail()],
        )
        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        "not-a-dict",
                        {"content": "not-a-list"},
                        {"content": ["not-a-dict-block"]},
                    ]
                },
            }
        )
        assert handler._unmask_response_event(event) == event

    def test_unmask_response_event_other_event_type_unchanged(self):
        handler = _make_streaming(
            request_data={"metadata": {"pii_tokens": {"<EMAIL_ADDRESS_1>": "x"}}},
            guardrail_callbacks=[_FakeWSGuardrail()],
        )
        event = json.dumps(
            {"type": "response.in_progress", "delta": "<EMAIL_ADDRESS_1>"}
        )
        assert handler._unmask_response_event(event) == event

    @pytest.mark.asyncio
    async def test_mask_response_completed_event(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[guardrail]
        )

        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "contact alice@example.com",
                                }
                            ]
                        }
                    ]
                },
            }
        )
        masked = json.loads(await handler._mask_response_completed(event))
        assert (
            masked["response"]["output"][0]["content"][0]["text"]
            == "contact <EMAIL_ADDRESS_1>"
        )

    @pytest.mark.asyncio
    async def test_mask_response_completed_masks_function_call_arguments(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[guardrail]
        )

        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "function_call",
                            "name": "send_email",
                            "arguments": '{"to": "alice@example.com"}',
                        }
                    ]
                },
            }
        )
        masked = json.loads(await handler._mask_response_completed(event))
        assert (
            masked["response"]["output"][0]["arguments"]
            == '{"to": "<EMAIL_ADDRESS_1>"}'
        )

    @pytest.mark.asyncio
    async def test_mask_response_completed_masks_reasoning_summary(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[guardrail]
        )

        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "reasoning",
                            "summary": [
                                {
                                    "type": "summary_text",
                                    "text": "user is alice@example.com",
                                }
                            ],
                        }
                    ]
                },
            }
        )
        masked = json.loads(await handler._mask_response_completed(event))
        assert (
            masked["response"]["output"][0]["summary"][0]["text"]
            == "user is <EMAIL_ADDRESS_1>"
        )

    @pytest.mark.asyncio
    async def test_mask_response_completed_delta_unchanged(self):
        guardrail = _FakeWSGuardrail()
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[guardrail]
        )

        event = json.dumps(
            {"type": "response.output_text.delta", "delta": "alice@example.com"}
        )
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_mask_response_completed_no_guardrails_unchanged(self):
        handler = _make_streaming(request_data={})
        event = json.dumps(
            {"type": "response.output_text.delta", "delta": "alice@example.com"}
        )
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_mask_response_completed_invalid_json_unchanged(self):
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[_FakeWSGuardrail()]
        )
        assert await handler._mask_response_completed("not json {{{") == "not json {{{"

    @pytest.mark.asyncio
    async def test_mask_response_completed_malformed_unchanged(self):
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[_FakeWSGuardrail()]
        )
        event = json.dumps(
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        "not-a-dict",
                        {"content": "not-a-list"},
                        {"content": ["not-a-dict-block"]},
                    ]
                },
            }
        )
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_mask_response_completed_non_dict_response_unchanged(self):
        handler = _make_streaming(
            request_data={}, output_guardrail_callbacks=[_FakeWSGuardrail()]
        )
        event = json.dumps({"type": "response.completed", "response": ["bad"]})
        assert await handler._mask_response_completed(event) == event

    @pytest.mark.asyncio
    async def test_client_to_backend_masks_and_enforces_model(self):
        from unittest.mock import AsyncMock

        guardrail = _FakeWSGuardrail()
        backend_ws = MagicMock()
        backend_ws.send = AsyncMock()
        websocket = MagicMock()
        websocket.receive_text = AsyncMock(
            side_effect=[
                json.dumps(
                    {"type": "response.create", "input": "ping alice@example.com"}
                ),
                Exception("stop"),
            ]
        )

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            request_data={},
            first_message=json.dumps(
                {"type": "response.create", "input": "alice@example.com"}
            ),
            guardrail_callbacks=[guardrail],
            authorized_model="auth-model",
        )

        await handler.client_to_backend()

        assert backend_ws.send.await_count == 2
        first_sent = json.loads(backend_ws.send.await_args_list[0][0][0])
        assert first_sent["model"] == "auth-model"
        assert first_sent["input"] == "<EMAIL_ADDRESS_1>"
        second_sent = json.loads(backend_ws.send.await_args_list[1][0][0])
        assert second_sent["model"] == "auth-model"
        assert second_sent["input"] == "ping <EMAIL_ADDRESS_1>"
        assert handler.request_data["metadata"]["pii_tokens"] == {
            "<EMAIL_ADDRESS_1>": "alice@example.com"
        }

    @pytest.mark.asyncio
    async def test_backend_to_client_suppresses_deltas_and_masks_completed(self):
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401  (lazy submodule must be importable)

        guardrail = _FakeWSGuardrail()
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps(
                    {"type": "response.output_text.delta", "delta": "alice@example.com"}
                ),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [
                                {
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "contact alice@example.com",
                                        }
                                    ]
                                }
                            ]
                        },
                    }
                ),
                Exception("stop"),
            ]
        )
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
            request_data={},
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        websocket.send_text.assert_awaited_once()
        forwarded = json.loads(websocket.send_text.await_args[0][0])
        assert forwarded["type"] == "response.completed"
        assert (
            forwarded["response"]["output"][0]["content"][0]["text"]
            == "contact <EMAIL_ADDRESS_1>"
        )

    @pytest.mark.asyncio
    async def test_backend_to_client_suppresses_function_call_arguments_done(self):
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401  (lazy submodule must be importable)

        guardrail = _FakeWSGuardrail()
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "type": "response.function_call_arguments.done",
                        "arguments": '{"to": "alice@example.com"}',
                    }
                ),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [
                                {
                                    "type": "function_call",
                                    "name": "send_email",
                                    "arguments": '{"to": "alice@example.com"}',
                                }
                            ]
                        },
                    }
                ),
                Exception("stop"),
            ]
        )
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
            request_data={},
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # The unmasked function-call arguments must never reach the client; only
        # the masked response.completed is forwarded.
        websocket.send_text.assert_awaited_once()
        sent_payload = websocket.send_text.await_args[0][0]
        forwarded = json.loads(sent_payload)
        assert forwarded["type"] == "response.completed"
        assert (
            forwarded["response"]["output"][0]["arguments"]
            == '{"to": "<EMAIL_ADDRESS_1>"}'
        )
        assert "alice@example.com" not in sent_payload

    @pytest.mark.asyncio
    async def test_backend_to_client_suppresses_reasoning_summary_text_done(self):
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401  (lazy submodule must be importable)

        guardrail = _FakeWSGuardrail()
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "type": "response.reasoning_summary_text.done",
                        "text": "contact alice@example.com",
                    }
                ),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [
                                {
                                    "content": [
                                        {
                                            "type": "output_text",
                                            "text": "done",
                                        }
                                    ]
                                }
                            ]
                        },
                    }
                ),
                Exception("stop"),
            ]
        )
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
            request_data={},
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # The reasoning-summary done event carries the full reasoning text before
        # response.completed arrives; it must be suppressed so unmasked PII never
        # reaches the client.
        websocket.send_text.assert_awaited_once()
        sent_payload = websocket.send_text.await_args[0][0]
        assert json.loads(sent_payload)["type"] == "response.completed"
        assert "alice@example.com" not in sent_payload

    @pytest.mark.asyncio
    async def test_backend_to_client_suppresses_reasoning_summary_part_done(self):
        from unittest.mock import AsyncMock

        import websockets.exceptions  # noqa: F401  (lazy submodule must be importable)

        guardrail = _FakeWSGuardrail()
        websocket = MagicMock()
        websocket.send_text = AsyncMock()
        backend_ws = MagicMock()
        backend_ws.recv = AsyncMock(
            side_effect=[
                json.dumps(
                    {
                        "type": "response.reasoning_summary_part.done",
                        "part": {
                            "type": "summary_text",
                            "text": "user is alice@example.com",
                        },
                    }
                ),
                json.dumps(
                    {
                        "type": "response.completed",
                        "response": {
                            "output": [
                                {
                                    "type": "reasoning",
                                    "summary": [
                                        {
                                            "type": "summary_text",
                                            "text": "user is alice@example.com",
                                        }
                                    ],
                                }
                            ]
                        },
                    }
                ),
                Exception("stop"),
            ]
        )
        logging_obj = MagicMock()
        logging_obj.dispatch_success_handlers = AsyncMock()

        handler = _make_streaming(
            websocket=websocket,
            backend_ws=backend_ws,
            logging_obj=logging_obj,
            request_data={},
            output_guardrail_callbacks=[guardrail],
        )

        await handler.backend_to_client()

        # The reasoning-summary part-done event carries the full reasoning text
        # before response.completed arrives; it must be suppressed, and the
        # reasoning summary in response.completed must itself be masked.
        websocket.send_text.assert_awaited_once()
        sent_payload = websocket.send_text.await_args[0][0]
        forwarded = json.loads(sent_payload)
        assert forwarded["type"] == "response.completed"
        assert (
            forwarded["response"]["output"][0]["summary"][0]["text"]
            == "user is <EMAIL_ADDRESS_1>"
        )
        assert "alice@example.com" not in sent_payload


class TestWebSocketChunkTypes:
    """Test handling of different chunk types from streaming responses"""

    def test_serialize_function_call_chunk(self):
        """Test serialization of function call chunks"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.function_call.added",
            "response_id": "resp_123",
            "item_id": "call_456",
            "output_index": 0,
            "call_id": "call_456",
            "name": "get_weather",
            "arguments": "",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.function_call.added" in serialized
        assert "get_weather" in serialized

    def test_serialize_function_call_arguments_delta(self):
        """Test serialization of function call arguments delta"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.function_call_arguments.delta",
            "response_id": "resp_123",
            "item_id": "call_456",
            "output_index": 0,
            "call_id": "call_456",
            "delta": '{"location"',
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.function_call_arguments.delta" in serialized
        assert "location" in serialized

    def test_serialize_function_call_arguments_done(self):
        """Test serialization of function call arguments done"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.function_call_arguments.done",
            "response_id": "resp_123",
            "item_id": "call_456",
            "output_index": 0,
            "call_id": "call_456",
            "arguments": '{"location": "Paris"}',
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.function_call_arguments.done" in serialized
        assert "Paris" in serialized

    def test_serialize_reasoning_content_delta(self):
        """Test serialization of reasoning content delta"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.reasoning_content.delta",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "delta": "Thinking step 1...",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.reasoning_content.delta" in serialized
        assert "Thinking step 1" in serialized

    def test_serialize_reasoning_content_done(self):
        """Test serialization of reasoning content done"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        chunk = {
            "type": "response.reasoning_content.done",
            "response_id": "resp_123",
            "item_id": "msg_456",
            "output_index": 0,
            "content_index": 0,
            "reasoning_content": "Complete reasoning...",
        }

        serialized = ManagedResponsesWebSocketHandler._serialize_chunk(chunk)
        assert serialized is not None
        assert "response.reasoning_content.done" in serialized
        assert "Complete reasoning" in serialized

    def test_extract_output_messages_preserves_multiple_messages(self):
        """Test that multiple output messages are all preserved"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "First message"}],
                    },
                    {
                        "type": "function_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": "{}",
                    },
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "Second message"}],
                    },
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 3
        assert messages[0]["content"][0]["text"] == "First message"
        assert messages[1]["type"] == "function_call"
        assert messages[2]["content"][0]["text"] == "Second message"

    def test_input_to_messages_with_mixed_content_types(self):
        """Test input conversion with mixed content types"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        input_list = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Question"},
                    {"type": "input_image", "image_url": "https://example.com/img.png"},
                ],
            }
        ]

        messages = ManagedResponsesWebSocketHandler._input_to_messages(input_list)
        assert len(messages) == 1
        assert len(messages[0]["content"]) == 2
        assert messages[0]["content"][0]["type"] == "input_text"
        assert messages[0]["content"][1]["type"] == "input_image"

    def test_extract_output_messages_with_mixed_text_types(self):
        """Test that both 'output_text' and 'text' types are extracted"""
        from litellm.responses.streaming_iterator import (
            ManagedResponsesWebSocketHandler,
        )

        completed_event = {
            "type": "response.completed",
            "response": {
                "id": "resp_123",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {"type": "output_text", "text": "Part 1"},
                            {"type": "text", "text": "Part 2"},
                        ],
                    }
                ],
            },
        }

        messages = ManagedResponsesWebSocketHandler._extract_output_messages(
            completed_event
        )
        assert len(messages) == 1
        assert messages[0]["content"][0]["text"] == "Part 1Part 2"


class TestNativeWebSocketUrlConstruction:
    """Test that native WebSocket URLs include the model query parameter.

    These tests mock websockets.connect so they exercise the actual URL-building
    code inside BaseLLMHTTPHandler.async_responses_websocket rather than
    reimplementing the logic themselves.
    """

    @pytest.mark.asyncio
    async def test_openai_ws_url_includes_model(self):
        """Handler must pass ?model= in the URL to the backend WebSocket."""
        from unittest.mock import AsyncMock, MagicMock, patch

        captured_urls = []

        class FakeConnect:
            def __init__(self, url, **kwargs):
                captured_urls.append(url)

            async def __aenter__(self):
                raise Exception("stop")

            async def __aexit__(self, *args):
                pass

        mock_config = MagicMock(spec=OpenAIResponsesAPIConfig)
        mock_config.supports_native_websocket.return_value = True
        mock_config.get_websocket_url.return_value = "wss://api.openai.com/v1/responses"
        mock_config.validate_environment.return_value = {}

        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        handler = BaseLLMHTTPHandler()

        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", FakeConnect):
            await handler.async_responses_websocket(
                model="gpt-4o-mini",
                websocket=mock_ws,
                logging_obj=mock_logging,
                responses_api_provider_config=mock_config,
                api_key="sk-test",
            )

        assert len(captured_urls) == 1
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(captured_urls[0]).query)
        assert qs.get("model") == [
            "gpt-4o-mini"
        ], f"Expected model in URL, got: {captured_urls[0]}"

    @pytest.mark.asyncio
    async def test_ws_url_preserves_existing_params_and_adds_model(self):
        """When api_base already has query params, model is added alongside them."""
        from unittest.mock import AsyncMock, MagicMock, patch

        captured_urls = []

        class FakeConnect:
            def __init__(self, url, **kwargs):
                captured_urls.append(url)

            async def __aenter__(self):
                raise Exception("stop")

            async def __aexit__(self, *args):
                pass

        mock_config = MagicMock(spec=OpenAIResponsesAPIConfig)
        mock_config.supports_native_websocket.return_value = True
        mock_config.get_websocket_url.return_value = (
            "wss://custom.example.com/v1/responses?api-version=2024-05-01"
        )
        mock_config.validate_environment.return_value = {}

        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        handler = BaseLLMHTTPHandler()
        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()

        with patch("websockets.connect", FakeConnect):
            await handler.async_responses_websocket(
                model="gpt-4o",
                websocket=mock_ws,
                logging_obj=mock_logging,
                responses_api_provider_config=mock_config,
                api_key="sk-test",
            )

        assert len(captured_urls) == 1
        from urllib.parse import parse_qs, urlparse

        qs = parse_qs(urlparse(captured_urls[0]).query)
        assert qs.get("model") == [
            "gpt-4o"
        ], f"model missing from URL: {captured_urls[0]}"
        assert qs.get("api-version") == [
            "2024-05-01"
        ], f"existing param lost: {captured_urls[0]}"

    @pytest.mark.asyncio
    async def test_ws_passes_litellm_params_to_get_websocket_url(self):
        """Deployment api_version must reach get_websocket_url (Azure WS URL)."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_config = MagicMock(spec=OpenAIResponsesAPIConfig)
        mock_config.supports_native_websocket.return_value = True
        mock_config.get_websocket_url.return_value = (
            "wss://example.openai.azure.com/openai/v1/responses"
        )
        mock_config.validate_environment.return_value = {}

        mock_logging = MagicMock()
        mock_logging.pre_call = MagicMock()

        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler

        handler = BaseLLMHTTPHandler()
        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()

        class FakeConnect:
            def __init__(self, url, **kwargs):
                pass

            async def __aenter__(self):
                raise Exception("stop")

            async def __aexit__(self, *args):
                pass

        with patch("websockets.connect", FakeConnect):
            await handler.async_responses_websocket(
                model="gpt-5.3-codex",
                websocket=mock_ws,
                logging_obj=mock_logging,
                responses_api_provider_config=mock_config,
                api_key="sk-test",
                api_base="https://example.openai.azure.com",
                api_version="2025-04-01-preview",
            )

        mock_config.get_websocket_url.assert_called_once()
        _, call_kwargs = mock_config.get_websocket_url.call_args
        assert call_kwargs["litellm_params"]["api_version"] == "2025-04-01-preview"

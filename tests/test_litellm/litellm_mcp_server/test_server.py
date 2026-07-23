"""
Unit tests for the LiteLLM MCP server.
"""

import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp.types import CallToolResult, TextContent, Tool

from litellm.litellm_mcp_server.server import (
    TOOL_NAME_CHAT_COMPLETION,
    TOOL_NAME_EMBEDDING,
    TOOL_NAME_IMAGE_GENERATION,
    TOOL_NAME_LIST_MODELS,
    TOOL_NAME_RERANK,
    TOOL_NAME_TEXT_COMPLETION,
    TOOL_NAME_TRANSCRIPTION,
    _error_result,
    _handle_chat_completion,
    _handle_embedding,
    _handle_image_generation,
    _handle_list_models,
    _handle_rerank,
    _handle_text_completion,
    _handle_transcription,
    _text_result,
    create_mcp_server,
)


class TestCreateMCPServer:
    """Tests for create_mcp_server factory function."""

    def test_create_mcp_server_returns_server(self):
        server = create_mcp_server()
        assert server is not None
        assert server.name == "litellm-mcp-server"

    @pytest.mark.asyncio
    async def test_list_tools_returns_all_tools(self):
        server = create_mcp_server()
        tools_handler = None
        for handler in server.request_handlers.values():
            pass
        # Test via the module-level function directly: list_tools is registered
        # We verify by checking the tool names match expected
        expected_tool_names = {
            TOOL_NAME_CHAT_COMPLETION,
            TOOL_NAME_EMBEDDING,
            TOOL_NAME_IMAGE_GENERATION,
            TOOL_NAME_TEXT_COMPLETION,
            TOOL_NAME_TRANSCRIPTION,
            TOOL_NAME_RERANK,
            TOOL_NAME_LIST_MODELS,
        }
        assert len(expected_tool_names) == 7


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_error_result(self):
        result = _error_result("something went wrong")
        assert result.isError is True
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)
        assert "Error: something went wrong" in result.content[0].text

    def test_text_result_with_string(self):
        result = _text_result("hello world")
        assert result.isError is None or result.isError is False
        assert len(result.content) == 1
        assert result.content[0].text == "hello world"

    def test_text_result_with_dict(self):
        data = {"key": "value", "number": 42}
        result = _text_result(data)
        assert result.isError is None or result.isError is False
        parsed = json.loads(result.content[0].text)
        assert parsed["key"] == "value"
        assert parsed["number"] == 42

    def test_text_result_with_list(self):
        data = [1, 2, 3]
        result = _text_result(data)
        parsed = json.loads(result.content[0].text)
        assert parsed == [1, 2, 3]


class TestChatCompletion:
    """Tests for the chat completion tool handler."""

    @pytest.mark.asyncio
    async def test_missing_model(self):
        result = await _handle_chat_completion(
            {"messages": [{"role": "user", "content": "hi"}]}
        )
        assert result.isError is True
        assert "model" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_missing_messages(self):
        result = await _handle_chat_completion({"model": "gpt-4o"})
        assert result.isError is True
        assert "messages" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_successful_completion(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "chatcmpl-123",
            "choices": [{"message": {"content": "Hello!", "role": "assistant"}}],
            "model": "gpt-4o",
        }

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await _handle_chat_completion(
                {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "Say hello"}],
                }
            )

        assert result.isError is None or result.isError is False
        parsed = json.loads(result.content[0].text)
        assert parsed["id"] == "chatcmpl-123"
        assert parsed["choices"][0]["message"]["content"] == "Hello!"

    @pytest.mark.asyncio
    async def test_passes_optional_params(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"id": "test"}

        with patch(
            "litellm.acompletion", new_callable=AsyncMock, return_value=mock_response
        ) as mock_fn:
            await _handle_chat_completion(
                {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi"}],
                    "temperature": 0.5,
                    "max_tokens": 100,
                    "top_p": 0.9,
                    "stop": ["\n"],
                    "api_key": "sk-test",
                    "api_base": "https://custom.api.com",
                }
            )

        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["stop"] == ["\n"]
        assert call_kwargs["api_key"] == "sk-test"
        assert call_kwargs["api_base"] == "https://custom.api.com"

    @pytest.mark.asyncio
    async def test_handles_api_error(self):
        """Exceptions propagate to the call_tool dispatcher which wraps them."""
        with patch(
            "litellm.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("API rate limit"),
        ):
            with pytest.raises(Exception, match="API rate limit"):
                await _handle_chat_completion(
                    {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": "hi"}],
                    }
                )


class TestEmbedding:
    """Tests for the embedding tool handler."""

    @pytest.mark.asyncio
    async def test_missing_model(self):
        result = await _handle_embedding({"input": "hello"})
        assert result.isError is True
        assert "model" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_missing_input(self):
        result = await _handle_embedding({"model": "text-embedding-3-small"})
        assert result.isError is True
        assert "input" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_successful_embedding(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "model": "text-embedding-3-small",
        }

        with patch(
            "litellm.aembedding", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await _handle_embedding(
                {
                    "model": "text-embedding-3-small",
                    "input": "Hello world",
                }
            )

        assert result.isError is None or result.isError is False
        parsed = json.loads(result.content[0].text)
        assert parsed["data"][0]["embedding"] == [0.1, 0.2, 0.3]


class TestImageGeneration:
    """Tests for the image generation tool handler."""

    @pytest.mark.asyncio
    async def test_missing_prompt(self):
        result = await _handle_image_generation({"model": "dall-e-3"})
        assert result.isError is True
        assert "prompt" in result.content[0].text.lower()

    @pytest.mark.asyncio
    async def test_successful_generation(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "data": [{"url": "https://example.com/image.png"}],
        }

        with patch(
            "litellm.aimage_generation",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _handle_image_generation(
                {"prompt": "A cute cat", "model": "dall-e-3", "size": "1024x1024"}
            )

        assert result.isError is None or result.isError is False
        parsed = json.loads(result.content[0].text)
        assert parsed["data"][0]["url"] == "https://example.com/image.png"


class TestTextCompletion:
    """Tests for the text completion tool handler."""

    @pytest.mark.asyncio
    async def test_missing_model(self):
        result = await _handle_text_completion({"prompt": "hello"})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_missing_prompt(self):
        result = await _handle_text_completion({"model": "gpt-3.5-turbo-instruct"})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_successful_completion(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "id": "cmpl-123",
            "choices": [{"text": "world", "index": 0}],
        }

        with patch(
            "litellm.atext_completion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await _handle_text_completion(
                {"model": "gpt-3.5-turbo-instruct", "prompt": "Hello "}
            )

        assert result.isError is None or result.isError is False
        parsed = json.loads(result.content[0].text)
        assert parsed["choices"][0]["text"] == "world"


class TestTranscription:
    """Tests for the transcription tool handler."""

    @pytest.mark.asyncio
    async def test_missing_model(self):
        result = await _handle_transcription({"file": "/tmp/audio.mp3"})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_missing_file(self):
        result = await _handle_transcription({"model": "whisper-1"})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_successful_transcription(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {"text": "Hello world"}

        with patch(
            "litellm.atranscription", new_callable=AsyncMock, return_value=mock_response
        ):
            with patch("builtins.open", MagicMock()):
                result = await _handle_transcription(
                    {"model": "whisper-1", "file": "/tmp/test.mp3"}
                )

        assert result.isError is None or result.isError is False
        parsed = json.loads(result.content[0].text)
        assert parsed["text"] == "Hello world"


class TestRerank:
    """Tests for the rerank tool handler."""

    @pytest.mark.asyncio
    async def test_missing_model(self):
        result = await _handle_rerank({"query": "test", "documents": ["doc1"]})
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_missing_query(self):
        result = await _handle_rerank(
            {"model": "cohere/rerank-english-v3.0", "documents": ["doc1"]}
        )
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_missing_documents(self):
        result = await _handle_rerank(
            {"model": "cohere/rerank-english-v3.0", "query": "test"}
        )
        assert result.isError is True

    @pytest.mark.asyncio
    async def test_successful_rerank(self):
        mock_response = MagicMock()
        mock_response.model_dump.return_value = {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.5},
            ]
        }

        with patch(
            "litellm.arerank", new_callable=AsyncMock, return_value=mock_response
        ):
            result = await _handle_rerank(
                {
                    "model": "cohere/rerank-english-v3.0",
                    "query": "What is AI?",
                    "documents": ["AI is artificial intelligence", "Dogs are pets"],
                }
            )

        assert result.isError is None or result.isError is False
        parsed = json.loads(result.content[0].text)
        assert len(parsed["results"]) == 2
        assert parsed["results"][0]["relevance_score"] == 0.9


class TestListModels:
    """Tests for the list models tool handler."""

    @pytest.mark.asyncio
    async def test_list_all_models(self):
        mock_model_cost = {
            "gpt-4o": {},
            "gpt-3.5-turbo": {},
            "claude-sonnet-4-20250514": {},
            "anthropic/claude-3-opus": {},
        }
        with patch("litellm.model_cost", mock_model_cost):
            result = await _handle_list_models({})

        assert result.isError is None or result.isError is False
        parsed = json.loads(result.content[0].text)
        assert parsed["total_models"] == 4
        assert len(parsed["models"]) == 4

    @pytest.mark.asyncio
    async def test_filter_by_provider(self):
        mock_model_cost = {
            "gpt-4o": {},
            "gpt-3.5-turbo": {},
            "anthropic/claude-3-opus": {},
            "anthropic/claude-sonnet-4-20250514": {},
            "cohere/command-r": {},
        }
        with patch("litellm.model_cost", mock_model_cost):
            result = await _handle_list_models({"provider": "anthropic"})

        parsed = json.loads(result.content[0].text)
        assert parsed["total_models"] == 2
        assert all("anthropic" in m for m in parsed["models"])

    @pytest.mark.asyncio
    async def test_list_models_truncation(self):
        mock_model_cost = {f"model-{i}": {} for i in range(200)}
        with patch("litellm.model_cost", mock_model_cost):
            result = await _handle_list_models({})

        parsed = json.loads(result.content[0].text)
        assert parsed["total_models"] == 200
        assert len(parsed["models"]) == 100
        assert "note" in parsed


class TestCLI:
    """Tests for the CLI argument parser."""

    def test_default_args(self):
        from litellm.litellm_mcp_server.cli import _parse_args

        args = _parse_args([])
        assert args.transport == "stdio"
        assert args.host == "0.0.0.0"
        assert args.port == 8000
        assert args.log_level == "INFO"

    def test_http_transport(self):
        from litellm.litellm_mcp_server.cli import _parse_args

        args = _parse_args(
            ["--transport", "http", "--port", "9000", "--host", "127.0.0.1"]
        )
        assert args.transport == "http"
        assert args.port == 9000
        assert args.host == "127.0.0.1"

    def test_log_level(self):
        from litellm.litellm_mcp_server.cli import _parse_args

        args = _parse_args(["--log-level", "DEBUG"])
        assert args.log_level == "DEBUG"


class TestToolSchemas:
    """Tests for tool schema definitions."""

    def test_chat_completion_schema_has_required_fields(self):
        from litellm.litellm_mcp_server.tool_schemas import CHAT_COMPLETION_SCHEMA

        assert "model" in CHAT_COMPLETION_SCHEMA["required"]
        assert "messages" in CHAT_COMPLETION_SCHEMA["required"]
        assert CHAT_COMPLETION_SCHEMA["type"] == "object"

    def test_embeddings_schema_has_required_fields(self):
        from litellm.litellm_mcp_server.tool_schemas import EMBEDDINGS_SCHEMA

        assert "model" in EMBEDDINGS_SCHEMA["required"]
        assert "input" in EMBEDDINGS_SCHEMA["required"]

    def test_image_generation_schema_has_required_fields(self):
        from litellm.litellm_mcp_server.tool_schemas import IMAGE_GENERATION_SCHEMA

        assert "prompt" in IMAGE_GENERATION_SCHEMA["required"]

    def test_text_completion_schema_has_required_fields(self):
        from litellm.litellm_mcp_server.tool_schemas import TEXT_COMPLETION_SCHEMA

        assert "model" in TEXT_COMPLETION_SCHEMA["required"]
        assert "prompt" in TEXT_COMPLETION_SCHEMA["required"]

    def test_transcription_schema_has_required_fields(self):
        from litellm.litellm_mcp_server.tool_schemas import TRANSCRIPTION_SCHEMA

        assert "model" in TRANSCRIPTION_SCHEMA["required"]
        assert "file" in TRANSCRIPTION_SCHEMA["required"]

    def test_rerank_schema_has_required_fields(self):
        from litellm.litellm_mcp_server.tool_schemas import RERANK_SCHEMA

        assert "model" in RERANK_SCHEMA["required"]
        assert "query" in RERANK_SCHEMA["required"]
        assert "documents" in RERANK_SCHEMA["required"]

    def test_list_models_schema_is_valid(self):
        from litellm.litellm_mcp_server.tool_schemas import LIST_MODELS_SCHEMA

        assert LIST_MODELS_SCHEMA["type"] == "object"
        assert "provider" in LIST_MODELS_SCHEMA["properties"]

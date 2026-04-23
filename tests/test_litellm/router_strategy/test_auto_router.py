import asyncio
import os
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.router_strategy.auto_router.auto_router import AutoRouter

pytestmark_skip_beta = pytest.mark.skip(
    reason="Skipping auto router tests - beta feature"
)


class TestExtractTextFromMessages:
    """Tests for AutoRouter._extract_text_from_messages (no semantic_router dependency)."""

    def test_should_extract_content_from_simple_user_message(self):
        messages = [{"role": "user", "content": "Hello world"}]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "Hello world"

    def test_should_extract_last_user_message_from_tool_call_conversation(self):
        messages = [
            {"role": "user", "content": "What's the weather in NYC?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "NYC"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc123",
                "content": "72°F and sunny",
            },
            {"role": "user", "content": "Now tell me about London"},
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "Now tell me about London"

    def test_should_find_user_message_when_last_message_is_assistant_with_tool_calls(
        self,
    ):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "What's the weather?"

    def test_should_find_user_message_when_last_message_is_tool_response(self):
        messages = [
            {"role": "user", "content": "What's the weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_abc",
                "content": "72°F and sunny",
            },
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "What's the weather?"

    def test_should_handle_multimodal_content_list(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            }
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "What's in this image?"

    def test_should_handle_multimodal_content_with_multiple_text_blocks(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "First part"},
                    {"type": "text", "text": "Second part"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/img.png"},
                    },
                ],
            }
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == "First part Second part"

    def test_should_return_empty_string_when_user_content_is_none(self):
        messages = [{"role": "user", "content": None}]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == ""

    def test_should_return_empty_string_when_no_user_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        ]
        result = AutoRouter._extract_text_from_messages(messages)
        assert result == ""

    def test_should_return_empty_string_for_empty_messages_list(self):
        result = AutoRouter._extract_text_from_messages([])
        assert result == ""


@pytest.fixture
def mock_router_instance():
    """Create a mock LiteLLM Router instance."""
    router = MagicMock()
    router.acompletion = AsyncMock()
    return router


@pytest.fixture
def mock_semantic_router():
    """Create a mock SemanticRouter instance."""
    mock_router = MagicMock()
    mock_route = MagicMock()
    mock_route.name = "test-route"
    mock_router.routes = [mock_route]
    return mock_router


@pytest.fixture
def mock_route_choice():
    """Create a mock RouteChoice instance."""
    mock_choice = MagicMock()
    mock_choice.name = "test-model"
    return mock_choice


@pytestmark_skip_beta
class TestAutoRouter:
    """Test class for AutoRouter methods."""

    @patch("semantic_router.routers.SemanticRouter")
    def test_init(self, mock_semantic_router_class, mock_router_instance):
        """Test that AutoRouter initializes correctly with all required parameters."""
        # Arrange
        mock_semantic_router_class.from_json.return_value = mock_semantic_router_class

        model_name = "test-auto-router"
        router_config_path = "test/path/router.json"
        default_model = "gpt-4o-mini"
        embedding_model = "text-embedding-model"

        # Act
        auto_router = AutoRouter(
            model_name=model_name,
            auto_router_config_path=router_config_path,
            default_model=default_model,
            embedding_model=embedding_model,
            litellm_router_instance=mock_router_instance,
        )

        # Assert
        assert auto_router.auto_router_config_path == router_config_path
        assert auto_router.auto_sync_value == AutoRouter.DEFAULT_AUTO_SYNC_VALUE
        assert auto_router.default_model == default_model
        assert auto_router.embedding_model == embedding_model
        assert auto_router.litellm_router_instance == mock_router_instance
        assert auto_router.routelayer is None
        mock_semantic_router_class.from_json.assert_called_once_with(router_config_path)

    @pytest.mark.asyncio
    @patch("semantic_router.routers.SemanticRouter")
    @patch("litellm.router_strategy.auto_router.litellm_encoder.LiteLLMRouterEncoder")
    async def test_async_pre_routing_hook_with_route_choice(
        self,
        mock_encoder_class,
        mock_semantic_router_class,
        mock_router_instance,
        mock_route_choice,
    ):
        """Test async_pre_routing_hook returns correct model when route is found."""
        # Arrange
        mock_loaded_router = MagicMock()
        mock_loaded_router.routes = ["route1", "route2"]
        mock_semantic_router_class.from_json.return_value = mock_loaded_router

        mock_routelayer = MagicMock()
        mock_routelayer.return_value = mock_route_choice
        mock_semantic_router_class.return_value = mock_routelayer

        auto_router = AutoRouter(
            model_name="test-auto-router",
            auto_router_config_path="test/path/router.json",
            default_model="gpt-4o-mini",
            embedding_model="text-embedding-model",
            litellm_router_instance=mock_router_instance,
        )

        messages = [{"role": "user", "content": "test message"}]

        # Act
        result = await auto_router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=messages
        )

        # Assert
        assert result is not None
        assert result.model == "test-model"  # Should use the route choice name
        assert result.messages == messages
        mock_routelayer.assert_called_once_with(text="test message")

    @pytest.mark.asyncio
    @patch("semantic_router.routers.SemanticRouter")
    @patch("litellm.router_strategy.auto_router.litellm_encoder.LiteLLMRouterEncoder")
    async def test_async_pre_routing_hook_with_list_route_choice(
        self,
        mock_encoder_class,
        mock_semantic_router_class,
        mock_router_instance,
        mock_route_choice,
    ):
        """Test async_pre_routing_hook handles list of RouteChoice objects correctly."""
        # Arrange
        mock_loaded_router = MagicMock()
        mock_loaded_router.routes = ["route1", "route2"]
        mock_semantic_router_class.from_json.return_value = mock_loaded_router

        mock_routelayer = MagicMock()
        mock_routelayer.return_value = [mock_route_choice]  # Return list
        mock_semantic_router_class.return_value = mock_routelayer

        auto_router = AutoRouter(
            model_name="test-auto-router",
            auto_router_config_path="test/path/router.json",
            default_model="gpt-4o-mini",
            embedding_model="text-embedding-model",
            litellm_router_instance=mock_router_instance,
        )

        messages = [{"role": "user", "content": "test message"}]

        # Act
        result = await auto_router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=messages
        )

        # Assert
        assert result is not None
        assert result.model == "test-model"
        assert result.messages == messages

    @pytest.mark.asyncio
    async def test_async_pre_routing_hook_no_messages(self, mock_router_instance):
        """Test async_pre_routing_hook returns None when no messages provided."""
        # Arrange
        with patch("semantic_router.routers.SemanticRouter"):
            auto_router = AutoRouter(
                model_name="test-auto-router",
                auto_router_config_path="test/path/router.json",
                default_model="gpt-4o-mini",
                embedding_model="text-embedding-model",
                litellm_router_instance=mock_router_instance,
            )

        # Act
        result = await auto_router.async_pre_routing_hook(
            model="test-model", request_kwargs={}, messages=None
        )

        # Assert
        assert result is None

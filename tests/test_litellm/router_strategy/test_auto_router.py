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

pytestmark = pytest.mark.skip(reason="Skipping auto router tests - beta feature")


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


class TestAutoRouter:
    """Test class for AutoRouter methods."""

    @patch('semantic_router.routers.SemanticRouter')
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
    @patch('semantic_router.routers.SemanticRouter')
    @patch('litellm.router_strategy.auto_router.litellm_encoder.LiteLLMRouterEncoder')
    async def test_async_pre_routing_hook_with_route_choice(
        self, 
        mock_encoder_class, 
        mock_semantic_router_class, 
        mock_router_instance,
        mock_route_choice
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
            model="test-model",
            request_kwargs={},
            messages=messages
        )
        
        # Assert
        assert result is not None
        assert result.model == "test-model"  # Should use the route choice name
        assert result.messages == messages
        mock_routelayer.assert_called_once_with(text="test message")

    @pytest.mark.asyncio
    @patch('semantic_router.routers.SemanticRouter')
    @patch('litellm.router_strategy.auto_router.litellm_encoder.LiteLLMRouterEncoder')
    async def test_async_pre_routing_hook_with_list_route_choice(
        self, 
        mock_encoder_class, 
        mock_semantic_router_class, 
        mock_router_instance,
        mock_route_choice
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
            model="test-model",
            request_kwargs={},
            messages=messages
        )
        
        # Assert
        assert result is not None
        assert result.model == "test-model"
        assert result.messages == messages

    @pytest.mark.asyncio
    async def test_async_pre_routing_hook_no_messages(self, mock_router_instance):
        """Test async_pre_routing_hook returns None when no messages provided."""
        # Arrange
        with patch('semantic_router.routers.SemanticRouter'):
            auto_router = AutoRouter(
                model_name="test-auto-router",
                auto_router_config_path="test/path/router.json",
                default_model="gpt-4o-mini",
                embedding_model="text-embedding-model",
                litellm_router_instance=mock_router_instance,
            )
        
        # Act
        result = await auto_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=None
        )
        
        # Assert
        assert result is None

    @pytest.mark.asyncio
    @patch("semantic_router.routers.SemanticRouter")
    @patch("litellm.router_strategy.auto_router.litellm_encoder.LiteLLMRouterEncoder")
    async def test_async_pre_routing_hook_with_anthropic_content_blocks(
        self,
        mock_encoder_class,
        mock_semantic_router_class,
        mock_router_instance,
        mock_route_choice,
    ):
        """Test that Anthropic-style list content blocks are joined into a plain string (line 127).

        When a message's content is a list of typed dicts (e.g. Anthropic format),
        only dict blocks are considered and their "text" values are joined with a space.
        Non-dict entries in the list must be skipped.
        """
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

        # Anthropic-style content blocks: list of dicts with a "type" and "text" key.
        # Also includes a non-dict entry to verify it is skipped by isinstance(block, dict).
        anthropic_content = [
            {"type": "text", "text": "What is the weather today?"},
            {"type": "text", "text": "In Paris."},
            {"type": "image_url", "image_url": {"url": "http://example.com/img.png"}},  # no "text" key
            "unexpected string block",  # non-dict — must be ignored
        ]
        messages = [{"role": "user", "content": anthropic_content}]

        # Act
        result = await auto_router.async_pre_routing_hook(
            model="test-model",
            request_kwargs={},
            messages=messages,
        )
        
        # Assert: only dict blocks contribute; "text" defaults to "" when missing
        mock_routelayer.assert_called_once_with(text="What is the weather today? In Paris. ")
        assert result is not None
        assert result.messages == messages

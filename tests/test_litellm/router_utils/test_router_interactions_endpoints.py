"""
Tests for Router interactions API endpoint initialization functions.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm import Router


class TestInitializeInteractionsEndpoints:
    """Test cases for _initialize_interactions_endpoints method"""

    def test_initialize_interactions_endpoints_creates_methods(self):
        """Test that _initialize_interactions_endpoints creates the expected interaction methods on the router."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "gpt-4"},
                }
            ]
        )

        # Verify the interaction methods are created
        assert hasattr(router, "acreate_interaction")
        assert hasattr(router, "create_interaction")
        assert hasattr(router, "aget_interaction")
        assert hasattr(router, "get_interaction")
        assert hasattr(router, "adelete_interaction")
        assert hasattr(router, "delete_interaction")
        assert hasattr(router, "acancel_interaction")
        assert hasattr(router, "cancel_interaction")

        # Verify they are callable
        assert callable(router.acreate_interaction)
        assert callable(router.create_interaction)
        assert callable(router.aget_interaction)
        assert callable(router.get_interaction)

    def test_initialize_interactions_endpoints_can_be_called_directly(self):
        """Test that _initialize_interactions_endpoints can be called directly to reinitialize endpoints."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "gpt-4"},
                }
            ]
        )

        # Call _initialize_interactions_endpoints directly
        router._initialize_interactions_endpoints()

        # Verify the interaction methods still exist after re-initialization
        assert hasattr(router, "acreate_interaction")
        assert hasattr(router, "create_interaction")
        assert callable(router.acreate_interaction)


class TestInitInteractionsApiEndpoints:
    """Test cases for _init_interactions_api_endpoints method"""

    @pytest.mark.asyncio
    async def test_init_interactions_api_endpoints_passes_custom_llm_provider(self):
        """Test that _init_interactions_api_endpoints passes custom_llm_provider to the original function."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "gpt-4"},
                }
            ]
        )

        mock_function = AsyncMock(return_value={"result": "success"})

        result = await router._init_interactions_api_endpoints(
            original_function=mock_function,
            custom_llm_provider="gemini",
            interaction_id="test-id",
        )

        mock_function.assert_called_once_with(
            custom_llm_provider="gemini",
            interaction_id="test-id",
        )
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_init_interactions_api_endpoints_defaults_to_gemini(self):
        """Test that _init_interactions_api_endpoints defaults to gemini when no custom_llm_provider is specified."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "gpt-4"},
                }
            ]
        )

        mock_function = AsyncMock(return_value={"result": "success"})

        result = await router._init_interactions_api_endpoints(
            original_function=mock_function,
            interaction_id="test-id",
        )

        mock_function.assert_called_once_with(
            custom_llm_provider="gemini",
            interaction_id="test-id",
        )
        assert result == {"result": "success"}

    @pytest.mark.asyncio
    async def test_init_interactions_api_endpoints_does_not_override_existing_provider(
        self,
    ):
        """Test that _init_interactions_api_endpoints does not override custom_llm_provider if already in kwargs."""
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4",
                    "litellm_params": {"model": "gpt-4"},
                }
            ]
        )

        mock_function = AsyncMock(return_value={"result": "success"})

        # Pass custom_llm_provider in kwargs directly (not as separate param)
        result = await router._init_interactions_api_endpoints(
            original_function=mock_function,
            custom_llm_provider="vertex_ai",
        )

        # Should use the provided custom_llm_provider
        mock_function.assert_called_once_with(
            custom_llm_provider="vertex_ai",
        )
        assert result == {"result": "success"}


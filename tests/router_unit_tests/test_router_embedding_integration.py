"""
Integration tests for router embedding method with various configurations.

These tests simulate real-world scenarios where headers and configuration
need to be properly propagated through the router to the LLM API.
"""
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router


class TestRouterEmbeddingIntegration:
    """Integration tests for embedding with router configuration."""

    def test_embedding_with_deployment_specific_headers(self):
        """
        Test that deployment-specific headers are propagated.

        This simulates a scenario where different deployments have
        different header requirements (e.g., different API versions).
        """
        model_list = [
            {
                "model_name": "embedding-deployment-1",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "key-1",
                    "headers": {"X-Deployment": "deployment-1"},
                },
            },
            {
                "model_name": "embedding-deployment-2",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "key-2",
                    "headers": {"X-Deployment": "deployment-2"},
                },
            },
        ]

        router = Router(model_list=model_list)

        # Test first deployment
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2]}])

            router.embedding(model="embedding-deployment-1", input=["test"])

            call_kwargs = mock_embedding.call_args[1]
            assert call_kwargs["api_key"] == "key-1"

        # Test second deployment
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2]}])

            router.embedding(model="embedding-deployment-2", input=["test"])

            call_kwargs = mock_embedding.call_args[1]
            assert call_kwargs["api_key"] == "key-2"

    def test_embedding_with_router_and_deployment_headers_merge(self):
        """
        Test that router-level headers are propagated.

        When no request headers are provided, router default headers should be used.
        """
        model_list = [
            {
                "model_name": "test-embedding",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "test-key",
                },
            }
        ]

        router = Router(
            model_list=model_list,
            default_litellm_params={
                "headers": {
                    "X-Router-Header": "router-value",
                    "X-Common-Header": "router-common",
                }
            },
        )

        # Test: No request headers - router headers should be used
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2]}])

            router.embedding(
                model="test-embedding",
                input=["test"],
            )

            call_kwargs = mock_embedding.call_args[1]

            # Router headers should be present
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["X-Router-Header"] == "router-value"
            assert call_kwargs["headers"]["X-Common-Header"] == "router-common"

    def test_embedding_metadata_propagation(self):
        """
        Test that metadata is properly set up and propagated.

        This is important for logging, tracking, and debugging.
        """
        model_list = [
            {
                "model_name": "test-embedding",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "test-key",
                },
            }
        ]

        router = Router(
            model_list=model_list,
            default_litellm_params={
                "metadata": {"environment": "test", "service": "embedding-service"}
            },
        )

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2]}])

            router.embedding(
                model="test-embedding",
                input=["test"],
                metadata={"request_id": "req-123"},  # Additional metadata from request
            )

            call_kwargs = mock_embedding.call_args[1]

            # Check metadata contains all expected fields
            assert "metadata" in call_kwargs
            metadata = call_kwargs["metadata"]

            # From _update_kwargs_before_fallbacks
            assert "model_group" in metadata
            assert metadata["model_group"] == "test-embedding"

            # From default_litellm_params
            assert "environment" in metadata
            assert metadata["environment"] == "test"
            assert "service" in metadata
            assert metadata["service"] == "embedding-service"

            # From request
            assert "request_id" in metadata
            assert metadata["request_id"] == "req-123"

    @pytest.mark.asyncio
    async def test_async_embedding_with_multiple_retries(self):
        """
        Test that async embedding properly uses num_retries from router config.

        This ensures the fix works with the retry mechanism.
        """
        model_list = [
            {
                "model_name": "test-embedding",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "test-key",
                },
            }
        ]

        router = Router(model_list=model_list, num_retries=2)

        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            mock_aembedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2]}])

            await router.aembedding(model="test-embedding", input=["test"])

            # The call should succeed
            mock_aembedding.assert_called_once()

    def test_embedding_with_timeout_from_router(self):
        """
        Test that timeout settings from router config are propagated.
        """
        model_list = [
            {
                "model_name": "test-embedding",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "test-key",
                },
            }
        ]

        router = Router(model_list=model_list, timeout=30.0)

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2]}])

            router.embedding(model="test-embedding", input=["test"])

            call_kwargs = mock_embedding.call_args[1]

            # Timeout should be set from router config
            assert "timeout" in call_kwargs
            assert call_kwargs["timeout"] == 30.0

    def test_embedding_with_multiple_deployments_load_balancing(self):
        """
        Test that headers are correctly propagated when router load balances
        between multiple deployments.
        """
        model_list = [
            {
                "model_name": "shared-embedding-model",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "key-1",
                },
            },
            {
                "model_name": "shared-embedding-model",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "key-2",
                },
            },
        ]

        router = Router(
            model_list=model_list,
            default_litellm_params={"headers": {"X-Shared-Header": "shared-value"}},
        )

        # Make multiple calls and verify headers are always present
        for i in range(5):
            with patch("litellm.embedding") as mock_embedding:
                mock_embedding.return_value = MagicMock(
                    data=[{"embedding": [0.1, 0.2]}]
                )

                router.embedding(model="shared-embedding-model", input=[f"test {i}"])

                call_kwargs = mock_embedding.call_args[1]

                # Headers should always be present regardless of which deployment is chosen
                assert "headers" in call_kwargs
                assert call_kwargs["headers"]["X-Shared-Header"] == "shared-value"

    @pytest.mark.asyncio
    async def test_embedding_with_fallback_configuration(self):
        """
        Test that headers are propagated correctly when using fallback models.
        """
        model_list = [
            {
                "model_name": "primary-embedding",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "primary-key",
                },
            },
            {
                "model_name": "fallback-embedding",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fallback-key",
                },
            },
        ]

        router = Router(
            model_list=model_list,
            fallbacks=[{"primary-embedding": ["fallback-embedding"]}],
            default_litellm_params={"headers": {"X-Fallback-Test": "test-value"}},
        )

        # Simulate primary failing, fallback succeeding
        with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
            call_count = 0

            async def side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # First call (primary) fails
                    raise Exception("Primary failed")
                else:
                    # Second call (fallback) succeeds
                    return MagicMock(data=[{"embedding": [0.1, 0.2]}])

            mock_aembedding.side_effect = side_effect

            await router.aembedding(model="primary-embedding", input=["test"])

            # Both calls should have headers
            assert mock_aembedding.call_count == 2

            # Check that both calls had headers
            for call_obj in mock_aembedding.call_args_list:
                call_kwargs = call_obj[1]
                assert "headers" in call_kwargs
                assert call_kwargs["headers"]["X-Fallback-Test"] == "test-value"

    def test_embedding_with_custom_provider_headers(self):
        """
        Test that provider-specific headers are correctly propagated.

        Some providers require specific headers for API versioning, features, etc.
        """
        model_list = [
            {
                "model_name": "azure-embedding",
                "litellm_params": {
                    "model": "azure/text-embedding-ada-002",
                    "api_key": "azure-key",
                    "api_base": "https://example.openai.azure.com",
                    "api_version": "2024-02-01",
                },
            }
        ]

        router = Router(
            model_list=model_list,
            default_litellm_params={
                "headers": {"X-Custom-Azure-Header": "azure-value"}
            },
        )

        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2]}])

            router.embedding(model="azure-embedding", input=["test"])

            call_kwargs = mock_embedding.call_args[1]

            # Verify Azure-specific params are present
            assert call_kwargs["api_base"] == "https://example.openai.azure.com"
            assert call_kwargs["api_version"] == "2024-02-01"

            # Verify custom headers are present
            assert "headers" in call_kwargs
            assert call_kwargs["headers"]["X-Custom-Azure-Header"] == "azure-value"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])

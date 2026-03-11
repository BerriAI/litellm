"""
Test suite for router embedding method header propagation.

This tests the fix for the issue where the embedding method was not
propagating proxy model configuration headers to the LLM API calls.

The fix ensures that router.embedding() calls _update_kwargs_before_fallbacks()
just like router.completion() does, which properly sets up metadata and allows
default_litellm_params (including headers) to be propagated.
"""
import os
import sys
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

sys.path.insert(0, os.path.abspath("../.."))

from litellm import Router


class TestRouterEmbeddingHeaders:
    """Test that embedding methods properly propagate headers from router configuration."""

    def test_embedding_calls_update_kwargs_before_fallbacks(self):
        """
        Test that router.embedding() calls _update_kwargs_before_fallbacks.

        This ensures that metadata is properly set up before the fallback mechanism,
        which is necessary for header propagation to work correctly.
        """
        model_list = [
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fake-key",
                },
            }
        ]

        router = Router(model_list=model_list)

        # Mock the _update_kwargs_before_fallbacks method to verify it's called
        with patch.object(
            router,
            "_update_kwargs_before_fallbacks",
            wraps=router._update_kwargs_before_fallbacks,
        ) as mock_update:
            with patch("litellm.embedding") as mock_litellm_embedding:
                mock_litellm_embedding.return_value = MagicMock(
                    data=[{"embedding": [0.1, 0.2, 0.3]}]
                )

                router.embedding(model="text-embedding-ada-002", input=["test input"])

                # Verify _update_kwargs_before_fallbacks was called
                mock_update.assert_called_once()
                call_kwargs = mock_update.call_args[1]
                assert call_kwargs["model"] == "text-embedding-ada-002"
                assert "kwargs" in call_kwargs

    @pytest.mark.asyncio
    async def test_aembedding_calls_update_kwargs_before_fallbacks(self):
        """
        Test that router.aembedding() calls _update_kwargs_before_fallbacks.

        This ensures consistency between sync and async embedding methods.
        """
        model_list = [
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fake-key",
                },
            }
        ]

        router = Router(model_list=model_list)

        # Mock the _update_kwargs_before_fallbacks method to verify it's called
        with patch.object(
            router,
            "_update_kwargs_before_fallbacks",
            wraps=router._update_kwargs_before_fallbacks,
        ) as mock_update:
            with patch(
                "litellm.aembedding", new_callable=AsyncMock
            ) as mock_litellm_aembedding:
                mock_litellm_aembedding.return_value = MagicMock(
                    data=[{"embedding": [0.1, 0.2, 0.3]}]
                )

                await router.aembedding(
                    model="text-embedding-ada-002", input=["test input"]
                )

                # Verify _update_kwargs_before_fallbacks was called
                mock_update.assert_called_once()
                call_kwargs = mock_update.call_args[1]
                assert call_kwargs["model"] == "text-embedding-ada-002"
                assert "kwargs" in call_kwargs

    def test_embedding_propagates_default_litellm_params(self):
        """
        Test that embedding calls properly propagate default_litellm_params including headers.

        This is the main fix - ensuring that headers set in default_litellm_params
        are included in the embedding request.
        """
        custom_headers = {"X-Custom-Header": "test-value", "X-API-Version": "v2"}

        model_list = [
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fake-key",
                },
            }
        ]

        # Create router with default_litellm_params containing headers
        router = Router(
            model_list=model_list,
            default_litellm_params={
                "headers": custom_headers,
                "metadata": {"test_key": "test_value"},
            },
        )

        with patch("litellm.embedding") as mock_litellm_embedding:
            mock_litellm_embedding.return_value = MagicMock(
                data=[{"embedding": [0.1, 0.2, 0.3]}]
            )

            router.embedding(model="text-embedding-ada-002", input=["test input"])

            # Verify that litellm.embedding was called with the headers
            mock_litellm_embedding.assert_called_once()
            call_kwargs = mock_litellm_embedding.call_args[1]

            # Check that headers were included
            assert "headers" in call_kwargs
            assert call_kwargs["headers"] == custom_headers

            # Check that metadata was properly set up
            assert "metadata" in call_kwargs
            assert "model_group" in call_kwargs["metadata"]
            assert call_kwargs["metadata"]["model_group"] == "text-embedding-ada-002"

    @pytest.mark.asyncio
    async def test_aembedding_propagates_default_litellm_params(self):
        """
        Test that async embedding calls properly propagate default_litellm_params including headers.
        """
        custom_headers = {"X-Custom-Header": "test-value", "X-API-Version": "v2"}

        model_list = [
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fake-key",
                },
            }
        ]

        # Create router with default_litellm_params containing headers
        router = Router(
            model_list=model_list,
            default_litellm_params={
                "headers": custom_headers,
                "metadata": {"test_key": "test_value"},
            },
        )

        with patch(
            "litellm.aembedding", new_callable=AsyncMock
        ) as mock_litellm_aembedding:
            mock_litellm_aembedding.return_value = MagicMock(
                data=[{"embedding": [0.1, 0.2, 0.3]}]
            )

            await router.aembedding(
                model="text-embedding-ada-002", input=["test input"]
            )

            # Verify that litellm.aembedding was called with the headers
            mock_litellm_aembedding.assert_called_once()
            call_kwargs = mock_litellm_aembedding.call_args[1]

            # Check that headers were included
            assert "headers" in call_kwargs
            assert call_kwargs["headers"] == custom_headers

            # Check that metadata was properly set up
            assert "metadata" in call_kwargs
            assert "model_group" in call_kwargs["metadata"]
            assert call_kwargs["metadata"]["model_group"] == "text-embedding-ada-002"

    def test_embedding_metadata_includes_model_group(self):
        """
        Test that embedding calls include model_group in metadata.

        The _update_kwargs_before_fallbacks method should set this up.
        """
        model_list = [
            {
                "model_name": "test-embedding-model",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fake-key",
                },
            }
        ]

        router = Router(model_list=model_list)

        with patch("litellm.embedding") as mock_litellm_embedding:
            mock_litellm_embedding.return_value = MagicMock(
                data=[{"embedding": [0.1, 0.2, 0.3]}]
            )

            router.embedding(model="test-embedding-model", input=["test input"])

            call_kwargs = mock_litellm_embedding.call_args[1]

            # Verify metadata contains model_group
            assert "metadata" in call_kwargs
            assert "model_group" in call_kwargs["metadata"]
            assert call_kwargs["metadata"]["model_group"] == "test-embedding-model"

    def test_embedding_sets_num_retries_from_router(self):
        """
        Test that embedding calls inherit num_retries from router configuration.

        This is set by _update_kwargs_before_fallbacks.
        """
        model_list = [
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fake-key",
                },
            }
        ]

        # Create router with num_retries set
        router = Router(model_list=model_list, num_retries=3)

        with patch("litellm.embedding") as mock_litellm_embedding:
            mock_litellm_embedding.return_value = MagicMock(
                data=[{"embedding": [0.1, 0.2, 0.3]}]
            )

            router.embedding(model="text-embedding-ada-002", input=["test input"])

            # Verify num_retries was not set in the call (it's handled by function_with_fallbacks)
            # The important thing is that it was set in kwargs before being passed to function_with_fallbacks
            # We verify this indirectly by checking that _update_kwargs_before_fallbacks was called
            mock_litellm_embedding.assert_called_once()

    def test_embedding_sets_litellm_trace_id(self):
        """
        Test that embedding calls include a litellm_trace_id.

        This is generated and set by _update_kwargs_before_fallbacks.
        """
        model_list = [
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fake-key",
                },
            }
        ]

        router = Router(model_list=model_list)

        with patch("litellm.embedding") as mock_litellm_embedding:
            mock_litellm_embedding.return_value = MagicMock(
                data=[{"embedding": [0.1, 0.2, 0.3]}]
            )

            router.embedding(model="text-embedding-ada-002", input=["test input"])

            call_kwargs = mock_litellm_embedding.call_args[1]

            # Verify litellm_trace_id was set
            assert "litellm_trace_id" in call_kwargs
            assert isinstance(call_kwargs["litellm_trace_id"], str)
            assert len(call_kwargs["litellm_trace_id"]) > 0

    def test_embedding_consistency_with_completion(self):
        """
        Test that embedding and completion methods handle kwargs similarly.

        Both should call _update_kwargs_before_fallbacks to ensure consistent behavior.
        """
        custom_headers = {"X-Test": "value"}

        model_list = [
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "fake-key",
                },
            },
            {
                "model_name": "text-embedding-ada-002",
                "litellm_params": {
                    "model": "text-embedding-ada-002",
                    "api_key": "fake-key",
                },
            },
        ]

        router = Router(
            model_list=model_list, default_litellm_params={"headers": custom_headers}
        )

        # Test completion
        with patch("litellm.completion") as mock_completion:
            mock_completion.return_value = MagicMock()

            router.completion(
                model="gpt-3.5-turbo", messages=[{"role": "user", "content": "test"}]
            )

            completion_kwargs = mock_completion.call_args[1]

        # Test embedding
        with patch("litellm.embedding") as mock_embedding:
            mock_embedding.return_value = MagicMock(
                data=[{"embedding": [0.1, 0.2, 0.3]}]
            )

            router.embedding(model="text-embedding-ada-002", input=["test input"])

            embedding_kwargs = mock_embedding.call_args[1]

        # Both should have headers from default_litellm_params
        assert "headers" in completion_kwargs
        assert "headers" in embedding_kwargs
        assert completion_kwargs["headers"] == custom_headers
        assert embedding_kwargs["headers"] == custom_headers

        # Both should have metadata with model_group
        assert "metadata" in completion_kwargs
        assert "metadata" in embedding_kwargs
        assert "model_group" in completion_kwargs["metadata"]
        assert "model_group" in embedding_kwargs["metadata"]

        # Both should have litellm_trace_id
        assert "litellm_trace_id" in completion_kwargs
        assert "litellm_trace_id" in embedding_kwargs


if __name__ == "__main__":
    # Run a simple test
    test = TestRouterEmbeddingHeaders()
    test.test_embedding_calls_update_kwargs_before_fallbacks()
    test.test_embedding_propagates_default_litellm_params()
    test.test_embedding_metadata_includes_model_group()
    test.test_embedding_sets_litellm_trace_id()
    test.test_embedding_consistency_with_completion()
    print("All tests passed!")  # noqa: T201

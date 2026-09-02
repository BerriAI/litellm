"""
Integration tests for router embedding method with various configurations.

These tests simulate real-world scenarios where headers and configuration
need to be properly propagated through the router to the LLM API.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.llms.base_llm.vector_store.transformation import (
    LiteLLMVectorStoreEmbeddingExecutor,
    RouterVectorStoreEmbeddingExecutor,
)

QUERY_VECTOR = [0.5, -0.25, 0.125]
OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"
STORE_EMBEDDINGS_URL = "https://embedding.example/v1/embeddings"


def _mock_embedding_route(respx_mock: respx.MockRouter, url: str) -> respx.Route:
    return respx_mock.post(url).mock(
        return_value=httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": QUERY_VECTOR}],
                "model": "text-embedding-3-small",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    )


def _sent(route: respx.Route, index: int) -> tuple[str, str, list[str]]:
    request = route.calls[index].request
    body = json.loads(request.read())
    return request.headers["authorization"], body["model"], body["input"]


def _alias_router() -> Router:
    return Router(
        model_list=[
            {
                "model_name": "team-alias",
                "litellm_params": {
                    "model": "openai/text-embedding-3-small",
                    "api_key": "deployment-key",
                },
            }
        ]
    )


class TestRouterEmbeddingIntegration:
    """Integration tests for embedding with router configuration."""

    def test_vector_store_request_metadata_prefers_litellm_metadata(self):
        assert Router._vector_store_request_metadata(
            {
                "litellm_metadata": {"user_api_key_team_id": "team-a"},
                "metadata": {"user_api_key_team_id": "team-b"},
            }
        ) == {"user_api_key_team_id": "team-a"}

        assert Router._vector_store_request_metadata({"metadata": {"user_api_key_team_id": "team-b"}}) == {
            "user_api_key_team_id": "team-b"
        }
        assert Router._vector_store_request_metadata({}) == {}

    def test_sync_vector_store_wrapper_injects_router_embedding_executor(self):
        router = Router(model_list=[])
        original = MagicMock(return_value="searched")
        wrapped = router.factory_function(original, call_type="vector_store_search")

        assert (
            wrapped(
                vector_store_id="store",
                query="query",
                custom_llm_provider="valkey",
                metadata={"user_api_key_team_id": "team-a"},
            )
            == "searched"
        )

        call_kwargs = original.call_args.kwargs
        assert call_kwargs["custom_llm_provider"] == "valkey"
        executor = call_kwargs["_direct_vector_store_embedding_executor"]
        assert isinstance(executor, RouterVectorStoreEmbeddingExecutor)
        assert executor.metadata == {"user_api_key_team_id": "team-a"}

    def test_sync_vector_store_wrapper_preserves_model_routing(self):
        router = Router(model_list=[])
        original = MagicMock()
        wrapped = router.factory_function(original, call_type="vector_store_search")

        with patch.object(router, "_generic_api_call_with_fallbacks", return_value="routed") as fallback:
            assert wrapped(model="vector-alias", vector_store_id="store", query="query") == "routed"

        assert fallback.call_args.kwargs["model"] == "vector-alias"
        assert fallback.call_args.kwargs["original_function"] is original
        assert isinstance(
            fallback.call_args.kwargs["_direct_vector_store_embedding_executor"],
            RouterVectorStoreEmbeddingExecutor,
        )

    @pytest.mark.asyncio
    async def test_vector_store_embedding_executors_cover_sdk_and_router_paths(
        self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        openai_route = _mock_embedding_route(respx_mock, OPENAI_EMBEDDINGS_URL)
        store_route = _mock_embedding_route(respx_mock, STORE_EMBEDDINGS_URL)
        sdk_executor = LiteLLMVectorStoreEmbeddingExecutor()

        sync_response = sdk_executor.embed("openai/text-embedding-3-small", "sync", {"api_key": "explicit"})
        async_response = await sdk_executor.aembed("openai/text-embedding-3-small", "async", {"api_key": "explicit"})

        assert sync_response.data[0]["embedding"] == QUERY_VECTOR
        assert async_response.data[0]["embedding"] == QUERY_VECTOR
        assert _sent(openai_route, 0) == ("Bearer explicit", "text-embedding-3-small", ["sync"])
        assert _sent(openai_route, 1) == ("Bearer explicit", "text-embedding-3-small", ["async"])

        explicit_config = {
            "api_base": "https://embedding.example/v1",
            "api_key": "store-key",
            "metadata": {
                "configured": True,
                "user_api_key_team_id": "untrusted-team",
            },
            "model": "untrusted-model",
        }
        mock_router = MagicMock()
        mock_router.embedding.return_value = sync_response
        router_executor = RouterVectorStoreEmbeddingExecutor(
            router=mock_router,
            metadata={"user_api_key_team_id": "team-a"},
        )
        assert router_executor.embed("team-alias", "query", explicit_config) is sync_response
        mock_router.embedding.assert_called_once_with(
            model="team-alias",
            input=["query"],
            api_base="https://embedding.example/v1",
            api_key="store-key",
            metadata={"configured": True, "user_api_key_team_id": "team-a"},
        )

        alias_executor = RouterVectorStoreEmbeddingExecutor(
            router=_alias_router(),
            metadata={"user_api_key_team_id": "team-a"},
        )
        sync_alias = alias_executor.embed("team-alias", "sync query", explicit_config)
        async_alias = await alias_executor.aembed("team-alias", "async query", explicit_config)

        assert sync_alias.data[0]["embedding"] == QUERY_VECTOR
        assert async_alias.data[0]["embedding"] == QUERY_VECTOR
        assert openai_route.call_count == 2
        assert _sent(store_route, 0) == ("Bearer store-key", "text-embedding-3-small", ["sync query"])
        assert _sent(store_route, 1) == ("Bearer store-key", "text-embedding-3-small", ["async query"])

    @pytest.mark.asyncio
    async def test_router_executor_falls_back_to_sdk_for_models_the_router_does_not_serve(
        self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        store_route = _mock_embedding_route(respx_mock, STORE_EMBEDDINGS_URL)
        executor = RouterVectorStoreEmbeddingExecutor(
            router=_alias_router(),
            metadata={"user_api_key_team_id": "team-a"},
        )
        inline_config = {"api_base": "https://embedding.example/v1", "api_key": "store-key"}

        sync_response = executor.embed("openai/text-embedding-3-large", "sync query", inline_config)
        async_response = await executor.aembed("openai/text-embedding-3-large", "async query", inline_config)

        assert sync_response.data[0]["embedding"] == QUERY_VECTOR
        assert async_response.data[0]["embedding"] == QUERY_VECTOR
        assert _sent(store_route, 0) == ("Bearer store-key", "text-embedding-3-large", ["sync query"])
        assert _sent(store_route, 1) == ("Bearer store-key", "text-embedding-3-large", ["async query"])

    def test_router_executor_routes_deployment_model_names_through_the_router(
        self, respx_mock: respx.MockRouter, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        openai_route = _mock_embedding_route(respx_mock, OPENAI_EMBEDDINGS_URL)
        executor = RouterVectorStoreEmbeddingExecutor(router=_alias_router(), metadata={})

        response = executor.embed("openai/text-embedding-3-small", "query", {})

        assert response.data[0]["embedding"] == QUERY_VECTOR
        assert _sent(openai_route, 0) == ("Bearer deployment-key", "text-embedding-3-small", ["query"])

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
                    "model": "text-embedding-3-small",
                    "api_key": "key-1",
                    "headers": {"X-Deployment": "deployment-1"},
                },
            },
            {
                "model_name": "embedding-deployment-2",
                "litellm_params": {
                    "model": "text-embedding-3-small",
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
                    "model": "text-embedding-3-small",
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
                    "model": "text-embedding-3-small",
                    "api_key": "test-key",
                },
            }
        ]

        router = Router(
            model_list=model_list,
            default_litellm_params={"metadata": {"environment": "test", "service": "embedding-service"}},
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
                    "model": "text-embedding-3-small",
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
                    "model": "text-embedding-3-small",
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
                    "model": "text-embedding-3-small",
                    "api_key": "key-1",
                },
            },
            {
                "model_name": "shared-embedding-model",
                "litellm_params": {
                    "model": "text-embedding-3-small",
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
                mock_embedding.return_value = MagicMock(data=[{"embedding": [0.1, 0.2]}])

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
                    "model": "text-embedding-3-small",
                    "api_key": "primary-key",
                },
            },
            {
                "model_name": "fallback-embedding",
                "litellm_params": {
                    "model": "text-embedding-3-small",
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
                    "model": "azure/text-embedding-3-small",
                    "api_key": "azure-key",
                    "api_base": "https://example.openai.azure.com",
                    "api_version": "2024-02-01",
                },
            }
        ]

        router = Router(
            model_list=model_list,
            default_litellm_params={"headers": {"X-Custom-Azure-Header": "azure-value"}},
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

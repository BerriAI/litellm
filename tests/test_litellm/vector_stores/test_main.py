"""
Tests for litellm/vector_stores/main.py.

Pins the router threading contract for vector store search: the router is an
explicit named parameter that reaches the HTTP handler, and it must never leak
into litellm_params/kwargs where logging would model_dump() it (the #19550
serialization trap).
"""

from unittest.mock import MagicMock, patch

import litellm.vector_stores.main as vector_stores_main
from litellm.vector_stores.main import search

MOCK_SEARCH_RESPONSE = {
    "object": "vector_store.search_results.page",
    "search_query": "q",
    "data": [],
}


def test_search_threads_router_to_handler():
    """search() must pass its router param through to the HTTP handler"""
    mock_router = MagicMock()
    logger = MagicMock()

    with (
        patch(
            "litellm.vector_stores.main.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=MagicMock(),
        ),
        patch.object(
            vector_stores_main.base_llm_http_handler,
            "vector_store_search_handler",
            return_value=MOCK_SEARCH_RESPONSE,
        ) as mock_handler,
    ):
        search(
            vector_store_id="bkt:idx",
            query="q",
            custom_llm_provider="s3_vectors",
            router=mock_router,
            litellm_logging_obj=logger,
        )

    mock_handler.assert_called_once()
    assert mock_handler.call_args.kwargs["router"] is mock_router


def test_search_router_not_in_litellm_params():
    """Regression (#19550 class): the router must stay out of GenericLiteLLMParams,
    otherwise pre-call logging model_dump()s it and breaks serialization."""
    mock_router = MagicMock()
    logger = MagicMock()

    with (
        patch(
            "litellm.vector_stores.main.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=MagicMock(),
        ),
        patch.object(
            vector_stores_main.base_llm_http_handler,
            "vector_store_search_handler",
            return_value=MOCK_SEARCH_RESPONSE,
        ) as mock_handler,
    ):
        search(
            vector_store_id="bkt:idx",
            query="q",
            custom_llm_provider="s3_vectors",
            router=mock_router,
            litellm_logging_obj=logger,
        )

    litellm_params = mock_handler.call_args.kwargs["litellm_params"]
    assert "router" not in litellm_params.model_dump(exclude_none=True)
    assert getattr(litellm_params, "router", None) is None

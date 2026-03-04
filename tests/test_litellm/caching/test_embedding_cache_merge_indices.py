from datetime import datetime
from unittest.mock import MagicMock

from litellm.caching.caching_handler import LLMCachingHandler


def test_embedding_cache_merge_preserves_global_indices_on_miss_merge():
    """
    Regression for #22659:
    when combining cached + non-cached embedding results, merged output indices
    must match original input positions (not local API result positions).
    """
    llm_caching_handler = LLMCachingHandler(
        original_function=MagicMock(),
        request_kwargs={},
        start_time=datetime.now(),
    )

    # First input is cache hit, second is cache miss
    cached_result = [
        {
            "embedding": [0.1, 0.2],
            "model": "text-embedding-3-small",
        },
        None,
    ]

    kwargs = {
        "model": "text-embedding-3-small",
        "input": ["cached-input", "new-input"],
    }

    mock_logging_obj = MagicMock()
    mock_logging_obj.async_success_handler = MagicMock()

    (
        final_embedding_cached_response,
        embedding_all_elements_cache_hit,
    ) = llm_caching_handler._process_async_embedding_cached_response(
        final_embedding_cached_response=None,
        cached_result=cached_result,
        kwargs=kwargs,
        logging_obj=mock_logging_obj,
        start_time=datetime.now(),
        model="text-embedding-3-small",
    )

    assert embedding_all_elements_cache_hit is False
    assert kwargs["input"] == ["new-input"]

    api_embedding_item = MagicMock()
    api_embedding_item.embedding = [0.3, 0.4]
    api_embedding_item.index = 0  # provider-local index before merge
    api_embedding_item.object = "embedding"

    mock_api_response = MagicMock()
    mock_api_response.model = "text-embedding-3-small"
    mock_api_response.usage = None
    mock_api_response.data = [api_embedding_item]

    caching_handler_response = MagicMock()
    caching_handler_response.final_embedding_cached_response = (
        final_embedding_cached_response
    )
    caching_handler_response.embedding_all_elements_cache_hit = (
        embedding_all_elements_cache_hit
    )

    response = llm_caching_handler._combine_cached_embedding_response_with_api_result(
        _caching_handler_response=caching_handler_response,
        embedding_response=mock_api_response,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert len(response.data) == 2
    assert response.data[0].index == 0
    assert response.data[1].index == 1

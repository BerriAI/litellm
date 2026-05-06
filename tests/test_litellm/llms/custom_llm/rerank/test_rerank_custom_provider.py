"""
Unit tests for custom_provider_map support in rerank() / arerank().

Regression test for: ValueError: 'X' is not a valid LlmProviders
when a provider registered via custom_provider_map is used with
mode: rerank (e.g. during health checks).
"""

import pytest

import litellm
from litellm import CustomLLM
from litellm.types.rerank import RerankResponse, RerankResponseResult


def test_rerank_custom_provider_dispatches_to_handler():
    """rerank() must route to the custom handler without crashing on the LlmProviders enum cast."""

    class MyRerankLLM(CustomLLM):
        def rerank(
            self,
            model,
            query,
            documents,
            top_n,
            rank_fields,
            return_documents,
            max_chunks_per_doc,
            logging_obj,
            optional_params,
            api_key=None,
            api_base=None,
            timeout=None,
            litellm_params=None,
        ):
            result: RerankResponseResult = {"index": 0, "relevance_score": 0.99}
            return RerankResponse(id="test-id", results=[result])

    handler = MyRerankLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_rerank_llm", "custom_handler": handler}
    ]

    resp = litellm.rerank(
        model="custom_rerank_llm/my-fake-model",
        query="What is the capital of France?",
        documents=["Paris is the capital.", "London is a city."],
        top_n=1,
    )

    assert isinstance(resp, RerankResponse)
    assert resp.id == "test-id"
    assert resp.results is not None
    assert resp.results[0]["relevance_score"] == 0.99


@pytest.mark.asyncio
async def test_arerank_custom_provider_dispatches_to_handler():
    """arerank() must route to the custom handler without crashing on the LlmProviders enum cast."""

    class MyRerankLLM(CustomLLM):
        async def arerank(
            self,
            model,
            query,
            documents,
            top_n,
            rank_fields,
            return_documents,
            max_chunks_per_doc,
            logging_obj,
            optional_params,
            api_key=None,
            api_base=None,
            timeout=None,
            litellm_params=None,
        ):
            result: RerankResponseResult = {"index": 0, "relevance_score": 0.88}
            return RerankResponse(id="async-test-id", results=[result])

    handler = MyRerankLLM()
    litellm.custom_provider_map = [
        {"provider": "custom_rerank_llm_async", "custom_handler": handler}
    ]

    resp = await litellm.arerank(
        model="custom_rerank_llm_async/my-fake-model",
        query="What is the capital of France?",
        documents=["Paris is the capital.", "London is a city."],
        top_n=1,
    )

    assert isinstance(resp, RerankResponse)
    assert resp.id == "async-test-id"
    assert resp.results is not None
    assert resp.results[0]["relevance_score"] == 0.88


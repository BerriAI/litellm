"""
Tests for the RAG query pipeline in litellm/rag/main.py.

The RAG pipeline forwards its kwargs (including the parent litellm_logging_obj)
into @client-decorated sub-calls (vector store search, completion). Each logging
object allows exactly one async_success event, so if sub-calls are not marked as
internal, the vector store search consumes the slot first and the LLM
completion's usage/cost is never logged (spend tracking and budget enforcement
are bypassed). These tests pin the invariant that the single billing event for
aquery carries the completion response with real usage and cost.
"""

import asyncio
from unittest.mock import patch

import pytest

import litellm
from litellm._internal_context import is_internal_call
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes, ModelResponse


class RecordingLogger(CustomLogger):
    def __init__(self):
        super().__init__()
        self.success_events = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.success_events.append({"kwargs": kwargs, "response_obj": response_obj})


@pytest.mark.asyncio
@pytest.mark.parametrize("use_router", [False, True])
async def test_aquery_single_billing_event_carries_completion_usage_and_cost(use_router):
    """
    litellm.aquery must produce exactly one success event, and that event must
    carry the LLM completion (a ModelResponse with non-zero usage and cost),
    not the vector store search response. The proxy always passes a router, so
    both the router and non-router completion branches are pinned.
    """
    recording_logger = RecordingLogger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [recording_logger]

    router_kwargs = {}
    if use_router:
        router_kwargs["router"] = litellm.Router(
            model_list=[
                {
                    "model_name": "gpt-4o-mini",
                    "litellm_params": {"model": "openai/gpt-4o-mini", "api_key": "test-key"},
                }
            ]
        )

    try:
        response = await litellm.aquery(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "What is the secret project codename?"}],
            retrieval_config={"vector_store_id": "vs_test_123", "custom_llm_provider": "openai"},
            mock_response="The secret project codename is AZURE-FALCON-42.",
            **router_kwargs,
        )

        assert isinstance(response, ModelResponse)
        assert is_internal_call.get() is False

        for _ in range(50):
            if recording_logger.success_events:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)
    finally:
        litellm.callbacks = original_callbacks

    assert len(recording_logger.success_events) == 1
    event = recording_logger.success_events[0]

    response_obj = event["response_obj"]
    assert isinstance(response_obj, ModelResponse)
    assert response_obj.usage.total_tokens > 0

    standard_logging_object = event["kwargs"]["standard_logging_object"]
    assert standard_logging_object["call_type"] == "aquery"
    assert standard_logging_object["total_tokens"] > 0
    assert standard_logging_object["prompt_tokens"] > 0
    assert standard_logging_object["completion_tokens"] > 0
    assert standard_logging_object["response_cost"] > 0


@pytest.mark.asyncio
async def test_aquery_response_hidden_params_carry_completion_cost():
    """
    The aquery response must expose the completion's response_cost via hidden
    params, so the proxy can return the x-litellm-response-cost header.
    """
    response = await litellm.aquery(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hello"}],
        retrieval_config={"vector_store_id": "vs_test_123", "custom_llm_provider": "openai"},
        mock_response="hi there",
    )

    assert isinstance(response, ModelResponse)
    response_cost = response._hidden_params.get("response_cost")
    assert response_cost is not None
    assert response_cost > 0


@pytest.mark.asyncio
async def test_aquery_billed_cost_includes_priced_vector_store_search():
    """
    When the vector store provider prices search calls (e.g. per-query cost),
    that cost must be folded into the aquery billing instead of being dropped
    with the suppressed sub-call event.
    """
    recording_logger = RecordingLogger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [recording_logger]

    try:
        with patch("litellm.rag.main.vector_store_search_cost", return_value=(0.002, 0.0)):
            response = await litellm.aquery(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hello"}],
                retrieval_config={"vector_store_id": "vs_test_123", "custom_llm_provider": "openai"},
                mock_response="hi there",
            )

        for _ in range(50):
            if recording_logger.success_events:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)
    finally:
        litellm.callbacks = original_callbacks

    assert isinstance(response, ModelResponse)
    total_cost = response._hidden_params.get("response_cost")
    assert total_cost is not None
    assert total_cost > 0.002

    assert len(recording_logger.success_events) == 1
    standard_logging_object = recording_logger.success_events[0]["kwargs"]["standard_logging_object"]
    assert standard_logging_object["response_cost"] == total_cost


@pytest.mark.asyncio
async def test_aquery_with_rerank_bills_once_and_folds_rerank_cost():
    """
    When rerank is enabled, its sub-call must run under the internal-call
    context (no standalone billing event) and its cost must be folded into
    the single aquery billing event.
    """
    from litellm.types.rerank import RerankResponse

    recording_logger = RecordingLogger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [recording_logger]
    rerank_seen = {}

    async def fake_arerank(**kwargs):
        rerank_seen["internal"] = is_internal_call.get()
        rerank_result = RerankResponse(id="rr_1", results=[{"index": 0, "relevance_score": 0.9}], meta={})
        rerank_result._hidden_params["response_cost"] = 0.001
        return rerank_result

    try:
        with patch("litellm.arerank", side_effect=fake_arerank):
            response = await litellm.aquery(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hello"}],
                retrieval_config={"vector_store_id": "vs_test_123", "custom_llm_provider": "openai"},
                rerank={"enabled": True, "model": "cohere/rerank-english-v3.0", "top_n": 1},
                mock_response="hi there",
            )

        for _ in range(50):
            if recording_logger.success_events:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)
    finally:
        litellm.callbacks = original_callbacks

    assert rerank_seen["internal"] is True
    assert is_internal_call.get() is False

    assert isinstance(response, ModelResponse)
    total_cost = response._hidden_params.get("response_cost")
    assert total_cost is not None
    assert total_cost > 0.001

    assert len(recording_logger.success_events) == 1
    standard_logging_object = recording_logger.success_events[0]["kwargs"]["standard_logging_object"]
    assert standard_logging_object["call_type"] == "aquery"
    assert standard_logging_object["response_cost"] == total_cost


@pytest.mark.asyncio
async def test_aquery_streaming_bills_sub_call_costs_into_final_event():
    """
    On the streaming path the response cost is computed from the assembled
    chunks after the pipeline returns, so there is no response object to fold
    sub-call costs into. The pipeline must instead carry the accumulated
    search and rerank cost through the logging object so the single streamed
    billing event includes it; otherwise a caller passing stream=true incurs
    priced vector search and rerank costs that never reach spend tracking.
    """
    from litellm.types.rerank import RerankResponse

    recording_logger = RecordingLogger()
    original_callbacks = litellm.callbacks
    litellm.callbacks = [recording_logger]
    rerank_seen = {}

    async def fake_arerank(**kwargs):
        rerank_seen["internal"] = is_internal_call.get()
        rerank_result = RerankResponse(id="rr_1", results=[{"index": 0, "relevance_score": 0.9}], meta={})
        rerank_result._hidden_params["response_cost"] = 0.001
        return rerank_result

    try:
        with (
            patch("litellm.rag.main.vector_store_search_cost", return_value=(0.002, 0.0)),
            patch("litellm.arerank", side_effect=fake_arerank),
        ):
            response = await litellm.aquery(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "hello"}],
                retrieval_config={"vector_store_id": "vs_test_123", "custom_llm_provider": "openai"},
                rerank={"enabled": True, "model": "cohere/rerank-english-v3.0", "top_n": 1},
                mock_response="hi there",
                stream=True,
            )
            async for _ in response:
                pass

        for _ in range(50):
            if recording_logger.success_events:
                break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.5)
    finally:
        litellm.callbacks = original_callbacks

    assert rerank_seen["internal"] is True
    assert is_internal_call.get() is False

    assert len(recording_logger.success_events) == 1
    standard_logging_object = recording_logger.success_events[0]["kwargs"]["standard_logging_object"]
    assert standard_logging_object["call_type"] == "aquery"
    assert standard_logging_object["response_cost"] >= 0.003


def test_rag_call_types_are_registered():
    """
    query/aquery/ingest/aingest are @client-decorated entry points, so their
    function names must resolve to CallTypes members (deployment hooks and
    call-type driven logic silently no-op for unregistered call types).
    """
    assert CallTypes("query") is CallTypes.query
    assert CallTypes("aquery") is CallTypes.aquery
    assert CallTypes("ingest") is CallTypes.ingest
    assert CallTypes("aingest") is CallTypes.aingest

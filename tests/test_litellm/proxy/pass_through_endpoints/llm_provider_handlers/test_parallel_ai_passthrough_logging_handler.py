from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.llms.parallel_ai.extract.cost_calculator import PARALLEL_AI_EXTRACT_MODEL
from litellm.proxy.pass_through_endpoints.llm_provider_handlers.parallel_ai_passthrough_logging_handler import (
    ParallelAIPassthroughLoggingHandler,
)
from litellm.proxy.pass_through_endpoints.success_handler import PassThroughEndpointLogging


def test_extract_route_detection_requires_parallel_provider() -> None:
    assert ParallelAIPassthroughLoggingHandler.is_extract_route(
        "https://api.parallel.ai/v1/extract",
        "parallel_ai",
    )
    assert not ParallelAIPassthroughLoggingHandler.is_extract_route(
        "https://api.parallel.ai/v1/extract",
        None,
    )
    assert not ParallelAIPassthroughLoggingHandler.is_extract_route(
        "https://api.parallel.ai/v1/search",
        "parallel_ai",
    )


def test_extract_handler_sets_usage_aware_cost_and_model() -> None:
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "parallel-extract-call"
    logging_obj.model_call_details = {}
    response_body = {
        "extract_id": "extract_test",
        "results": [],
        "errors": [],
        "usage": [{"name": "sku_extract_excerpts", "count": 2}],
        "session_id": "session_test",
    }

    result = ParallelAIPassthroughLoggingHandler.parallel_ai_extract_handler(
        response_body=response_body,
        logging_obj=logging_obj,
        request_body={"urls": ["https://example.com/1", "https://example.com/2"]},
    )

    assert result["kwargs"]["model"] == PARALLEL_AI_EXTRACT_MODEL
    assert result["kwargs"]["custom_llm_provider"] == "parallel_ai"
    assert result["kwargs"]["response_cost"] == 0.002
    assert logging_obj.model_call_details["model"] == PARALLEL_AI_EXTRACT_MODEL
    assert logging_obj.model_call_details["custom_llm_provider"] == "parallel_ai"
    assert logging_obj.model_call_details["response_cost"] == 0.002


def test_success_handler_dispatches_parallel_extract_billing() -> None:
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "parallel-extract-dispatch"
    logging_obj.model_call_details = {}
    response_body = {
        "extract_id": "extract_dispatch",
        "results": [],
        "errors": [],
        "usage": [{"name": "sku_extract_excerpts", "count": 1}],
        "session_id": "session_dispatch",
    }
    response = httpx.Response(
        200,
        json=response_body,
        request=httpx.Request("POST", "https://api.parallel.ai/v1/extract"),
    )

    normalized = PassThroughEndpointLogging().normalize_llm_passthrough_logging_payload(
        httpx_response=response,
        response_body=response_body,
        request_body={"urls": ["https://example.com"]},
        logging_obj=logging_obj,
        url_route="https://api.parallel.ai/v1/extract",
        result="",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        custom_llm_provider="parallel_ai",
    )

    assert normalized["standard_logging_response_object"] is not None
    assert normalized["kwargs"]["model"] == PARALLEL_AI_EXTRACT_MODEL
    assert normalized["kwargs"]["response_cost"] == 0.001


@pytest.mark.asyncio
async def test_success_handler_sends_extract_cost_to_async_loggers() -> None:
    logging_obj = MagicMock()
    logging_obj.litellm_call_id = "parallel-extract-logging"
    logging_obj.model_call_details = {}
    logging_obj.dispatch_success_handlers = AsyncMock()
    response_body = {
        "extract_id": "extract_logging",
        "results": [],
        "errors": [],
        "usage": [{"name": "sku_extract_excerpts", "count": 2}],
        "session_id": "session_logging",
    }
    response = httpx.Response(
        200,
        json=response_body,
        request=httpx.Request("POST", "https://api.parallel.ai/v1/extract"),
    )

    await PassThroughEndpointLogging().pass_through_async_success_handler(
        httpx_response=response,
        response_body=response_body,
        logging_obj=logging_obj,
        url_route="https://api.parallel.ai/v1/extract",
        result="",
        start_time=datetime.now(),
        end_time=datetime.now(),
        cache_hit=False,
        request_body={"urls": ["https://example.com/1", "https://example.com/2"]},
        passthrough_logging_payload={
            "url": "https://api.parallel.ai/v1/extract",
            "request_body": {"urls": ["https://example.com/1", "https://example.com/2"]},
            "request_method": "POST",
            "cost_per_request": None,
        },
        custom_llm_provider="parallel_ai",
    )

    logging_obj.dispatch_success_handlers.assert_awaited_once()
    dispatched_kwargs = logging_obj.dispatch_success_handlers.await_args.kwargs
    assert dispatched_kwargs["model"] == PARALLEL_AI_EXTRACT_MODEL
    assert dispatched_kwargs["custom_llm_provider"] == "parallel_ai"
    assert dispatched_kwargs["response_cost"] == 0.002
    assert dispatched_kwargs["prefer_async_handlers"] is True

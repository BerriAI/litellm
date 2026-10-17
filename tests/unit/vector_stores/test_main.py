"""
Tests for litellm/vector_stores/main.py.

Pins the router threading contract for vector store search: the router is an
explicit named parameter that reaches the HTTP handler wrapped in the embedding
executor, and it must never leak into litellm_params/kwargs where logging would
model_dump() it (the #19550 serialization trap).
"""

import json
from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm.vector_stores.main as vector_stores_main
from litellm.llms.base_llm.vector_store.transformation import (
    RouterVectorStoreEmbeddingExecutor,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.vector_stores.main import acreate, aupdate, search

MOCK_SEARCH_RESPONSE = {
    "object": "vector_store.search_results.page",
    "search_query": "q",
    "data": [],
}


@pytest.mark.parametrize("query", ["q", ["q", "another question"]])
def test_search_wraps_router_into_the_handler_embedding_executor(query: str | list[str]):
    """search() hands the HTTP handler a Router-backed embedding executor carrying the
    request metadata, and no bare router kwarg (LIT-6750)"""
    mock_router = MagicMock()
    logger = MagicMock()

    with (
        patch(  # test-quality-ok: stubs provider config resolution; the seam under test is the executor threading
            "litellm.vector_stores.main.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=MagicMock(),
        ),
        patch.object(  # test-quality-ok: the handler call is the observable boundary for the executor contract
            vector_stores_main.base_llm_http_handler,
            "vector_store_search_handler",
            return_value=MOCK_SEARCH_RESPONSE,
        ) as mock_handler,
    ):
        response = search(
            vector_store_id="bkt:idx",
            query=query,
            custom_llm_provider="s3_vectors",
            router=mock_router,
            litellm_logging_obj=logger,
            litellm_metadata={"user_api_key_team_id": "team-a"},
        )

    assert response == MOCK_SEARCH_RESPONSE
    mock_handler.assert_called_once()
    assert "router" not in mock_handler.call_args.kwargs
    assert mock_handler.call_args.kwargs["query"] == query
    executor = mock_handler.call_args.kwargs["embedding_executor"]
    assert isinstance(executor, RouterVectorStoreEmbeddingExecutor)
    assert executor.router is mock_router
    assert dict(executor.metadata) == {"user_api_key_team_id": "team-a"}


def test_search_router_not_in_litellm_params():
    """Regression (#19550 class): the router must stay out of GenericLiteLLMParams,
    otherwise pre-call logging model_dump()s it and breaks serialization."""
    mock_router = MagicMock()
    logger = MagicMock()

    with (
        patch(  # test-quality-ok: stubs provider config resolution; the seam under test is litellm_params contents
            "litellm.vector_stores.main.ProviderConfigManager.get_provider_vector_stores_config",
            return_value=MagicMock(),
        ),
        patch.object(  # test-quality-ok: the handler call is where a leaked router in litellm_params would surface
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


def test_search_forwards_top_level_user_context_to_bedrock_retrieve():
    """Regression (LIT-4415): a top-level userContext, the shape the OpenAI SDK's extra_body
    produces on the proxy path, reaches the Bedrock Retrieve request body."""
    client = MagicMock(spec=HTTPHandler)
    client.post.return_value = MagicMock(status_code=200, json=MagicMock(return_value={"retrievalResults": []}))

    search(
        vector_store_id="kb123",
        query="q",
        custom_llm_provider="bedrock",
        aws_region_name="us-west-2",
        aws_access_key_id="test-key-id",
        aws_secret_access_key="test-secret-key",
        userContext={"userId": "alice@example.com"},
        client=client,
        litellm_logging_obj=MagicMock(),
    )

    posted = json.loads(client.post.call_args.kwargs["data"])
    assert posted["userContext"] == {"userId": "alice@example.com"}
    assert posted["retrievalQuery"] == {"text": "q"}


PROXY_STAMPED_METADATA: Final = {
    "team": "llmproxy",
    "user_api_key": "hashed-key",
    "user_api_key_hash": "hashed-key",
    "user_api_key_team_id": "team-1",
    "requester_ip_address": "127.0.0.1",
    "user_agent": "curl/8.7.1",
    "endpoint": "http://localhost:4000/v1/vector_stores",
    "litellm_api_version": "1.0.0",
    "deployment": "openai/gpt-5",
    "deployment_model_name": "gpt-5",
    "api_base": "https://api.openai.com/v1",
    "agent_id": "agent-1",
    "actor_agent_id": "agent-2",
    "target_agent_id": "agent-3",
    "billing_agent_id": "agent-4",
    "agent_execution_mode": "autonomous",
    "verified_human_user_id": "user-1",
    "litellm_caller_session_id": "session-1",
    "user_api_keyboard": "qwerty",
    1: "a",
}


def _copy(metadata: Mapping[object, object] | None) -> dict[object, object] | None:
    return None if metadata is None else dict(metadata)


async def _metadata_sent_to_provider(
    operation: str, metadata: Mapping[object, object] | None, **kwargs: object
) -> list[object]:
    sent_metadata: Final[list[object]] = []

    def echo_vector_store(request: httpx.Request) -> httpx.Response:
        body: Final = json.loads(request.content)
        sent_metadata.append(body.get("metadata"))
        return httpx.Response(
            200,
            json={
                "id": "vs_1",
                "object": "vector_store",
                "created_at": 0,
                "name": "x",
                "usage_bytes": 0,
                "file_counts": {"in_progress": 0, "completed": 0, "failed": 0, "cancelled": 0, "total": 0},
                "status": "completed",
                "metadata": body.get("metadata"),
            },
        )

    client: Final = AsyncHTTPHandler(transport=httpx.MockTransport(echo_vector_store))
    if operation == "create":
        await acreate(name="x", metadata=_copy(metadata), api_key="sk-test", client=client, **kwargs)
    else:
        await aupdate(vector_store_id="vs_1", metadata=_copy(metadata), api_key="sk-test", client=client, **kwargs)
    return sent_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
async def test_proxy_vector_store_writes_send_only_the_callers_metadata_to_the_provider(operation: str):
    sent_metadata: Final = await _metadata_sent_to_provider(
        operation,
        PROXY_STAMPED_METADATA,
        proxy_server_request={"url": "http://localhost:4000/v1/vector_stores", "method": "POST", "body": {}},
    )

    assert sent_metadata == [{"team": "llmproxy", "user_api_keyboard": "qwerty", "1": "a"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["create", "update"])
@pytest.mark.parametrize(
    "callers_metadata", [{"team": "llmproxy", "endpoint": "ingest", "deployment": "prod", "agent_id": "bot-7"}, None]
)
async def test_direct_sdk_vector_store_writes_send_the_callers_metadata_unchanged(
    operation: str, callers_metadata: Mapping[object, object] | None
):
    sent_metadata: Final = await _metadata_sent_to_provider(operation, callers_metadata)

    assert sent_metadata == [callers_metadata]

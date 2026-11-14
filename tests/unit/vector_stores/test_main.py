"""
Tests for litellm/vector_stores/main.py.

Pins the router threading contract for vector store search: the router is an
explicit named parameter that reaches the HTTP handler wrapped in the embedding
executor, and it must never leak into litellm_params/kwargs where logging would
model_dump() it (the #19550 serialization trap).
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Final
from unittest.mock import MagicMock, patch

import httpx
import pytest
import requests
import requests_mock as requests_mock_lib
import responses
import respx

import litellm
import litellm.vector_stores.main as vector_stores_main
from litellm.llms.base_llm.vector_store.transformation import (
    RouterVectorStoreEmbeddingExecutor,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.vector_stores.main import acreate, aretrieve, aupdate, retrieve, search

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


QUERY_IMAGE: Final = {"type": "base64", "media_type": "image/jpeg", "data": "/9j/4AAQSkZJRg=="}


def _search_with_recorded_post(**search_kwargs: object) -> MagicMock:
    client: Final = MagicMock(spec=HTTPHandler)
    client.post.return_value = MagicMock(
        status_code=200,
        json=MagicMock(
            return_value=(
                {"object": "vector_store.search_results.page", "search_query": "", "data": []}
                if search_kwargs.get("custom_llm_provider") == "openai"
                else {"retrievalResults": []}
            )
        ),
    )
    credentials: Final = (
        {"api_key": "sk-test"}
        if search_kwargs.get("custom_llm_provider") == "openai"
        else {"aws_region_name": "us-west-2", "aws_access_key_id": "test-key-id", "aws_secret_access_key": "secret"}
    )
    search(vector_store_id="kb123", client=client, litellm_logging_obj=MagicMock(), **credentials, **search_kwargs)
    return client


@pytest.mark.parametrize("query", [None, "", "   ", ["", " "]])
def test_search_by_query_image_sends_a_bedrock_image_retrieval_query(query: str | list[str] | None):
    query_kwargs: Final = {} if query is None else {"query": query}

    client: Final = _search_with_recorded_post(custom_llm_provider="bedrock", query_image=QUERY_IMAGE, **query_kwargs)

    posted: Final = json.loads(client.post.call_args.kwargs["data"])
    assert posted["retrievalQuery"] == {
        "type": "IMAGE",
        "image": {"format": "jpeg", "inlineContent": "/9j/4AAQSkZJRg=="},
    }


@pytest.mark.parametrize(
    ("search_kwargs", "message"),
    [
        (
            {"custom_llm_provider": "bedrock", "query": "red circle", "query_image": QUERY_IMAGE},
            "Send either query or query_image, not both.",
        ),
        ({"custom_llm_provider": "bedrock"}, "query or query_image is required."),
        (
            {"custom_llm_provider": "bedrock", "query": 5, "query_image": QUERY_IMAGE},
            "Send either query or query_image, not both.",
        ),
        (
            {"custom_llm_provider": "bedrock", "query": [1, 2], "query_image": QUERY_IMAGE},
            "Send either query or query_image, not both.",
        ),
        (
            {"custom_llm_provider": "openai", "query_image": QUERY_IMAGE},
            "openai vector stores do not support query_image. To drop it",
        ),
        (
            {"custom_llm_provider": "openai", "query": "red circle", "query_image": QUERY_IMAGE},
            "openai vector stores do not support query_image. To drop it",
        ),
        (
            {"custom_llm_provider": "openai", "query": "", "query_image": QUERY_IMAGE, "drop_params": True},
            "drop_params dropped it",
        ),
    ],
)
def test_search_rejects_an_unusable_query_with_a_bad_request(search_kwargs: dict[str, object], message: str):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        _search_with_recorded_post(**search_kwargs)

    assert exc_info.value.status_code == 400
    assert message in str(exc_info.value)


def test_search_rejects_an_invalid_query_image_without_echoing_its_data():
    image_data: Final = "c2VjcmV0LWltYWdlLWJ5dGVz"

    with pytest.raises(litellm.BadRequestError) as exc_info:
        _search_with_recorded_post(
            custom_llm_provider="bedrock",
            query_image={"type": "base64", "media_type": "image/tiff", "data": image_data},
        )

    assert exc_info.value.status_code == 400
    assert "query_image.media_type" in str(exc_info.value)
    assert image_data not in str(exc_info.value)


@pytest.mark.parametrize("data", ["not base64!", "/9j/4AAQ SkZJRg==", "/9j/4AAQ\tSkZJRg=="])
def test_search_rejects_query_image_data_that_is_not_base64(data: str):
    with pytest.raises(litellm.BadRequestError) as exc_info:
        _search_with_recorded_post(
            custom_llm_provider="bedrock",
            query_image={"type": "base64", "media_type": "image/jpeg", "data": data},
        )

    assert exc_info.value.status_code == 400
    assert "query_image.data: not valid base64" in str(exc_info.value)
    assert data not in str(exc_info.value)


@pytest.mark.parametrize("line_break", ["\n", "\r\n"])
def test_search_sends_line_wrapped_query_image_data_to_bedrock_as_sent(line_break: str):
    data: Final = line_break.join(("/9j/4AAQ", "SkZJRg=="))

    client: Final = _search_with_recorded_post(
        custom_llm_provider="bedrock",
        query_image={"type": "base64", "media_type": "image/jpeg", "data": data},
    )

    posted: Final = json.loads(client.post.call_args.kwargs["data"])
    assert posted["retrievalQuery"]["image"]["inlineContent"] == data


def test_search_by_a_query_image_sent_in_extra_body_sends_a_bedrock_image_retrieval_query():
    client: Final = _search_with_recorded_post(custom_llm_provider="bedrock", extra_body={"query_image": QUERY_IMAGE})

    posted: Final = json.loads(client.post.call_args.kwargs["data"])
    assert posted["retrievalQuery"] == {
        "type": "IMAGE",
        "image": {"format": "jpeg", "inlineContent": "/9j/4AAQSkZJRg=="},
    }


@pytest.mark.parametrize("drop_params_source", ["request", "global"])
def test_search_drops_query_image_for_a_provider_without_image_search_when_drop_params_is_on(
    drop_params_source: str, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(litellm, "drop_params", drop_params_source == "global")
    drop_kwargs: Final = {"drop_params": True} if drop_params_source == "request" else {}

    client: Final = _search_with_recorded_post(
        custom_llm_provider="openai", query="red circle", query_image=QUERY_IMAGE, **drop_kwargs
    )

    posted: Final = json.loads(client.post.call_args.kwargs["data"])
    assert posted["query"] == "red circle"
    assert "query_image" not in posted


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


def _vector_store_not_found(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        404,
        json={
            "error": {
                "message": "No vector store found with id 'vs_missing'.",
                "type": "invalid_request_error",
                "param": "vector_store_id",
                "code": None,
            }
        },
        request=request,
    )


async def _retrieve_missing_openai_vector_store(is_async: bool) -> object:
    transport: Final = httpx.MockTransport(_vector_store_not_found)
    if is_async:
        return await aretrieve(
            vector_store_id="vs_missing",
            custom_llm_provider="openai",
            api_key="sk-test",
            client=AsyncHTTPHandler(transport=transport),
        )
    return retrieve(
        vector_store_id="vs_missing",
        custom_llm_provider="openai",
        api_key="sk-test",
        client=HTTPHandler(client=httpx.Client(transport=transport)),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [True, False])
async def test_retrieving_a_missing_openai_vector_store_raises_not_found(is_async: bool):
    with pytest.raises(litellm.NotFoundError) as caught:
        await _retrieve_missing_openai_vector_store(is_async)

    assert caught.value.status_code == 404
    assert "No vector store found with id 'vs_missing'." in caught.value.message


GOOGLE_TOKEN_URL: Final = "https://oauth2.googleapis.com/token"


def _authorized_user_credentials(refresh_token: str) -> str:
    return json.dumps(
        {
            "type": "authorized_user",
            "client_id": "unit-client-id",
            "client_secret": "unit-client-secret",
            "refresh_token": refresh_token,
        }
    )


def _search_rag_store(api_base: str, vertex_credentials: str):
    return litellm.vector_stores.asearch(
        query="what is LiteLLM?",
        vector_store_id="unit-store",
        custom_llm_provider="vertex_ai/rag_api",
        api_base=api_base,
        vertex_project="unit-project",
        vertex_location="us-central1",
        vertex_credentials=vertex_credentials,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_type", "search_method", "search_response"),
    (
        ("search_api", ":search", {"results": []}),
        ("rag_api", ":retrieveContexts", {"contexts": {"contexts": []}}),
    ),
)
async def test_concurrent_vertex_searches_with_the_same_credentials_fetch_one_google_token(
    api_type: str,
    search_method: str,
    search_response: dict[str, object],
    respx_mock: respx.MockRouter,
    requests_mock: requests_mock_lib.Mocker,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    credentials: Final = _authorized_user_credentials(f"unit-refresh-{uuid.uuid4().hex}")

    def slow_token_response(
        request: requests_mock_lib.Request, context: requests_mock_lib.Context
    ) -> dict[str, object]:
        time.sleep(0.2)
        return {"access_token": "ya29.unit-token", "expires_in": 3600, "token_type": "Bearer"}

    token_route: Final = requests_mock.post(GOOGLE_TOKEN_URL, json=slow_token_response)
    search_route: Final = respx_mock.post(f"https://vertex.test/{api_type}{search_method}").mock(
        return_value=httpx.Response(200, json=search_response)
    )

    await asyncio.gather(
        *(
            litellm.vector_stores.asearch(
                query="what is LiteLLM?",
                vector_store_id="unit-store",
                custom_llm_provider=f"vertex_ai/{api_type}",
                api_base=f"https://vertex.test/{api_type}",
                vertex_project="unit-project",
                vertex_location="us-central1",
                vertex_credentials=credentials,
            )
            for _ in range(5)
        )
    )

    assert search_route.call_count == 5
    assert token_route.call_count == 1
    assert search_route.calls.last.request.headers["authorization"] == "Bearer ya29.unit-token"


@pytest.mark.asyncio
async def test_a_slow_google_token_fetch_for_one_rag_store_does_not_hold_up_another(
    respx_mock: respx.MockRouter,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    slow_fetch_started: Final = threading.Event()
    other_store_searched: Final = threading.Event()

    def access_token_for(form: str) -> str:
        if "slow-store" not in form:
            return "ya29.fast"
        slow_fetch_started.set()
        return f"ya29.slow-waited-{other_store_searched.wait(timeout=5)}"

    def token_response(request: requests.PreparedRequest) -> tuple[int, dict[str, str], str]:
        form: Final = request.body.decode() if isinstance(request.body, bytes) else str(request.body)
        token: Final = {"access_token": access_token_for(form), "expires_in": 3600, "token_type": "Bearer"}
        return 200, {}, json.dumps(token)

    slow_route: Final = respx_mock.post("https://vertex.test/slow:retrieveContexts").mock(
        return_value=httpx.Response(200, json={"contexts": {"contexts": []}})
    )
    respx_mock.post("https://vertex.test/fast:retrieveContexts").mock(
        return_value=httpx.Response(200, json={"contexts": {"contexts": []}})
    )

    with responses.RequestsMock() as google:
        google.add_callback(responses.POST, GOOGLE_TOKEN_URL, callback=token_response)
        slow_search: Final = asyncio.ensure_future(
            _search_rag_store("https://vertex.test/slow", _authorized_user_credentials(f"slow-store-{uuid.uuid4().hex}"))
        )
        assert await asyncio.to_thread(slow_fetch_started.wait, 5)
        await _search_rag_store("https://vertex.test/fast", _authorized_user_credentials(f"fast-store-{uuid.uuid4().hex}"))
        other_store_searched.set()
        await slow_search

    assert slow_route.calls.last.request.headers["authorization"] == "Bearer ya29.slow-waited-True"


@pytest.mark.asyncio
async def test_a_rag_store_reloads_a_rotated_key_file_when_google_rejects_the_old_key(
    tmp_path: pathlib.Path,
    respx_mock: respx.MockRouter,
    requests_mock: requests_mock_lib.Mocker,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    key_file: Final = tmp_path / "vertex-key.json"
    key_file.write_text(_authorized_user_credentials(f"old-key-{uuid.uuid4().hex}"))
    old_key_deleted: Final = threading.Event()

    def token_response(
        request: requests_mock_lib.Request, context: requests_mock_lib.Context
    ) -> dict[str, object]:
        is_old_key: Final = "old-key" in request.text
        if is_old_key and old_key_deleted.is_set():
            context.status_code = 400
            return {"error": "invalid_grant", "error_description": "Token has been expired or revoked."}
        return {"access_token": "ya29.old" if is_old_key else "ya29.new", "expires_in": 0, "token_type": "Bearer"}

    requests_mock.post(GOOGLE_TOKEN_URL, json=token_response)
    search_route: Final = respx_mock.post("https://vertex.test/rotated:retrieveContexts").mock(
        return_value=httpx.Response(200, json={"contexts": {"contexts": []}})
    )

    await _search_rag_store("https://vertex.test/rotated", str(key_file))
    key_file.write_text(_authorized_user_credentials(f"new-key-{uuid.uuid4().hex}"))
    old_key_deleted.set()
    await _search_rag_store("https://vertex.test/rotated", str(key_file))

    assert [call.request.headers["authorization"] for call in search_route.calls] == [
        "Bearer ya29.old",
        "Bearer ya29.new",
    ]

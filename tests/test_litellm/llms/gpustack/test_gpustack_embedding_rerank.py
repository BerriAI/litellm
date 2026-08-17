import json
from typing import Final

import httpx
import pytest

import litellm
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.gpustack.common_utils import get_gpustack_endpoint, get_gpustack_headers


def test_gpustack_provider_resolution_preserves_owner_route_id() -> None:
    model, provider, _, _ = litellm.get_llm_provider(model="gpustack/owner/bge-m3")

    assert model == "owner/bge-m3"
    assert provider == litellm.LlmProviders.GPUSTACK.value


@pytest.mark.parametrize(
    ("api_base", "endpoint", "expected"),
    [
        (
            "https://gpustack.test/v1/embeddings?tenant=a#fragment",
            "/embeddings",
            "https://gpustack.test/v1/embeddings?tenant=a#fragment",
        ),
        (
            "https://embeddings/v1?tenant=a",
            "/embeddings",
            "https://embeddings/v1/embeddings?tenant=a",
        ),
        (
            "https://rerank/prefix?tenant=a",
            "/rerank",
            "https://rerank/prefix/v1/rerank?tenant=a",
        ),
        (
            "https://gpustack.test/v1?tenant=a/",
            "/embeddings",
            "https://gpustack.test/v1/embeddings?tenant=a/",
        ),
        (
            "https://gpustack.test/v1#section/",
            "/rerank",
            "https://gpustack.test/v1/rerank#section/",
        ),
    ],
)
def test_gpustack_endpoint_normalization_uses_url_path(
    api_base: str,
    endpoint: str,
    expected: str,
) -> None:
    assert get_gpustack_endpoint(api_base, endpoint) == expected


def test_gpustack_headers_deduplicate_caller_header_casing() -> None:
    headers = get_gpustack_headers(
        {
            "Authorization": "Bearer first",
            "authorization": "Bearer second",
            "CONTENT-TYPE": "application/custom",
        },
        "generated-key",
        include_accept=True,
    )

    assert [key for key in headers if key.lower() == "authorization"] == ["authorization"]
    assert headers["authorization"] == "Bearer second"
    assert [key for key in headers if key.lower() == "content-type"] == ["CONTENT-TYPE"]


def test_gpustack_embedding_posts_to_v1_embeddings_with_caller_authorization() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "owner/bge-m3",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "usage": {"prompt_tokens": 3, "total_tokens": 3},
            },
        )

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))

    response = litellm.embedding(
        model="gpustack/owner/bge-m3",
        input=["hello"],
        api_base="https://gpustack.test",
        api_key="generated-key",
        encoding_format="float",
        headers={"authorization": "Bearer caller-key"},
        client=client,
    )

    assert len(captured_requests) == 1
    request: Final = captured_requests[0]
    assert str(request.url) == "https://gpustack.test/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer caller-key"
    assert request.headers.get_list("authorization") == ["Bearer caller-key"]
    assert json.loads(request.content) == {
        "model": "owner/bge-m3",
        "input": ["hello"],
        "encoding_format": "float",
    }
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    assert response.usage.total_tokens == 3


@pytest.mark.asyncio()
async def test_gpustack_aembedding_uses_env_base_and_key_without_double_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "model": "embeddings",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.4, 0.5]}],
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    monkeypatch.setenv("GPUSTACK_API_BASE", "https://gpustack-env.test/v1")
    monkeypatch.setenv("GPUSTACK_API_KEY", "env-key")
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        response = await litellm.aembedding(
            model="gpustack/embeddings",
            input=["hello"],
            client=client,
        )
        await GLOBAL_LOGGING_WORKER.clear_queue()
        await GLOBAL_LOGGING_WORKER.stop()
    finally:
        await client.close()

    assert len(captured_requests) == 1
    request: Final = captured_requests[0]
    assert str(request.url) == "https://gpustack-env.test/v1/embeddings"
    assert request.headers["Authorization"] == "Bearer env-key"
    assert json.loads(request.content) == {"model": "embeddings", "input": ["hello"]}
    assert response.data[0]["embedding"] == [0.4, 0.5]


def test_gpustack_rerank_posts_supported_fields_and_preserves_usage() -> None:
    captured_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "model": "owner/reranker",
                "results": [
                    {
                        "index": 1,
                        "document": {"text": "second"},
                        "relevance_score": 0.98,
                    }
                ],
                "usage": {"total_tokens": 7},
            },
        )

    client: Final = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(handler)))

    response = litellm.rerank(
        model="gpustack/owner/reranker",
        query="best doc",
        documents=["first", "second"],
        top_n=1,
        return_documents=True,
        rank_fields=["ignored"],
        max_tokens_per_doc=128,
        api_base="https://gpustack.test/v1/rerank",
        api_key="generated-key",
        headers={"authorization": "Bearer caller-key"},
        client=client,
    )

    assert len(captured_requests) == 1
    request: Final = captured_requests[0]
    assert str(request.url) == "https://gpustack.test/v1/rerank"
    assert request.headers["Authorization"] == "Bearer caller-key"
    assert request.headers.get_list("authorization") == ["Bearer caller-key"]
    assert json.loads(request.content) == {
        "model": "owner/reranker",
        "query": "best doc",
        "documents": ["first", "second"],
        "top_n": 1,
        "return_documents": True,
    }
    assert response.results[0]["document"]["text"] == "second"
    assert response.results[0]["relevance_score"] == 0.98
    assert response.meta["billed_units"]["total_tokens"] == 7
    assert response.meta["tokens"]["input_tokens"] == 7


@pytest.mark.asyncio()
async def test_gpustack_arerank_uses_env_base_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "results": [{"index": 0, "document": {"text": "first"}, "relevance_score": 0.8}],
                "usage": {"total_tokens": None},
            },
        )

    monkeypatch.setenv("GPUSTACK_API_BASE", "https://gpustack.test/")
    monkeypatch.setenv("GPUSTACK_API_KEY", "generated-key")
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        response = await litellm.arerank(
            model="gpustack/reranker",
            query="best doc",
            documents=["first", "second"],
            return_documents=True,
            client=client,
        )
        await GLOBAL_LOGGING_WORKER.clear_queue()
        await GLOBAL_LOGGING_WORKER.stop()
    finally:
        await client.close()

    assert len(captured_requests) == 1
    request: Final = captured_requests[0]
    assert str(request.url) == "https://gpustack.test/v1/rerank"
    assert request.headers["Authorization"] == "Bearer generated-key"
    assert json.loads(request.content) == {
        "model": "reranker",
        "query": "best doc",
        "documents": ["first", "second"],
        "return_documents": True,
    }
    assert response.results[0]["index"] == 0
    assert response.meta["billed_units"]["total_tokens"] == 0

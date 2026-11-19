import json
from contextlib import nullcontext
from typing import Final
from dataclasses import replace

import httpx
import pytest
from fastapi import HTTPException, Request

import litellm
from litellm.proxy.pass_through_endpoints.vector_store_fallbacks import (
    PASSTHROUGH_FALLBACK_STATE,
    NativeSearchOperation,
    VectorStorePassthroughPlan,
    send_with_vector_store_fallbacks,
    set_passthrough_fallback_plan,
)
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.fallbacks import VectorStoreFallbackPlan, VectorStoreTarget
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry


def plan_for(provider: NativeSearchOperation, endpoint: str) -> VectorStorePassthroughPlan:
    targets: Final = tuple(
        VectorStoreTarget(
            name,
            provider,
            {
                "api_base": f"https://{name}.example/prefix",
                **({"api_key": f"{name}-key"} if provider in ("azure_ai", "milvus") else {}),
                "aws_access_key_id": f"{name}-aws-key",
                "aws_secret_access_key": f"{name}-secret",
                "aws_region_name": "us-east-1",
                "vertex_project": f"{name}-project",
                "vector_store_id": f"{name}-datastore",
            },
        )
        for name in ("primary", "backup")
    )
    return VectorStorePassthroughPlan(
        VectorStoreFallbackPlan(targets, cooldown_time=0), provider, endpoint, "physical-index"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [200, 400, 401, 403, 404, 408, 429, 500, 503])
@pytest.mark.parametrize(
    "provider,endpoint",
    [
        ("azure_ai", "indexes/physical-index/docs/search"),
        ("milvus", "v2/vectordb/entities/search"),
    ],
)
async def test_native_status_controls_failover_without_changing_body_or_resource(
    status: int,
    provider: NativeSearchOperation,
    endpoint: str,
) -> None:
    seen: Final[list[httpx.Request]] = []
    responses: Final[list[httpx.Response]] = []
    body: Final = b'{"collectionName":"physical-index","query":"hello"}'

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        response: Final = httpx.Response(
            status if request.url.host == "primary.example" else 200, content=b'{"native":true}'
        )
        responses.append(response)
        return response

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        response: Final = await send_with_vector_store_fallbacks(
            client,
            client.build_request(
                "POST",
                "https://primary.example/ignored?api-version=test",
                content=body,
                headers={"api-key": "primary-key", "Authorization": "Bearer primary-key"},
            ),
            plan_for(provider, endpoint),
        )
        await response.aread()
    retryable: Final = status in (408, 429, 500, 503)
    assert [request.url.host for request in seen] == (
        ["primary.example", "backup.example"] if retryable else ["primary.example"]
    )
    assert response.status_code == (200 if retryable else status)
    assert response.content == b'{"native":true}'
    assert all(request.content == body and request.url.path == f"/prefix/{endpoint}" for request in seen)
    assert all(request.url.query == b"api-version=test" for request in seen)
    if retryable:
        assert responses[0].is_closed
        assert response.headers["x-litellm-vector-store-id"] == "backup"
        assert response.headers["x-litellm-attempted-fallbacks"] == "1"
        assert seen[-1].headers.get("api-key") == ("backup-key" if provider == "azure_ai" else None)
        assert seen[-1].headers.get("authorization") == ("Bearer backup-key" if provider == "milvus" else None)


@pytest.mark.asyncio
async def test_connection_failure_falls_back_but_last_native_error_is_preserved() -> None:
    def upstream(request: httpx.Request) -> httpx.Response:
        if request.url.host == "primary.example":
            raise httpx.ConnectError("unavailable", request=request)
        return httpx.Response(503, content=b'{"error":"native overload"}', headers={"retry-after": "7"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        response: Final = await send_with_vector_store_fallbacks(
            client,
            client.build_request("GET", "https://primary.example/indexes/physical-index/docs?search=test"),
            plan_for("azure_ai", "indexes/physical-index/docs"),
        )
    assert response.status_code == 503
    assert response.content == b'{"error":"native overload"}'
    assert response.headers["retry-after"] == "7"
    assert response.is_closed


@pytest.mark.asyncio
async def test_opensearch_resigns_each_target_and_keeps_physical_index() -> None:
    seen: Final[list[httpx.Request]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503 if request.url.host == "primary.example" else 200, json={"hits": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        response: Final = await send_with_vector_store_fallbacks(
            client,
            client.build_request(
                "POST",
                "https://primary.example/public-alias/_search?size=3",
                json={"query": {"match_all": {}}},
                headers={"Authorization": "stale signature"},
            ),
            plan_for("opensearch", "public-alias/_search"),
        )
        await response.aread()
    assert [request.url.path for request in seen] == ["/prefix/physical-index/_search"] * 2
    assert all(request.url.query == b"size=3" for request in seen)
    assert "Credential=primary-aws-key/" in seen[0].headers["authorization"]
    assert "Credential=backup-aws-key/" in seen[1].headers["authorization"]
    assert seen[0].content == seen[1].content
    assert response.json() == {"hits": []}


@pytest.mark.asyncio
@pytest.mark.parametrize("configured_project", [True, False])
async def test_vertex_retargets_datastore_project_and_credentials(
    monkeypatch: pytest.MonkeyPatch, configured_project: bool
) -> None:
    from google.auth.credentials import Credentials
    import google.auth

    class StaticCredentials(Credentials):
        def refresh(self, request: object) -> None:
            self.token = "test-access-token"

    monkeypatch.setattr(google.auth, "default", lambda **_: (StaticCredentials(), "adc-project"))
    seen: Final[list[httpx.Request]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503 if "primary-datastore" in request.url.path else 200, json={"results": []})

    endpoint: Final = "v1/projects/client/locations/global/collections/default_collection/dataStores/projects/servingConfigs/default_config:search"
    original: Final = plan_for("vertex_ai", endpoint)
    plan: Final = (
        original
        if configured_project
        else replace(
            original,
            fallback_plan=replace(
                original.fallback_plan,
                targets=tuple(
                    replace(
                        target, params={key: value for key, value in target.params.items() if key != "vertex_project"}
                    )
                    for target in original.fallback_plan.targets
                ),
            ),
        )
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        response: Final = await send_with_vector_store_fallbacks(
            client,
            client.build_request("POST", "https://discoveryengine.googleapis.com/" + endpoint, json={"query": "hi"}),
            plan,
        )
        await response.aread()
    assert [request.url.path for request in seen] == [
        f"/v1/projects/{name + '-project' if configured_project else 'client'}/locations/global/collections/default_collection/dataStores/{name}-datastore/servingConfigs/default_config:search"
        for name in ("primary", "backup")
    ]
    assert all(request.headers["authorization"] == "Bearer test-access-token" for request in seen)
    assert all(json.loads(request.content) == {"query": "hi"} for request in seen)
    assert response.headers["x-litellm-vector-store-id"] == "backup"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "is_config,backup_provider,method,endpoint,expected",
    [
        (True, "azure_ai", "POST", "indexes/public/docs/search", "plan"),
        (False, "azure_ai", "POST", "indexes/public/docs/search", "none"),
        (True, "openai", "POST", "indexes/public/docs/search", "error"),
        (True, "openai", "POST", "indexes/public/docs/index", "none"),
        (True, "azure_ai", "GET", "indexes/public/docs", "plan"),
        (True, "azure_ai", "GET", "indexes/public/docs/key", "none"),
        (True, "azure_ai", "DELETE", "indexes/public/docs/search", "none"),
        (True, "azure_ai", "POST", "indexes/public/docs", "none"),
        (True, "azure_ai", "GET", "indexes/public/docs/search", "none"),
        (True, "azure_ai", "POST", "indexes/public/docs/search/extra", "none"),
    ],
)
async def test_config_only_same_provider_search_resolution(
    monkeypatch: pytest.MonkeyPatch,
    is_config: bool,
    backup_provider: str,
    method: str,
    endpoint: str,
    expected: str,
) -> None:
    primary: Final = LiteLLM_ManagedVectorStore(
        is_config=is_config,
        vector_store_id="primary",
        custom_llm_provider="azure_ai",
        litellm_params={"fallbacks": ["backup"]},
    )
    backup: Final = LiteLLM_ManagedVectorStore(
        is_config=True, vector_store_id="backup", custom_llm_provider=backup_provider
    )
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    request: Final = Request({"type": "http", "method": method, "path": "/azure/" + endpoint, "headers": []})
    if expected == "error":
        with pytest.raises(HTTPException, match="same provider"):
            await set_passthrough_fallback_plan(
                request,
                primary,
                UserAPIKeyAuth(user_role="proxy_admin"),
                operation="azure_ai",
                endpoint=endpoint,
                primary_resource_id="physical-index",
            )
    else:
        await set_passthrough_fallback_plan(
            request,
            primary,
            UserAPIKeyAuth(user_role="proxy_admin"),
            operation="azure_ai",
            endpoint=endpoint,
            primary_resource_id="physical-index",
        )
        assert isinstance(getattr(request.state, PASSTHROUGH_FALLBACK_STATE, None), VectorStorePassthroughPlan) == (
            expected == "plan"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation,provider,method,endpoint,expected",
    [
        ("milvus", "milvus", "POST", "v2/vectordb/entities/search", "plan"),
        ("milvus", "milvus", "POST", "v2/vectordb/entities/insert", "none"),
        ("milvus", "milvus", "GET", "v2/vectordb/entities/search", "none"),
        ("milvus", "milvus", "POST", "v2/vectordb/entities/query", "none"),
        ("milvus", "milvus", "POST", "v2/vectordb/entities/hybrid_search", "none"),
        ("opensearch", "opensearch", "GET", "alias/_search", "plan"),
        ("opensearch", "opensearch", "POST", "alias/_search", "plan"),
        ("opensearch", "opensearch", "DELETE", "alias/_search", "none"),
        ("opensearch", "opensearch", "GET", "alias/_count", "none"),
        ("opensearch", "opensearch", "GET", "alias/_doc/1", "none"),
        ("opensearch", "opensearch", "POST", "alias/_search/extra", "none"),
        ("opensearch", "opensearch", "POST", "alias//_search", "none"),
        ("opensearch", "opensearch", "POST", "alias/_bulk", "none"),
        (
            "vertex_ai",
            "vertex_ai/search_api",
            "POST",
            "v1/projects/project/locations/global/collections/default_collection/dataStores/store/servingConfigs/default:search",
            "plan",
        ),
        (
            "vertex_ai",
            "vertex_ai/search_api",
            "GET",
            "v1/projects/project/locations/global/collections/default_collection/dataStores/store/servingConfigs/default:search",
            "none",
        ),
        (
            "vertex_ai",
            "vertex_ai/search_api",
            "POST",
            "v1/projects/project/locations/global/collections/default_collection/dataStores/store/branches/default_branch/documents:import",
            "none",
        ),
        (
            "vertex_ai",
            "vertex_ai",
            "POST",
            "v1/projects/project/locations/global/collections/default_collection/dataStores/store/servingConfigs/default:search",
            "error",
        ),
        (
            "vertex_ai",
            "vertex_ai/search_api",
            "POST",
            "v1/projects/project/locations/global/collections/default_collection/engines/engine/servingConfigs/default:search",
            "none",
        ),
    ],
)
async def test_native_plan_only_accepts_search_with_compatible_api(
    monkeypatch: pytest.MonkeyPatch,
    operation: NativeSearchOperation,
    provider: str,
    method: str,
    endpoint: str,
    expected: str,
) -> None:
    primary: Final = LiteLLM_ManagedVectorStore(
        is_config=True,
        vector_store_id="primary",
        custom_llm_provider=provider,
        litellm_params={"fallbacks": ["backup"]},
    )
    backup: Final = LiteLLM_ManagedVectorStore(is_config=True, vector_store_id="backup", custom_llm_provider=provider)
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    request: Final = Request({"type": "http", "method": method, "path": "/" + endpoint, "headers": []})
    if expected == "error":
        with pytest.raises(HTTPException, match="search_api data stores"):
            await set_passthrough_fallback_plan(
                request,
                primary,
                UserAPIKeyAuth(user_role="proxy_admin"),
                operation=operation,
                endpoint=endpoint,
                primary_resource_id="physical-index",
            )
    else:
        await set_passthrough_fallback_plan(
            request,
            primary,
            UserAPIKeyAuth(user_role="proxy_admin"),
            operation=operation,
            endpoint=endpoint,
            primary_resource_id="physical-index",
        )
        assert isinstance(getattr(request.state, PASSTHROUGH_FALLBACK_STATE, None), VectorStorePassthroughPlan) == (
            expected == "plan"
        )


@pytest.mark.asyncio
async def test_config_store_without_fallbacks_keeps_ordinary_sender() -> None:
    request: Final = Request(
        {"type": "http", "method": "POST", "path": "/azure/indexes/index/docs/search", "headers": []}
    )
    await set_passthrough_fallback_plan(
        request,
        LiteLLM_ManagedVectorStore(is_config=True, vector_store_id="ordinary", custom_llm_provider="azure_ai"),
        UserAPIKeyAuth(user_role="proxy_admin"),
        operation="azure_ai",
        endpoint="indexes/index/docs/search",
        primary_resource_id="index",
    )
    assert not hasattr(request.state, PASSTHROUGH_FALLBACK_STATE)


@pytest.mark.asyncio
async def test_mid_body_disconnect_closes_primary_and_sends_modified_body_to_backup() -> None:
    from collections.abc import AsyncIterator
    from litellm.proxy.pass_through_endpoints.pass_through_endpoints import HttpPassThroughEndpointHelpers

    class DroppedBody(httpx.AsyncByteStream):
        closed: bool = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b'{"value":['
            raise httpx.RemoteProtocolError("connection dropped mid-response")

        async def aclose(self) -> None:
            self.closed = True

    dropped: Final = DroppedBody()
    observed: Final[list[httpx.Request]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        if request.url.host == "primary.example":
            return httpx.Response(200, stream=dropped)
        return httpx.Response(200, json={"value": ["backup result"]})

    request: Final = Request(
        {"type": "http", "method": "POST", "path": "/azure/indexes/stream-docs/docs/search", "headers": []}
    )
    setattr(request.state, PASSTHROUGH_FALLBACK_STATE, plan_for("azure_ai", "indexes/stream-docs/docs/search"))
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        response: Final = await HttpPassThroughEndpointHelpers.non_streaming_http_request_handler(
            request=request,
            async_client=client,
            url=httpx.URL("https://primary.example/indexes/stream-docs/docs/search"),
            headers={},
            _parsed_body={"search": "changed by hook"},
        )
    assert response.json() == {"value": ["backup result"]}
    assert response.headers["x-litellm-vector-store-id"] == "backup"
    assert dropped.closed
    assert [json.loads(item.content) for item in observed] == [{"search": "changed by hook"}] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["credentials", "foreign_index"])
async def test_invalid_target_preparation_is_terminal_without_sending(invalid: str) -> None:
    observed: Final[list[httpx.Request]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        observed.append(request)
        return httpx.Response(200)

    original: Final = plan_for("azure_ai" if invalid == "credentials" else "opensearch", "index/_search")
    primary: Final = original.fallback_plan.targets[0]
    plan: Final = replace(
        original,
        fallback_plan=replace(
            original.fallback_plan,
            targets=(
                replace(primary, params={key: value for key, value in primary.params.items() if key != "api_key"}),
                original.fallback_plan.targets[1],
            ),
        ),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        with pytest.raises((ValueError, HTTPException)):
            await send_with_vector_store_fallbacks(
                client,
                client.build_request(
                    "POST",
                    "https://primary.example/index/_search",
                    json={"query": {"terms": {"user": {"index": "foreign", "id": "1", "path": "user"}}}},
                ),
                plan,
            )
    assert observed == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [400, 503])
async def test_malformed_gzip_keeps_received_status_retryability(status: int) -> None:
    seen: Final[list[httpx.Request]] = []
    primary: Final = httpx.Response(
        status, headers={"Content-Encoding": "gzip"}, stream=httpx.ByteStream(b"invalid gzip")
    )

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return primary if request.url.host == "primary.example" else httpx.Response(200, json={"value": ["backup"]})

    expectation: Final = pytest.raises(httpx.DecodingError) if status == 400 else nullcontext()
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        with expectation:
            response: Final = await send_with_vector_store_fallbacks(
                client,
                client.build_request("POST", "https://primary.example/indexes/docs/docs/search", json={"search": "hi"}),
                plan_for("azure_ai", "indexes/docs/docs/search"),
            )
            assert response.status_code == 200
            assert response.json() == {"value": ["backup"]}
            assert response.headers["x-litellm-vector-store-id"] == "backup"
    assert primary.is_closed
    assert [request.url.host for request in seen] == (
        ["primary.example"] if status == 400 else ["primary.example", "backup.example"]
    )

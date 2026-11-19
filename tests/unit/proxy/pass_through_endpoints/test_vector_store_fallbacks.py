import json
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from typing import Final

import httpx
import pytest
from fastapi import HTTPException, Request

import litellm
from litellm.proxy.pass_through_endpoints.llm_passthrough_endpoints import (
    _azure_search_fallback_request,
    _milvus_fallback_request,
    _opensearch_fallback_request,
    VERTEX_SEARCH_PATH,
    _vertex_search_fallback_request,
)
from litellm.proxy.pass_through_endpoints.vector_store_fallbacks import (
    PASSTHROUGH_FALLBACK_STATE,
    NativeSearchProvider,
    TargetRequestBuilder,
    VectorStorePassthroughPlan,
    retargeted_request,
    send_with_vector_store_fallbacks,
    set_passthrough_fallback_plan,
)
from litellm.llms.opensearch.vector_stores.index_passthrough import index_operation
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager
from litellm.types.vector_stores import LiteLLM_ManagedVectorStore
from litellm.vector_stores.fallbacks import VectorStoreFallbackPlan, VectorStoreTarget
from litellm.vector_stores.vector_store_registry import VectorStoreRegistry


ENCODED_TRAVERSAL: Final = (
    "v1/projects/p/locations/global/collections/default_collection/dataStores/allowed/servingConfigs/"
    "x%252F..%252F..%252F..%252Fsecret%252FservingConfigs%252Fdefault_config:search"
)


def route_builder(provider: NativeSearchProvider, endpoint: str) -> TargetRequestBuilder:
    match provider:
        case "azure_ai" | "milvus":
            config: Final = ProviderConfigManager.get_provider_vector_stores_config(LlmProviders(provider))
            assert config is not None
            builder: Final = _azure_search_fallback_request if provider == "azure_ai" else _milvus_fallback_request
            return partial(builder, config, endpoint)
        case "opensearch":
            return partial(
                _opensearch_fallback_request, ("_search",), index_operation("POST", ("_search",)), "physical-index", True
            )
        case "vertex_ai":
            matched: Final = VERTEX_SEARCH_PATH.fullmatch(endpoint)
            assert matched is not None
            return partial(_vertex_search_fallback_request, matched)


async def refuse_build(target: VectorStoreTarget, request: httpx.Request) -> httpx.Request:
    raise AssertionError("eligibility tests never build a target request")


def plan_for(provider: NativeSearchProvider, endpoint: str, client_query: str = "") -> VectorStorePassthroughPlan:
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
        VectorStoreFallbackPlan(targets, cooldown_time=0), route_builder(provider, endpoint), client_query
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
    provider: NativeSearchProvider,
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
                f"https://primary.example/prefix/{endpoint}?api-version=test",
                content=body,
                headers={"api-key": "primary-key", "Authorization": "Bearer primary-key"},
            ),
            plan_for(provider, endpoint, "api-version=test"),
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
            plan_for("azure_ai", "indexes/physical-index/docs", "search=test"),
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
                "https://primary.example/prefix/physical-index/_search?size=3",
                json={"query": {"match_all": {}}},
                headers={"Authorization": "primary signature"},
            ),
            plan_for("opensearch", "public-alias/_search", "size=3"),
        )
        await response.aread()
    assert [request.url.path for request in seen] == ["/prefix/physical-index/_search"] * 2
    assert all(request.url.query == b"size=3" for request in seen)
    assert seen[0].headers["authorization"] == "primary signature"
    assert "Credential=backup-aws-key/" in seen[1].headers["authorization"]
    assert seen[0].content == seen[1].content
    assert response.json() == {"hits": []}


@pytest.mark.parametrize(
    ("client_url", "expected"),
    [
        ("https://primary/search", "https://backup/search?api-version=1"),
        ("https://primary/search?q=water", "https://backup/search?api-version=1&q=water"),
    ],
)
def test_retargeted_request_merges_client_query_into_fallback_query(client_url: str, expected: str) -> None:
    retargeted: Final = retargeted_request(
        httpx.Request("GET", client_url, headers={"api-key": "primary", "accept": "application/json"}),
        "https://backup/search?api-version=1",
        {"api-key": "backup"},
    )
    assert str(retargeted.url) == expected
    assert (retargeted.headers["api-key"], retargeted.headers["accept"]) == ("backup", "application/json")


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

    endpoint: Final = "v1/projects/client/locations/global/collections/default_collection/dataStores/primary-datastore/servingConfigs/default_config:search"
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
            client.build_request(
                "POST",
                "https://discoveryengine.googleapis.com/" + endpoint,
                json={"query": "hi"},
                headers={"Authorization": "Bearer route-token"},
            ),
            plan,
        )
        await response.aread()
    assert [request.url.path for request in seen] == [
        "/" + endpoint,
        f"/v1/projects/{'backup-project' if configured_project else 'adc-project'}/locations/global/collections/default_collection/dataStores/backup-datastore/servingConfigs/default_config:search",
    ]
    assert [request.headers["authorization"] for request in seen] == [
        "Bearer route-token",
        "Bearer test-access-token",
    ]
    assert all(json.loads(request.content) == {"query": "hi"} for request in seen)
    assert response.headers["x-litellm-vector-store-id"] == "backup"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,primary_provider,backup_provider,backup_params,is_config,expected",
    [
        ("azure_ai", "azure_ai", "azure_ai", {}, True, "plan"),
        ("azure_ai", "azure_ai", "azure_ai", {}, False, "none"),
        ("azure_ai", "azure_ai", "openai", {}, True, "same provider"),
        ("milvus", "azure_ai", "azure_ai", {}, True, "same provider"),
        ("vertex_ai", "vertex_ai/search_api", "vertex_ai/search_api", {}, True, "plan"),
        ("vertex_ai", "vertex_ai", "vertex_ai", {}, True, "search_api data stores"),
        (
            "vertex_ai",
            "vertex_ai/search_api",
            "vertex_ai/search_api",
            {"vertex_engine_id": "engine"},
            True,
            "search_api data stores",
        ),
    ],
)
async def test_plan_requires_yaml_stores_of_one_compatible_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: NativeSearchProvider,
    primary_provider: str,
    backup_provider: str,
    backup_params: dict[str, str],
    is_config: bool,
    expected: str,
) -> None:
    primary: Final = LiteLLM_ManagedVectorStore(
        is_config=is_config,
        vector_store_id="primary",
        custom_llm_provider=primary_provider,
        litellm_params={"fallbacks": ["backup"]},
    )
    backup: Final = LiteLLM_ManagedVectorStore(
        is_config=True, vector_store_id="backup", custom_llm_provider=backup_provider, litellm_params=backup_params
    )
    monkeypatch.setattr(litellm, "vector_store_registry", VectorStoreRegistry([primary, backup]))
    request: Final = Request({"type": "http", "method": "POST", "path": "/search", "headers": []})
    attach: Final = set_passthrough_fallback_plan(
        request, primary, UserAPIKeyAuth(user_role="proxy_admin"), provider=provider, build_request=refuse_build
    )
    if expected in ("plan", "none"):
        await attach
        assert isinstance(getattr(request.state, PASSTHROUGH_FALLBACK_STATE, None), VectorStorePassthroughPlan) == (
            expected == "plan"
        )
    else:
        with pytest.raises(HTTPException, match=expected):
            await attach


@pytest.mark.parametrize(
    "path,matches",
    [
        (
            "v1/projects/project/locations/global/collections/default_collection/dataStores/store/servingConfigs/default:search",
            True,
        ),
        (
            "v1alpha/projects/123/locations/us/collections/c_1/dataStores/store-a_1/servingConfigs/s-1:search",
            True,
        ),
        (
            "v1/projects/project/locations/global/collections/default_collection/dataStores/store/branches/default_branch/documents:import",
            False,
        ),
        (
            "v1/projects/project/locations/global/collections/default_collection/engines/engine/servingConfigs/default:search",
            False,
        ),
        (
            "v1/projects/Project/locations/global/collections/default_collection/dataStores/store/servingConfigs/default:search",
            False,
        ),
        (ENCODED_TRAVERSAL, False),
    ],
)
def test_vertex_search_path_accepts_only_plain_data_store_search(path: str, matches: bool) -> None:
    assert (VERTEX_SEARCH_PATH.fullmatch(path) is not None) == matches


@pytest.mark.asyncio
async def test_config_store_without_fallbacks_keeps_ordinary_sender() -> None:
    request: Final = Request(
        {"type": "http", "method": "POST", "path": "/azure/indexes/index/docs/search", "headers": []}
    )
    await set_passthrough_fallback_plan(
        request,
        LiteLLM_ManagedVectorStore(is_config=True, vector_store_id="ordinary", custom_llm_provider="azure_ai"),
        UserAPIKeyAuth(user_role="proxy_admin"),
        provider="azure_ai",
        build_request=refuse_build,
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
        return httpx.Response(503)

    original: Final = plan_for("azure_ai" if invalid == "credentials" else "opensearch", "index/_search")
    backup: Final = original.fallback_plan.targets[1]
    plan: Final = replace(
        original,
        fallback_plan=replace(
            original.fallback_plan,
            targets=(
                original.fallback_plan.targets[0],
                replace(backup, params={key: value for key, value in backup.params.items() if key != "api_key"}),
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
    assert [request.url.host for request in observed] == ["primary.example"]


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



@pytest.mark.asyncio
async def test_azure_backup_uses_its_own_base_query_and_service_name() -> None:
    seen: Final[list[httpx.Request]] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(503 if request.url.host == "primary.example" else 200, json={"value": []})

    original: Final = plan_for("azure_ai", "indexes/docs/docs/search")
    primary, backup = original.fallback_plan.targets
    plan: Final = replace(
        original,
        fallback_plan=replace(
            original.fallback_plan,
            targets=(
                primary,
                replace(
                    backup,
                    params={
                        **{key: value for key, value in backup.params.items() if key != "api_base"},
                        "azure_search_service_name": "backup-service",
                    },
                ),
            ),
        ),
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        response: Final = await send_with_vector_store_fallbacks(
            client,
            client.build_request(
                "POST", "https://primary.example/indexes/docs/docs/search?api-version=primary", json={"search": "hi"}
            ),
            plan,
        )
    assert response.status_code == 200
    assert [str(request.url) for request in seen] == [
        "https://primary.example/indexes/docs/docs/search?api-version=primary",
        "https://backup-service.search.windows.net/indexes/docs/docs/search",
    ]

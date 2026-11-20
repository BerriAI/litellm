import json
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from functools import partial
from itertools import product
from pathlib import Path
from typing import Final

import pytest
import yaml
from pydantic import JsonValue

from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.forward_proxy import tunnelling_forward_proxy
from tests.integration._support.process import owned_proxy
from tests.integration._support.redis_process import owned_redis
from tests.integration._support.tls import server_context, write_self_signed_cert
from tests.integration._support.vertex import service_account_json
from tests.integration._support.wire import Reply, Request, Wire, wire_server

PROVIDERS: Final = ("azure_ai", "milvus", "opensearch")


@contextmanager
def _native_search_gateway(
    gateway: Gateway,
    wire: Wire,
    config: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[Gateway, Wire]]:
    directory: Final = tmp_path_factory.mktemp("native-search-isolated")
    with (
        owned_redis(directory) as cache,
        owned_proxy(
            gateway,
            directory,
            {"DATABASE_URL": os.environ["DATABASE_URL"], "REDIS_HOST": cache.host, "REDIS_PORT": str(cache.port)},
            config=config,
            workers=2,
        ) as proxy,
    ):
        yield proxy, wire


@pytest.fixture
def native_search_gateway(
    native_search_gateway_factory: Callable[[], AbstractContextManager[tuple[Gateway, Wire]]],
) -> Iterator[tuple[Gateway, Wire]]:
    with native_search_gateway_factory() as isolated:
        yield isolated


@pytest.fixture(scope="module")
def native_search_gateway_factory(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[Callable[[], AbstractContextManager[tuple[Gateway, Wire]]]]:
    directory: Final = tmp_path_factory.mktemp("native-search-fallbacks")

    def upstream(request: Request) -> Reply:
        primary: Final = "/primary/" in request.target or b"all-fail" in request.body
        return Reply(status=503 if primary else 200, body=b'{"native_results":["document"]}')

    with wire_server(upstream) as wire, gateway_from_environment() as gateway:
        for provider in PROVIDERS:
            created: Final = gateway.request(
                "POST",
                "/v1/indexes",
                {
                    "index_name": f"stream-fallback-{provider}",
                    "litellm_params": {
                        "vector_store_name": f"primary-{provider}",
                        "vector_store_index": "physical-index",
                    },
                },
            )
            assert created.status_code == 200, created.text
            db_store: Final = gateway.request(
                "POST",
                "/vector_store/new",
                {
                    "vector_store_id": f"db-{provider}",
                    "vector_store_name": f"db-{provider}",
                    "custom_llm_provider": provider,
                    "litellm_params": {
                        "api_base": f"{wire.url}/backup",
                        **(
                            {"api_key": "backup-key"}
                            if provider != "opensearch"
                            else {
                                "aws_access_key_id": "backup-aws-key",
                                "aws_secret_access_key": "backup-secret",
                                "aws_region_name": "us-east-1",
                            }
                        ),
                    },
                },
            )
            assert db_store.status_code == 200, db_store.text
            for source, store in (("config", f"backup-{provider}"), ("db", f"db-{provider}")):
                ordinary: Final = gateway.request(
                    "POST",
                    "/v1/indexes",
                    {
                        "index_name": f"ordinary-{source}-{provider}",
                        "litellm_params": {
                            "vector_store_name": store,
                            "vector_store_index": "physical-index",
                        },
                    },
                )
                assert ordinary.status_code == 200, ordinary.text
        for provider in ("azure_ai", "opensearch"):
            registered_get: Final = gateway.request(
                "POST",
                "/v1/indexes",
                {
                    "index_name": f"get-fallback-{provider}",
                    "litellm_params": {
                        "vector_store_name": f"get-primary-{provider}",
                        "vector_store_index": "physical-index",
                    },
                },
            )
            assert registered_get.status_code == 200, registered_get.text
        stores: Final = [
            {
                "vector_store_name": f"{role}-{provider}",
                "litellm_params": {
                    "vector_store_id": f"{role}-{provider}",
                    "custom_llm_provider": provider,
                    "api_base": f"{wire.url}/{role}",
                    **(
                        {"api_key": f"{role}-key"}
                        if provider != "opensearch"
                        else {
                            "aws_access_key_id": f"{role}-aws-key",
                            "aws_secret_access_key": f"{role}-secret",
                            "aws_region_name": "us-east-1",
                        }
                    ),
                    **({"fallbacks": [f"backup-{provider}"]} if role == "primary" else {}),
                },
            }
            for provider, role in product(PROVIDERS, ("primary", "backup"))
        ]
        get_stores: Final = [
            {
                **entry,
                "vector_store_name": "get-" + entry["vector_store_name"],
                "litellm_params": {
                    **entry["litellm_params"],
                    "vector_store_id": "get-" + entry["litellm_params"]["vector_store_id"],
                    **(
                        {"fallbacks": ["get-" + entry["litellm_params"]["fallbacks"][0]]}
                        if "fallbacks" in entry["litellm_params"]
                        else {}
                    ),
                },
            }
            for entry in stores
            if entry["litellm_params"]["custom_llm_provider"] in ("azure_ai", "opensearch")
        ]
        configuration: Final = {
            **yaml.safe_load((Path(__file__).parents[1] / "proxy_config.yaml").read_text()),
            "router_settings": {"cooldown_time": 60},
            "vector_store_registry": [*stores, *get_stores],
        }
        config: Final = directory / "config.yaml"
        config.write_text(yaml.safe_dump(configuration))
        yield partial(_native_search_gateway, gateway, wire, config, tmp_path_factory)


@pytest.mark.parametrize(
    "provider,path,body,upstream_path",
    [
        (
            "azure_ai",
            "/azure/indexes/stream-fallback-azure_ai/docs/search?api-version=2024-07-01",
            {"search": "hello"},
            "/indexes/stream-fallback-azure_ai/docs/search?api-version=2024-07-01",
        ),
        (
            "milvus",
            "/milvus/v2/vectordb/entities/search",
            {"collectionName": "stream-fallback-milvus", "data": [[1.0, 0.0]]},
            "/v2/vectordb/entities/search",
        ),
        (
            "opensearch",
            "/opensearch/stream-fallback-opensearch/_search?size=3",
            {"query": {"match_all": {}}},
            "/physical-index/_search?size=3",
        ),
    ],
)
def test_native_search_reuses_alias_and_returns_backup_body(
    native_search_gateway: tuple[Gateway, Wire],
    provider: str,
    path: str,
    body: Mapping[str, JsonValue],
    upstream_path: str,
) -> None:
    gateway, wire = native_search_gateway
    wire.drain()
    response: Final = gateway.request("POST", path, body)
    assert response.status_code == 200, response.text
    assert response.json() == {"native_results": ["document"]}
    assert response.headers["x-litellm-vector-store-id"] == f"backup-{provider}"
    assert response.headers["x-litellm-attempted-fallbacks"] == "1"
    assert "/backup" in response.headers["x-litellm-model-api-base"]
    observed: Final = wire.drain()
    assert [request.target for request in observed] == [f"/{role}{upstream_path}" for role in ("primary", "backup")]
    expected_body: Final = {**body, "collectionName": "physical-index"} if provider == "milvus" else body
    assert [json.loads(request.body) for request in observed] == [expected_body, expected_body]
    if provider == "opensearch":
        assert "Credential=backup-aws-key/" in observed[-1].headers["authorization"]
    else:
        header: Final = "api-key" if provider == "azure_ai" else "authorization"
        assert observed[-1].headers[header] == ("backup-key" if provider == "azure_ai" else "Bearer backup-key")


@pytest.mark.parametrize(
    "method,path,body,upstream_target",
    [
        (
            "POST",
            "/milvus/v2/vectordb/entities/insert",
            {"collectionName": "stream-fallback-milvus", "data": [{"id": 1}]},
            "/primary/v2/vectordb/entities/insert",
        ),
        (
            "POST",
            "/milvus/v2/vectordb/entities/query",
            {"collectionName": "stream-fallback-milvus", "filter": "id > 0"},
            "/primary/v2/vectordb/entities/query",
        ),
        (
            "POST",
            "/azure/indexes/stream-fallback-azure_ai/docs/index",
            {"value": [{"id": "1"}]},
            "/primary/indexes/stream-fallback-azure_ai/docs/index",
        ),
        (
            "GET",
            "/azure/indexes/stream-fallback-azure_ai/docs/search",
            None,
            "/primary/indexes/stream-fallback-azure_ai/docs/search",
        ),
        ("GET", "/opensearch/stream-fallback-opensearch/_count", None, "/primary/physical-index/_count"),
        ("GET", "/opensearch/stream-fallback-opensearch/_doc/1", None, "/primary/physical-index/_doc/1"),
    ],
)
def test_native_non_search_requests_never_fail_over(
    native_search_gateway: tuple[Gateway, Wire],
    method: str,
    path: str,
    body: Mapping[str, JsonValue] | None,
    upstream_target: str,
) -> None:
    gateway, wire = native_search_gateway
    wire.drain()
    response: Final = gateway.request(method, path, body)
    assert response.status_code == 503, response.text
    assert "x-litellm-vector-store-id" not in response.headers
    assert [request.target for request in wire.drain()] == [upstream_target]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("source", ("config", "db"))
def test_ordinary_native_search_uses_config_and_db_stores_without_fallbacks(
    native_search_gateway: tuple[Gateway, Wire],
    provider: str,
    source: str,
) -> None:
    gateway, wire = native_search_gateway
    alias: Final = f"ordinary-{source}-{provider}"
    path, body, expected_path = {
        "azure_ai": (f"/azure/indexes/{alias}/docs/search", {"search": "hi"}, f"/indexes/{alias}/docs/search"),
        "milvus": (
            "/milvus/v2/vectordb/entities/search",
            {"collectionName": alias, "data": [[1.0]]},
            "/v2/vectordb/entities/search",
        ),
        "opensearch": (f"/opensearch/{alias}/_search", {"query": {"match_all": {}}}, "/physical-index/_search"),
    }[provider]
    wire.drain()
    response: Final = gateway.request("POST", path, body)
    assert response.status_code == 200, response.text
    assert response.json() == {"native_results": ["document"]}
    assert "x-litellm-vector-store-id" not in response.headers
    assert [request.target for request in wire.drain()] == ["/backup" + expected_path]


def test_all_cooled_native_stores_return_retry_after_without_upstream_calls(
    native_search_gateway: tuple[Gateway, Wire],
) -> None:
    gateway, wire = native_search_gateway
    path: Final = "/azure/indexes/stream-fallback-azure_ai/docs/search"
    first: Final = gateway.request("POST", path, {"search": "all-fail"})
    assert first.status_code == 503, first.text
    wire.drain()
    cooled: Final = gateway.request("POST", path, {"search": "all-fail"})
    assert cooled.status_code == 503, cooled.text
    assert int(cooled.headers["retry-after"]) > 0
    assert "All configured vector stores are in cooldown" in cooled.text
    assert wire.drain() == ()


@pytest.mark.parametrize(
    "provider,path,upstream_path",
    [
        (
            "azure_ai",
            "/azure/indexes/get-fallback-azure_ai/docs?search=hello&api-version=2024-07-01",
            "/indexes/get-fallback-azure_ai/docs?search=hello&api-version=2024-07-01",
        ),
        (
            "opensearch",
            "/opensearch/get-fallback-opensearch/_search?q=hello&size=3",
            "/physical-index/_search?q=hello&size=3",
        ),
    ],
)
def test_native_get_search_fails_over_with_query_and_no_body(
    native_search_gateway: tuple[Gateway, Wire],
    provider: str,
    path: str,
    upstream_path: str,
) -> None:
    gateway, wire = native_search_gateway
    wire.drain()
    response: Final = gateway.request("GET", path)
    assert response.status_code == 200, response.text
    assert response.json() == {"native_results": ["document"]}
    assert response.headers["x-litellm-vector-store-id"] == f"get-backup-{provider}"
    observed: Final = wire.drain()
    assert [(item.method, item.target, item.body) for item in observed] == [
        ("GET", f"/{role}{upstream_path}", b"") for role in ("primary", "backup")
    ]


@pytest.fixture(scope="module")
def vertex_search_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire]]:
    directory: Final = tmp_path_factory.mktemp("vertex-search-fallback")

    def search(request: Request) -> Reply:
        assert request.headers["authorization"] == "Bearer scripted-search-token"
        return Reply(
            status=503 if "/dataStores/primary-store/" in request.target else 200,
            body=b'{"results":[{"id":"backup-document"}]}',
        )

    with (
        wire_server(search, tls=server_context(*write_self_signed_cert(directory))) as upstream,
        wire_server(
            lambda _: Reply(body=b'{"access_token":"scripted-search-token","expires_in":3600,"token_type":"Bearer"}')
        ) as tokens,
        tunnelling_forward_proxy(int(upstream.url.rsplit(":", 1)[1])) as tunnel,
        gateway_from_environment() as gateway,
    ):
        configuration: Final = yaml.safe_load((Path(__file__).parents[1] / "proxy_config.yaml").read_text())
        configuration["vector_store_registry"] = [
            {
                "vector_store_name": f"{role}-store",
                "litellm_params": {
                    "vector_store_id": f"{role}-store",
                    "custom_llm_provider": "vertex_ai/search_api",
                    "vertex_project": f"{role}-project",
                    "vertex_location": "global",
                    "vertex_credentials": service_account_json(f"{role}-project", tokens.url),
                    **({"fallbacks": ["backup-store"]} if role == "primary" else {}),
                },
            }
            for role in ("primary", "backup")
        ]
        config: Final = directory / "config.yaml"
        config.write_text(yaml.safe_dump(configuration))
        ambient_credentials: Final = directory / "ambient-credentials.json"
        ambient_credentials.write_text(service_account_json("primary-project", tokens.url))
        with owned_proxy(
            gateway,
            directory,
            {
                "GOOGLE_APPLICATION_CREDENTIALS": str(ambient_credentials),
                "HTTPS_PROXY": tunnel.url,
                "NO_PROXY": "127.0.0.1,localhost",
                "SSL_VERIFY": "False",
                "AIOHTTP_TRUST_ENV": "True",
                "DATABASE_URL": os.environ["DATABASE_URL"],
            },
            config=config,
            workers=2,
        ) as proxy:
            yield proxy, upstream


def test_vertex_search_fails_over_through_proxy_with_target_resource(
    vertex_search_gateway: tuple[Gateway, Wire],
) -> None:
    gateway, wire = vertex_search_gateway
    path: Final = "/v1/projects/primary-project/locations/global/collections/default_collection/dataStores/primary-store/servingConfigs/default_search:search"
    response: Final = gateway.request("POST", "/vertex_ai/discovery" + path, {"query": "hello"})
    assert response.status_code == 200, response.text
    assert response.json() == {"results": [{"id": "backup-document"}]}
    assert response.headers["x-litellm-vector-store-id"] == "backup-store"
    assert [(request.method, request.target, json.loads(request.body)) for request in wire.drain()] == [
        (
            "POST",
            path.replace("primary-project", f"{role}-project").replace("primary-store", f"{role}-store"),
            {"query": "hello"},
        )
        for role in ("primary", "backup")
    ]


def test_vertex_encoded_traversal_never_reaches_another_datastore(
    vertex_search_gateway: tuple[Gateway, Wire],
) -> None:
    gateway, wire = vertex_search_gateway
    wire.drain()
    path: Final = (
        "/v1/projects/primary-project/locations/global/collections/default_collection/dataStores/primary-store"
        "/servingConfigs/x%25252F..%25252F..%25252F..%25252Fbackup-store%25252FservingConfigs%25252Fdefault_config:search"
    )
    response: Final = gateway.request("POST", "/vertex_ai/discovery" + path, {"query": "hello"})
    assert response.status_code == 503, response.text
    assert "x-litellm-vector-store-id" not in response.headers
    assert [request.target.split("/servingConfigs/")[0] for request in wire.drain()] == [
        "/v1/projects/primary-project/locations/global/collections/default_collection/dataStores/primary-store"
    ]


def test_vertex_non_search_request_never_fails_over(vertex_search_gateway: tuple[Gateway, Wire]) -> None:
    gateway, wire = vertex_search_gateway
    wire.drain()
    path: Final = "/v1/projects/primary-project/locations/global/collections/default_collection/dataStores/primary-store/branches/default_branch/documents:import"
    response: Final = gateway.request("POST", "/vertex_ai/discovery" + path, {"inlineSource": {"documents": []}})
    assert response.status_code == 503, response.text
    assert "x-litellm-vector-store-id" not in response.headers
    assert [request.target for request in wire.drain()] == [path]

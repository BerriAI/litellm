import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import pytest
import yaml

from tests.integration._support.client import Gateway, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Wire, wire_server

PROVIDERS: Final = ("azure_ai", "milvus", "opensearch")


@pytest.fixture(scope="module")
def native_search_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire]]:
    directory: Final = tmp_path_factory.mktemp("native-search-fallbacks")

    def upstream(request):
        primary: Final = "/primary/" in request.target
        return Reply(status=503 if primary else 200, body=b'{"native_results":["document"]}')

    with wire_server(upstream) as wire, gateway_from_environment() as gateway:
        for provider in PROVIDERS:
            created: Final = gateway.request(
                "POST",
                "/v1/indexes",
                {
                    "index_name": f"fallback-{provider}",
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
        configuration: Final = yaml.safe_load((Path(__file__).parents[1] / "proxy_config.yaml").read_text())
        configuration["router_settings"] = {"cooldown_time": 0}
        configuration["vector_store_registry"] = [
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
            for provider in PROVIDERS
            for role in ("primary", "backup")
        ]
        config: Final = directory / "config.yaml"
        config.write_text(yaml.safe_dump(configuration))
        with owned_proxy(
            gateway, directory, {"DATABASE_URL": os.environ["DATABASE_URL"]}, config=config, workers=2
        ) as proxy:
            yield proxy, wire


@pytest.mark.parametrize(
    "provider,path,body,upstream_path",
    [
        (
            "azure_ai",
            "/azure/indexes/fallback-azure_ai/docs/search?api-version=2024-07-01",
            {"search": "hello"},
            "/indexes/fallback-azure_ai/docs/search?api-version=2024-07-01",
        ),
        (
            "milvus",
            "/milvus/v2/vectordb/entities/search",
            {"collectionName": "fallback-milvus", "data": [[1.0, 0.0]]},
            "/v2/vectordb/entities/search",
        ),
        (
            "opensearch",
            "/opensearch/fallback-opensearch/_search?size=3",
            {"query": {"match_all": {}}},
            "/physical-index/_search?size=3",
        ),
    ],
)
def test_native_search_reuses_alias_and_returns_backup_body(
    native_search_gateway: tuple[Gateway, Wire],
    provider: str,
    path: str,
    body,
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


def test_native_write_does_not_fail_over(native_search_gateway: tuple[Gateway, Wire]) -> None:
    gateway, wire = native_search_gateway
    wire.drain()
    response: Final = gateway.request(
        "POST", "/milvus/v2/vectordb/entities/insert", {"collectionName": "fallback-milvus", "data": [{"id": 1}]}
    )
    assert response.status_code == 503, response.text
    assert [request.target for request in wire.drain()] == ["/primary/v2/vectordb/entities/insert"]


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

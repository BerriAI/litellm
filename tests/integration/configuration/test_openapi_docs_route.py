import time
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, list_value, object_value
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from pydantic import JsonValue, TypeAdapter

from litellm.proxy._lazy_features import LAZY_FEATURES

DOCS_SECONDS: Final = 120
COLD_DOCS: Final = 4
CHATS: Final = 8
JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
OWNED_PROXY_TIMEOUT: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)
TWO_OWNED_PROXIES_TIMEOUT: Final = pytest.mark.timeout(2 * (2 * graceful_stop_seconds() + 120))
LAZY_ROUTES_FLAG: Final = "LITELLM_DISABLE_LAZY_ROUTES"
WARMED_FEATURE: Final = next(feature for feature in LAZY_FEATURES if feature.name == "policies")


def _timed_get(candidate: Gateway, path: str) -> tuple[httpx.Response, float]:
    started: Final = time.perf_counter()
    response: Final = candidate.request("GET", path)
    return response, time.perf_counter() - started


def _liveliness_until(done: Future[tuple[httpx.Response, float]], candidate: Gateway) -> Iterator[tuple[int, float]]:
    while not done.done():
        response, seconds = _timed_get(candidate, "/health/liveliness")
        yield response.status_code, seconds


def _schema(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, response.text[:500]
    return JSON_OBJECT.validate_json(response.content)


def _servers(response: httpx.Response) -> list[JsonValue]:
    return list(list_value(_schema(response).get("servers", [])))


def _warmed_feature_operations(response: httpx.Response) -> dict[str, JsonValue]:
    paths: Final = object_value(_schema(response)["paths"])
    return {path: operations for path, operations in paths.items() if WARMED_FEATURE.matches(path)}


@OWNED_PROXY_TIMEOUT
def test_liveliness_answers_while_a_cold_openapi_schema_builds(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_proxy_process(gateway, tmp_path, {}, client_timeout=DOCS_SECONDS) as owned,
        ThreadPoolExecutor(1) as pool,
    ):
        docs: Final = pool.submit(_timed_get, owned.gateway, "/openapi.json")
        probes: Final = tuple(_liveliness_until(docs, owned.gateway))
        docs_response, docs_seconds = docs.result()
    assert "/v1/chat/completions" in object_value(_schema(docs_response)["paths"])
    assert probes, "no liveliness probe ran while the docs request was pending"
    assert {status for status, _ in probes} == {200}, probes
    slowest: Final = max(seconds for _, seconds in probes)
    assert slowest < docs_seconds / 3, (
        f"a liveliness probe waited {slowest:.2f}s of the {docs_seconds:.2f}s cold docs request"
    )


@OWNED_PROXY_TIMEOUT
def test_concurrent_cold_openapi_requests_share_the_schema_later_requests_get(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_proxy_process(gateway, tmp_path, {}, client_timeout=DOCS_SECONDS) as owned,
        ThreadPoolExecutor(COLD_DOCS) as pool,
    ):
        cold: Final = tuple(pool.map(lambda _: owned.gateway.request("GET", "/openapi.json"), range(COLD_DOCS)))
        warm: Final = owned.gateway.request("GET", "/openapi.json")
    assert [response.status_code for response in (*cold, warm)] == [200] * (COLD_DOCS + 1)
    assert {response.content for response in cold} == {warm.content}
    assert "/v1/chat/completions" in object_value(_schema(warm)["paths"])


@OWNED_PROXY_TIMEOUT
def test_chat_completions_overlapping_a_cold_openapi_build_all_succeed(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_proxy_process(gateway, tmp_path, {}, client_timeout=DOCS_SECONDS) as owned,
        owned.gateway.scenario() as scenario,
        ThreadPoolExecutor(CHATS + 1) as pool,
    ):
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        docs: Final = pool.submit(owned.gateway.request, "GET", "/openapi.json")
        chats: Final = tuple(
            pool.map(
                lambda index: owned.gateway.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": f"overlapping docs build {index}"}],
                        "cache": {"no-cache": True},
                    },
                    key=key,
                ),
                range(CHATS),
            )
        )
        docs_response: Final = docs.result()
    assert [(response.status_code, response.text[:300]) for response in chats if response.status_code != 200] == []
    replies: Final = tuple(JSON_OBJECT.validate_json(response.content) for response in chats)
    assert {reply["object"] for reply in replies} == {"chat.completion"}
    assert all(list_value(reply["choices"]) for reply in replies), replies
    assert "/v1/chat/completions" in object_value(_schema(docs_response)["paths"])


@OWNED_PROXY_TIMEOUT
def test_docs_filtered_mode_serves_only_llm_routes(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {"DOCS_FILTERED": "True"}, client_timeout=DOCS_SECONDS) as owned:
        paths: Final = object_value(_schema(owned.gateway.request("GET", "/openapi.json"))["paths"])
    assert "/v1/chat/completions" in paths
    assert {"/key/generate", "/user/new", "/team/new"}.isdisjoint(paths), sorted(paths)


@OWNED_PROXY_TIMEOUT
def test_each_root_path_prefix_lists_itself_first_in_servers(gateway: Gateway, tmp_path: Path) -> None:
    prefixes: Final = ("/tenant-a", "/tenant-b", "/tenant-a")
    with owned_proxy_process(
        gateway, tmp_path, {"SERVER_ROOT_PATHS": "/tenant-a,/tenant-b"}, client_timeout=DOCS_SECONDS
    ) as owned:
        served: Final = tuple(
            (prefix, _servers(owned.gateway.request("GET", f"{prefix}/openapi.json"))) for prefix in prefixes
        )
    assert served == tuple((prefix, [{"url": prefix}]) for prefix in prefixes)


@OWNED_PROXY_TIMEOUT
def test_a_scalar_server_root_path_is_listed_once_in_servers(gateway: Gateway, tmp_path: Path) -> None:
    with owned_proxy_process(gateway, tmp_path, {"SERVER_ROOT_PATH": "/gateway"}, client_timeout=DOCS_SECONDS) as owned:
        servers: Final = _servers(owned.gateway.request("GET", "/gateway/openapi.json"))
    assert servers == [{"url": "/gateway"}]


@TWO_OWNED_PROXIES_TIMEOUT
def test_a_lazy_feature_loaded_during_a_cold_openapi_build_shows_its_live_routes_in_the_next_schema(
    gateway: Gateway, tmp_path: Path
) -> None:
    with owned_proxy_process(gateway, tmp_path, {LAZY_ROUTES_FLAG: "true"}, client_timeout=DOCS_SECONDS) as eager:
        live: Final = _warmed_feature_operations(eager.gateway.request("GET", "/openapi.json"))
    with (
        owned_proxy_process(
            gateway, tmp_path, {}, remove_environment=(LAZY_ROUTES_FLAG,), client_timeout=DOCS_SECONDS
        ) as lazy,
        ThreadPoolExecutor(1) as pool,
    ):
        docs: Final = pool.submit(lazy.gateway.request, "GET", "/openapi.json")
        warmed: Final = lazy.gateway.request("POST", f"/lazy/warm/{WARMED_FEATURE.name}")
        assert docs.result().status_code == 200
        after: Final = _warmed_feature_operations(lazy.gateway.request("GET", "/openapi.json"))
    assert warmed.status_code == 200, warmed.text
    assert live, "the eager proxy published no route of the warmed feature"
    assert after == live

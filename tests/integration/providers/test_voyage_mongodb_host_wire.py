import asyncio
import json
import os
import re
import signal
import socket
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import psutil
import pytest
import yaml
from integration._support.client import JSON_OBJECT, Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.forward_proxy import ForwardProxy, refusing_forward_proxy
from integration._support.process import owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from openai import APIStatusError, AsyncOpenAI, OpenAI
from pydantic import JsonValue, TypeAdapter

EMBEDDING_MODEL: Final = "voyage/voyage-3.5"
CONTEXTUAL_MODEL: Final = "voyage/voyage-context-3"
MULTIMODAL_MODEL: Final = "voyage/voyage-multimodal-3"
RERANK_MODEL: Final = "voyage/rerank-2.5"
MONGODB_KEY: Final = "al-synthetic-mongodb-issued-key"
VOYAGE_KEY: Final = "pa-synthetic-voyage-issued-key"
ENV_TOKEN: Final = "al-synthetic-env-token"
ENV_PRIMARY: Final = "pa-synthetic-env-primary"
ENV_SECONDARY: Final = "al-synthetic-env-secondary"
MONGODB_HOST: Final = "ai.mongodb.com:443"
VOYAGE_HOST: Final = "api.voyageai.com:443"
NO_CACHE: Final[dict[str, JsonValue]] = {"cache": {"no-cache": True}}
REFUSED_STATUS: Final = 500
REFUSED_MARKER: Final = "403"
OUTAGE_MARKER: Final = "APIConnectionError"
MONGODB_EMBEDDINGS: Final = "voyage-audit-mongodb-embeddings"
MONGODB_CONTEXTUAL: Final = "voyage-audit-mongodb-contextual"
MONGODB_MULTIMODAL: Final = "voyage-audit-mongodb-multimodal"
MONGODB_RERANK: Final = "voyage-audit-mongodb-rerank"
VOYAGE_EMBEDDINGS: Final = "voyage-audit-voyage-embeddings"
VOYAGE_RERANK: Final = "voyage-audit-voyage-rerank"
BLANK_EMBEDDINGS: Final = "voyage-audit-blank-key-embeddings"
BLANK_RERANK: Final = "voyage-audit-blank-key-rerank"
NULL_RERANK: Final = "voyage-audit-null-key-rerank"
SCRIPTED_CHAT: Final = "voyage-audit-scripted-chat"
TOKEN_EMBEDDINGS: Final = "voyage-audit-token-embeddings"
TOKEN_RERANK: Final = "voyage-audit-token-rerank"
TOKEN_YAML_RERANK: Final = "voyage-audit-token-yaml-rerank"
TOKEN_BLANK_EMBEDDINGS: Final = "voyage-audit-token-blank-embeddings"
TOKEN_BLANK_RERANK: Final = "voyage-audit-token-blank-rerank"
TOKEN_NULL_RERANK: Final = "voyage-audit-token-null-rerank"
PRECEDENCE_EMBEDDINGS: Final = "voyage-audit-precedence-embeddings"
PRECEDENCE_RERANK: Final = "voyage-audit-precedence-rerank"
PRECEDENCE_EXPLICIT_EMBEDDINGS: Final = "voyage-audit-precedence-explicit-embeddings"
PRECEDENCE_EXPLICIT_RERANK: Final = "voyage-audit-precedence-explicit-rerank"
VECTOR: Final = [0.1, 0.2, 0.3]
_STARTED_WORKER: Final = re.compile(r"Started server process \[(\d+)\]")
_PROXY_VARIABLES: Final = frozenset({"HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"})
_JSON_OBJECTS: Final = TypeAdapter(list[dict[str, JsonValue]])
_WORKER_PIDS: Final = TypeAdapter(list[int])
_load_yaml: Final[Callable[[str], object]] = yaml.safe_load
_OWNED_PROXY_CELL_SECONDS: Final = 2 * max(30.0, float(os.environ.get("INTEGRATION_PROXY_READY_SECONDS") or 70)) + 120
pytestmark: Final = pytest.mark.timeout(_OWNED_PROXY_CELL_SECONDS)


def _deployment(name: str, model: str, mode: str, **litellm_params: JsonValue) -> dict[str, JsonValue]:
    return {
        "model_name": name,
        "litellm_params": {"model": model, **litellm_params},
        "model_info": {"id": name, "mode": mode},
    }


def _isolated_deployments(upstream_url: str) -> list[dict[str, JsonValue]]:
    return [
        _deployment(MONGODB_EMBEDDINGS, EMBEDDING_MODEL, "embedding", api_key=MONGODB_KEY),
        _deployment(MONGODB_CONTEXTUAL, CONTEXTUAL_MODEL, "embedding", api_key=MONGODB_KEY),
        _deployment(MONGODB_MULTIMODAL, MULTIMODAL_MODEL, "embedding", api_key=MONGODB_KEY),
        _deployment(MONGODB_RERANK, RERANK_MODEL, "rerank", api_key=MONGODB_KEY),
        _deployment(VOYAGE_EMBEDDINGS, EMBEDDING_MODEL, "embedding", api_key=VOYAGE_KEY),
        _deployment(VOYAGE_RERANK, RERANK_MODEL, "rerank", api_key=VOYAGE_KEY),
        _deployment(BLANK_EMBEDDINGS, EMBEDDING_MODEL, "embedding", api_key=""),
        _deployment(BLANK_RERANK, RERANK_MODEL, "rerank", api_key=""),
        _deployment(NULL_RERANK, RERANK_MODEL, "rerank", api_key=None),
        _deployment(
            SCRIPTED_CHAT,
            "openai/gpt-4o-mini",
            "chat",
            api_key="integration-provider-key",
            api_base=f"{upstream_url}/v1",
        ),
    ]


def _token_only_deployments() -> list[dict[str, JsonValue]]:
    return [
        _deployment(TOKEN_EMBEDDINGS, EMBEDDING_MODEL, "embedding"),
        _deployment(TOKEN_RERANK, RERANK_MODEL, "rerank"),
        _deployment(TOKEN_YAML_RERANK, RERANK_MODEL, "rerank", api_key="os.environ/VOYAGE_AI_TOKEN"),
        _deployment(TOKEN_BLANK_EMBEDDINGS, EMBEDDING_MODEL, "embedding", api_key=""),
        _deployment(TOKEN_BLANK_RERANK, RERANK_MODEL, "rerank", api_key=""),
        _deployment(TOKEN_NULL_RERANK, RERANK_MODEL, "rerank", api_key=None),
    ]


def _precedence_deployments() -> list[dict[str, JsonValue]]:
    return [
        _deployment(PRECEDENCE_EMBEDDINGS, EMBEDDING_MODEL, "embedding"),
        _deployment(PRECEDENCE_RERANK, RERANK_MODEL, "rerank"),
        _deployment(PRECEDENCE_EXPLICIT_EMBEDDINGS, EMBEDDING_MODEL, "embedding", api_key=MONGODB_KEY),
        _deployment(PRECEDENCE_EXPLICIT_RERANK, RERANK_MODEL, "rerank", api_key=MONGODB_KEY),
    ]


def _write_config(directory: Path, name: str, model_list: list[dict[str, JsonValue]]) -> Path:
    base: Final = JSON_OBJECT.validate_python(_load_yaml(Path("tests/integration/proxy_config.yaml").read_text()))
    configuration: Final = {
        **base,
        "model_list": model_list,
        "router_settings": {"disable_cooldowns": True, "num_retries": 0},
    }
    path: Final = directory / f"{name}.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


def _scrubbed() -> tuple[str, ...]:
    return tuple(name for name in os.environ if name.upper().startswith("VOYAGE_") or name.upper() in _PROXY_VARIABLES)


def _forward_environment(forward_url: str, **extra: str) -> dict[str, str]:
    return {"HTTPS_PROXY": forward_url, "NO_PROXY": "127.0.0.1,localhost", **extra}


@pytest.fixture(scope="module")
def forward() -> Iterator[ForwardProxy]:
    with refusing_forward_proxy() as proxy:
        yield proxy


@pytest.fixture(scope="module")
def module_gateway() -> Iterator[Gateway]:
    with gateway_from_environment() as value:
        yield value


@pytest.fixture(scope="module")
def isolated(
    module_gateway: Gateway, forward: ForwardProxy, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("voyage-isolated")
    config: Final = _write_config(directory, "isolated", _isolated_deployments(module_gateway.upstream_url))
    with owned_proxy(
        module_gateway,
        directory,
        _forward_environment(forward.url),
        config=config,
        remove_environment=_scrubbed(),
        workers=2,
    ) as candidate:
        yield candidate


@pytest.fixture(scope="module")
def token_only(
    module_gateway: Gateway, forward: ForwardProxy, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("voyage-token-only")
    config: Final = _write_config(directory, "token-only", _token_only_deployments())
    with owned_proxy(
        module_gateway,
        directory,
        _forward_environment(forward.url, VOYAGE_AI_TOKEN=ENV_TOKEN),
        config=config,
        remove_environment=_scrubbed(),
    ) as candidate:
        yield candidate


@pytest.fixture(scope="module")
def precedence(
    module_gateway: Gateway, forward: ForwardProxy, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("voyage-precedence")
    config: Final = _write_config(directory, "precedence", _precedence_deployments())
    with owned_proxy(
        module_gateway,
        directory,
        _forward_environment(
            forward.url,
            VOYAGE_API_KEY=ENV_PRIMARY,
            VOYAGE_AI_API_KEY=ENV_SECONDARY,
            VOYAGE_AI_TOKEN=ENV_TOKEN,
        ),
        config=config,
        remove_environment=_scrubbed(),
    ) as candidate:
        yield candidate


def _embedding_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "input": f"host selection {uuid.uuid4().hex}", **NO_CACHE, **extra}


def _rerank_body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": model,
        "query": f"host selection {uuid.uuid4().hex}",
        "documents": ["first document", "second document"],
        **extra,
    }


def _embed(candidate: Gateway, model: str, **extra: JsonValue) -> httpx.Response:
    return candidate.request("POST", "/v1/embeddings", _embedding_body(model, **extra))


def _rerank(candidate: Gateway, model: str, path: str = "/v1/rerank", **extra: JsonValue) -> httpx.Response:
    return candidate.request("POST", path, _rerank_body(model, **extra))


def _error_message(call_id: str) -> str:
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    metadata: Final = rows[0]["metadata"]
    parsed: Final = json.loads(metadata) if isinstance(metadata, str) else object_value(metadata)
    return str(object_value(parsed["error_information"])["error_message"])


def _assert_refused(response: httpx.Response) -> None:
    assert response.status_code == REFUSED_STATUS, response.text
    assert REFUSED_MARKER in response.text, response.text
    assert REFUSED_MARKER in _error_message(response.headers["x-litellm-call-id"]), response.text


def _assert_dialed(forward: ForwardProxy, response: httpx.Response, host: str) -> None:
    _assert_refused(response)
    assert forward.targets() == (host,), response.text


def test_scripted_chat_keeps_serving_behind_the_forward_proxy(isolated: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    reply: Final = isolated.chat(SCRIPTED_CHAT, text=f"forward proxy control {uuid.uuid4().hex}")
    rows: Final = eventually(
        lambda: read_rows('SELECT model FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (str(reply["id"]),)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    assert rows[0]["model"] == "openai/gpt-4o-mini", rows
    assert forward.targets() == ()


def test_mongodb_key_embeddings_dial_ai_mongodb_openai_sdk_sync(isolated: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    with OpenAI(
        api_key=isolated.key,
        base_url=str(isolated.client.base_url).rstrip("/") + "/v1",
        max_retries=0,
        http_client=httpx.Client(timeout=15, trust_env=False),
    ) as client:
        with pytest.raises(APIStatusError) as caught:
            client.embeddings.create(
                model=MONGODB_EMBEDDINGS, input=f"sdk sync {uuid.uuid4().hex}", extra_body=NO_CACHE
            )
    assert caught.value.status_code == REFUSED_STATUS, caught.value.message
    assert REFUSED_MARKER in caught.value.message, caught.value.message
    assert forward.targets() == (MONGODB_HOST,)


async def test_mongodb_key_embeddings_dial_ai_mongodb_openai_sdk_async(
    isolated: Gateway, forward: ForwardProxy
) -> None:
    forward.drain()
    async with AsyncOpenAI(
        api_key=isolated.key,
        base_url=str(isolated.client.base_url).rstrip("/") + "/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(timeout=15, trust_env=False),
    ) as client:
        with pytest.raises(APIStatusError) as caught:
            await client.embeddings.create(
                model=MONGODB_EMBEDDINGS, input=f"sdk async {uuid.uuid4().hex}", extra_body=NO_CACHE
            )
    assert caught.value.status_code == REFUSED_STATUS, caught.value.message
    assert REFUSED_MARKER in caught.value.message, caught.value.message
    assert forward.targets() == (MONGODB_HOST,)


@pytest.mark.parametrize("deployment", (MONGODB_CONTEXTUAL, MONGODB_MULTIMODAL))
def test_mongodb_key_other_embedding_shapes_dial_ai_mongodb(
    isolated: Gateway, forward: ForwardProxy, deployment: str
) -> None:
    forward.drain()
    _assert_dialed(forward, _embed(isolated, deployment), MONGODB_HOST)


@pytest.mark.parametrize("path", ("/v1/rerank", "/rerank", "/v2/rerank"))
def test_mongodb_key_rerank_dials_ai_mongodb(isolated: Gateway, forward: ForwardProxy, path: str) -> None:
    forward.drain()
    _assert_dialed(forward, _rerank(isolated, MONGODB_RERANK, path), MONGODB_HOST)


def test_voyage_key_embeddings_still_dial_api_voyageai(isolated: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    _assert_dialed(forward, _embed(isolated, VOYAGE_EMBEDDINGS), VOYAGE_HOST)


def test_voyage_key_rerank_still_dials_api_voyageai(isolated: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    _assert_dialed(forward, _rerank(isolated, VOYAGE_RERANK), VOYAGE_HOST)


def test_request_body_mongodb_key_overrides_voyage_deployment_host(isolated: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    _assert_dialed(forward, _embed(isolated, VOYAGE_EMBEDDINGS, api_key=MONGODB_KEY), MONGODB_HOST)


@pytest.mark.parametrize("deployment", (MONGODB_EMBEDDINGS, MONGODB_RERANK))
def test_health_check_dials_ai_mongodb_for_mongodb_key(
    isolated: Gateway, forward: ForwardProxy, deployment: str
) -> None:
    forward.drain()
    response: Final = isolated.request("GET", "/health", params={"model": deployment})
    assert response.status_code == 503, response.text
    report: Final = JSON_OBJECT.validate_json(response.content)
    assert report["unhealthy_count"] == 1 and report["healthy_count"] == 0, report
    unhealthy: Final = _JSON_OBJECTS.validate_python(report["unhealthy_endpoints"])
    assert len(unhealthy) == 1 and REFUSED_MARKER in str(unhealthy[0]["error"]), report
    assert forward.targets() == (MONGODB_HOST,), report


@pytest.mark.parametrize(
    ("mode", "model"), (("embedding", EMBEDDING_MODEL), ("rerank", RERANK_MODEL)), ids=("embedding", "rerank")
)
def test_test_connection_dials_ai_mongodb_for_mongodb_key(
    isolated: Gateway, forward: ForwardProxy, mode: str, model: str
) -> None:
    forward.drain()
    report: Final = isolated.post(
        "/health/test_connection", {"litellm_params": {"model": model, "api_key": MONGODB_KEY}, "mode": mode}
    )
    assert report["status"] == "error", report
    assert REFUSED_MARKER in json.dumps(report), report
    assert forward.targets() == (MONGODB_HOST,), report


@pytest.mark.parametrize("api_key", (5, ["al-list-member"]), ids=("int", "list"))
def test_non_string_request_body_key_is_rejected_by_validation_before_any_dial(
    isolated: Gateway, forward: ForwardProxy, api_key: JsonValue
) -> None:
    forward.drain()
    response: Final = _embed(isolated, MONGODB_EMBEDDINGS, api_key=api_key)
    assert response.status_code == 500, response.text
    assert "LiteLLM_Params" in response.text and "api_key" in response.text, response.text
    assert forward.targets() == ()


def test_empty_request_body_key_falls_back_to_api_voyageai(isolated: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    _assert_dialed(forward, _embed(isolated, MONGODB_EMBEDDINGS, api_key=""), VOYAGE_HOST)


def test_five_kilobyte_mongodb_request_body_key_dials_ai_mongodb(isolated: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    _assert_dialed(forward, _embed(isolated, VOYAGE_EMBEDDINGS, api_key=MONGODB_KEY + "x" * 5000), MONGODB_HOST)


def test_duplicate_request_body_key_counts_once(isolated: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    raw: Final = (
        '{"model": "%s", "input": "duplicate key %s", "cache": {"no-cache": true}, "api_key": "%s", "api_key": "%s"}'
        % (VOYAGE_EMBEDDINGS, uuid.uuid4().hex, MONGODB_KEY, MONGODB_KEY)
    )
    response: Final = isolated.client.post(
        "/v1/embeddings",
        content=raw.encode(),
        headers={"Authorization": f"Bearer {isolated.key}", "content-type": "application/json"},
    )
    _assert_dialed(forward, response, MONGODB_HOST)


def test_blank_deployment_key_without_env_dials_api_voyageai_for_embeddings(
    isolated: Gateway, forward: ForwardProxy
) -> None:
    forward.drain()
    _assert_dialed(forward, _embed(isolated, BLANK_EMBEDDINGS), VOYAGE_HOST)


@pytest.mark.parametrize("deployment", (BLANK_RERANK, NULL_RERANK), ids=("blank", "null"))
def test_rerank_without_any_key_fails_before_dialing(isolated: Gateway, forward: ForwardProxy, deployment: str) -> None:
    forward.drain()
    response: Final = _rerank(isolated, deployment)
    assert response.status_code == 500, response.text
    assert "Voyage AI API key is required" in response.text, response.text
    assert forward.targets() == ()


@dataclass(frozen=True, slots=True)
class _Probe:
    host: str
    response: httpx.Response


def _burst_plan(count: int) -> tuple[tuple[str, str, str], ...]:
    shapes: Final = (
        ("/v1/embeddings", MONGODB_EMBEDDINGS, MONGODB_HOST),
        ("/v1/rerank", MONGODB_RERANK, MONGODB_HOST),
        ("/v1/embeddings", VOYAGE_EMBEDDINGS, VOYAGE_HOST),
        ("/v1/rerank", VOYAGE_RERANK, VOYAGE_HOST),
    )
    return tuple(shapes[index % len(shapes)] for index in range(count))


async def _burst(base_url: str, key: str, count: int, *, tolerate_transport_errors: bool = False) -> tuple[_Probe, ...]:
    async def one(client: httpx.AsyncClient, path: str, deployment: str, host: str) -> _Probe:
        body: Final = _embedding_body(deployment) if path == "/v1/embeddings" else _rerank_body(deployment)
        response: Final = await client.post(path, json=body, headers={"Authorization": f"Bearer {key}"})
        return _Probe(host, response)

    async with httpx.AsyncClient(base_url=base_url, timeout=30, trust_env=False) as client:
        results: Final = await asyncio.gather(
            *(one(client, path, deployment, host) for path, deployment, host in _burst_plan(count)),
            return_exceptions=tolerate_transport_errors,
        )
    for result in results:
        assert not isinstance(result, BaseException) or isinstance(result, httpx.TransportError), repr(result)
    return tuple(result for result in results if isinstance(result, _Probe))


def _host_counts(targets: tuple[str, ...]) -> dict[str, int]:
    return {host: targets.count(host) for host in sorted(set(targets))}


async def test_concurrent_mixed_keys_each_dial_their_own_host(isolated: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    served: Final = await _burst(str(isolated.client.base_url), isolated.key, 24)
    assert len(served) == 24
    for probe in served:
        _assert_refused(probe.response)
    assert _host_counts(forward.targets()) == {MONGODB_HOST: 12, VOYAGE_HOST: 12}


def test_voyage_ai_token_alone_routes_embeddings_to_ai_mongodb(token_only: Gateway, forward: ForwardProxy) -> None:
    forward.drain()
    _assert_dialed(forward, _embed(token_only, TOKEN_EMBEDDINGS), MONGODB_HOST)


@pytest.mark.parametrize(
    "deployment", (TOKEN_RERANK, TOKEN_YAML_RERANK, TOKEN_NULL_RERANK), ids=("missing", "yaml-env", "null")
)
def test_voyage_ai_token_alone_routes_rerank_to_ai_mongodb(
    token_only: Gateway, forward: ForwardProxy, deployment: str
) -> None:
    forward.drain()
    _assert_dialed(forward, _rerank(token_only, deployment), MONGODB_HOST)


@pytest.mark.parametrize(
    ("path", "deployment"),
    (("/v1/embeddings", TOKEN_BLANK_EMBEDDINGS), ("/v1/rerank", TOKEN_BLANK_RERANK)),
    ids=("embeddings", "rerank"),
)
def test_blank_deployment_key_falls_through_to_voyage_ai_token(
    token_only: Gateway, forward: ForwardProxy, path: str, deployment: str
) -> None:
    forward.drain()
    response: Final = (
        _embed(token_only, deployment) if path == "/v1/embeddings" else _rerank(token_only, deployment, path)
    )
    _assert_dialed(forward, response, MONGODB_HOST)


@pytest.mark.parametrize(
    ("path", "deployment"),
    (("/v1/embeddings", PRECEDENCE_EMBEDDINGS), ("/v1/rerank", PRECEDENCE_RERANK)),
    ids=("embeddings", "rerank"),
)
def test_voyage_api_key_wins_over_mongodb_fallback_env(
    precedence: Gateway, forward: ForwardProxy, path: str, deployment: str
) -> None:
    forward.drain()
    response: Final = (
        _embed(precedence, deployment) if path == "/v1/embeddings" else _rerank(precedence, deployment, path)
    )
    _assert_dialed(forward, response, VOYAGE_HOST)


@pytest.mark.parametrize(
    ("path", "deployment"),
    (("/v1/embeddings", PRECEDENCE_EXPLICIT_EMBEDDINGS), ("/v1/rerank", PRECEDENCE_EXPLICIT_RERANK)),
    ids=("embeddings", "rerank"),
)
def test_explicit_mongodb_deployment_key_wins_over_voyage_env(
    precedence: Gateway, forward: ForwardProxy, path: str, deployment: str
) -> None:
    forward.drain()
    response: Final = (
        _embed(precedence, deployment) if path == "/v1/embeddings" else _rerank(precedence, deployment, path)
    )
    _assert_dialed(forward, response, MONGODB_HOST)


def _embedding_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.headers["authorization"] == f"Bearer {MONGODB_KEY}", request.headers
    body: Final = JSON_OBJECT.validate_json(request.body)
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"object": "embedding", "embedding": VECTOR, "index": 0}],
                "model": str(body["model"]),
                "usage": {"total_tokens": 7},
            }
        ).encode()
    )


def _rerank_peer(request: Request) -> Reply:
    assert request.method == "POST", request.method
    assert request.headers["authorization"] == f"Bearer {MONGODB_KEY}", request.headers
    return Reply(
        body=json.dumps(
            {
                "object": "list",
                "data": [{"index": 1, "relevance_score": 0.9}, {"index": 0, "relevance_score": 0.1}],
                "model": "rerank-2.5",
                "usage": {"total_tokens": 11},
            }
        ).encode()
    )


def _spend_api_base(call_id: str) -> str:
    rows: Final = eventually(
        lambda: read_rows('SELECT api_base FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return str(rows[0]["api_base"])


@pytest.mark.parametrize(
    ("model", "target"),
    (
        (EMBEDDING_MODEL, "/embeddings"),
        (CONTEXTUAL_MODEL, "/contextualizedembeddings"),
        (MULTIMODAL_MODEL, "/multimodalembeddings"),
    ),
    ids=("embeddings", "contextual", "multimodal"),
)
def test_explicit_api_base_keeps_mongodb_key_embeddings_on_that_base(gateway: Gateway, model: str, target: str) -> None:
    with wire_server(_embedding_peer) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(model=model, api_key=MONGODB_KEY, api_base=wire.url)
        response: Final = _embed(gateway, deployment)
        assert response.status_code == 200, response.text
        data: Final = _JSON_OBJECTS.validate_python(JSON_OBJECT.validate_json(response.content)["data"])
        assert data[0]["embedding"] == VECTOR, response.text
        received: Final = wire.drain()
        assert tuple(request.target for request in received) == (target,), received
        assert _spend_api_base(response.headers["x-litellm-call-id"]).startswith(wire.url), response.text


@pytest.mark.parametrize("suffix", ("", "/v1"), ids=("bare", "v1"))
def test_explicit_api_base_keeps_mongodb_key_rerank_on_that_base(gateway: Gateway, suffix: str) -> None:
    with wire_server(_rerank_peer) as wire, gateway.scenario() as scenario:
        deployment: Final = scenario.model(model=RERANK_MODEL, api_key=MONGODB_KEY, api_base=wire.url + suffix)
        response: Final = _rerank(gateway, deployment)
        assert response.status_code == 200, response.text
        payload: Final = JSON_OBJECT.validate_json(response.content)
        results: Final = _JSON_OBJECTS.validate_python(payload["results"])
        assert [result["index"] for result in results] == [1, 0], response.text
        received: Final = wire.drain()
        assert tuple(request.target for request in received) == ("/v1/rerank",), received
        assert _spend_api_base(str(payload["id"])).startswith(wire.url), response.text


def test_public_provider_fields_name_voyage_as_mongodb(gateway: Gateway) -> None:
    response: Final = gateway.request("GET", "/public/providers/fields")
    assert response.status_code == 200, response.text
    voyage: Final = [
        entry for entry in _JSON_OBJECTS.validate_json(response.content) if entry["litellm_provider"] == "voyage"
    ]
    assert [entry["provider_display_name"] for entry in voyage] == ["VoyageAI by MongoDB"], response.text


def test_public_endpoints_name_voyage_as_mongodb(gateway: Gateway) -> None:
    response: Final = gateway.request("GET", "/public/endpoints")
    assert response.status_code == 200, response.text
    endpoints: Final = _JSON_OBJECTS.validate_python(JSON_OBJECT.validate_json(response.content)["endpoints"])
    names: Final = tuple(_voyage_display_names(endpoints))
    assert names and set(names) == {"VoyageAI by MongoDB"}, response.text


def _voyage_display_names(endpoints: list[dict[str, JsonValue]]) -> Iterator[str]:
    for endpoint in endpoints:
        for provider in _JSON_OBJECTS.validate_python(endpoint["providers"]):
            if provider["slug"] == "voyage":
                yield str(provider["display_name"])


def _served_call_ids(served: tuple[_Probe, ...]) -> tuple[str, ...]:
    return tuple(probe.response.headers["x-litellm-call-id"] for probe in served)


def _spend_rows(call_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (call_id,))


def _reserve_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        address: Final[Callable[[], tuple[str, int]]] = reserve.getsockname
        return address()[1]


async def test_forward_proxy_outage_mid_traffic_recovers_with_the_right_hosts(gateway: Gateway, tmp_path: Path) -> None:
    port: Final = _reserve_port()
    config: Final = _write_config(tmp_path, "outage", _isolated_deployments(gateway.upstream_url))
    with owned_proxy(
        gateway,
        tmp_path,
        _forward_environment(f"http://127.0.0.1:{port}"),
        config=config,
        remove_environment=_scrubbed(),
        workers=2,
    ) as candidate:
        with refusing_forward_proxy(port=port) as before:
            first: Final = await _burst(str(candidate.client.base_url), candidate.key, 12)
            before_counts: Final = _host_counts(before.targets())
        during: Final = await _burst(str(candidate.client.base_url), candidate.key, 12)
        liveliness: Final = candidate.request("GET", "/health/liveliness")
        assert liveliness.status_code == 200, liveliness.text
        with refusing_forward_proxy(port=port) as after:
            third: Final = await _burst(str(candidate.client.base_url), candidate.key, 12)
            after_counts: Final = _host_counts(after.targets())
        assert len(first) == len(during) == len(third) == 12
        for probe in (*first, *third):
            _assert_refused(probe.response)
        for probe in during:
            assert probe.response.status_code == REFUSED_STATUS, probe.response.text
            assert OUTAGE_MARKER in probe.response.text, probe.response.text
            assert OUTAGE_MARKER in _error_message(probe.response.headers["x-litellm-call-id"])
        assert before_counts == {MONGODB_HOST: 6, VOYAGE_HOST: 6}
        assert after_counts == {MONGODB_HOST: 6, VOYAGE_HOST: 6}


async def test_worker_sigkill_mid_traffic_leaves_the_sibling_routing_by_key(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _write_config(tmp_path, "sigkill", _isolated_deployments(gateway.upstream_url))
    with (
        refusing_forward_proxy() as forward,
        owned_proxy_process(
            gateway,
            tmp_path,
            _forward_environment(forward.url),
            config=config,
            remove_environment=_scrubbed(),
            workers=2,
        ) as owned,
    ):
        candidate: Final = owned.gateway
        workers: Final = eventually(
            lambda: _WORKER_PIDS.validate_python(_STARTED_WORKER.findall(owned.log.read_text())),
            lambda pids: len(pids) == 2,
            seconds=60,
        )
        burst: Final = asyncio.create_task(
            _burst(str(candidate.client.base_url), candidate.key, 20, tolerate_transport_errors=True)
        )
        await asyncio.to_thread(eventually, lambda: forward.received.qsize(), lambda size: size >= 4, 60)
        victim: Final = psutil.Process(workers[0])
        victim.suspend()
        victim.send_signal(signal.SIGKILL)
        served: Final = await burst
        for probe in served:
            assert probe.response.status_code == REFUSED_STATUS, probe.response.text
        assert set(forward.targets()) <= {MONGODB_HOST, VOYAGE_HOST}
        follow_up: Final = _rerank(candidate, MONGODB_RERANK)
        _assert_dialed(forward, follow_up, MONGODB_HOST)
        duplicates: Final = tuple(call_id for call_id in _served_call_ids(served) if len(_spend_rows(call_id)) > 1)
        assert duplicates == (), duplicates

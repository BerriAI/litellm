from __future__ import annotations

import signal
import socket
import threading
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
import psutil
import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, object_value
from integration._support.process import OwnedProxy, owned_proxy, owned_proxy_process
from integration.authorization._guardrail_opt_out import upstream_observations
from pydantic import JsonValue

PROXY_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
REMOVE_OPENAI_API_BASE: Final = ("OPENAI_API_BASE",)
JsonObject: TypeAlias = dict[str, JsonValue]


def _json_array(*values: JsonValue) -> JsonValue:
    return [*values]  # mutable-ok: JSON payloads and YAML sequences require list values


def _permission_for_stores(*store_ids: str) -> JsonObject:
    permission: Final[JsonObject] = {"vector_stores": _json_array(*store_ids)}
    return permission


def _worker_model_configuration(model_name: str, api_base: str) -> JsonObject:
    parameters: Final[JsonObject] = {
        "model": "openai/gpt-4o-mini",
        "api_key": "os.environ/OPENAI_API_KEY",
        "api_base": api_base,
    }
    configuration: Final[JsonObject] = {"model_name": model_name, "litellm_params": parameters}
    return configuration


def _no_registry_config(directory: Path, model: tuple[str, str] | None = None) -> Path:
    config: Final = object_value(yaml.safe_load(PROXY_CONFIG.read_text()))
    config_without_registry: Final[Mapping[str, JsonValue]] = MappingProxyType(
        {name: value for name, value in config.items() if name != "vector_store_registry"}
    )
    model_list: Final[JsonValue] = _json_array() if model is None else _json_array(_worker_model_configuration(*model))
    yaml_config: Final[JsonObject] = {**config_without_registry, "model_list": model_list}
    path: Final = directory / "proxy_no_vector_store_registry.yaml"
    path.write_text(yaml.safe_dump(yaml_config))
    return path


def _openai_environment(gateway: Gateway, api_base: str | None = None) -> Mapping[str, str]:
    return MappingProxyType(
        {"OPENAI_BASE_URL": api_base or gateway.upstream_url, "OPENAI_API_KEY": "synthetic-openai-key"}
    )


def _key_for_scope(scenario: Scenario, model: str, scope: Literal["key", "team"], store_id: str) -> str:
    if scope == "key":
        return scenario.key(models=_json_array(model), object_permission=_permission_for_stores(store_id))
    team: Final = scenario.team(models=_json_array(model), object_permission=_permission_for_stores(store_id))
    return scenario.key(team_id=team, models=_json_array(model))


def _rag_query(
    gateway: Gateway,
    model: str,
    store_id: str,
    marker: str,
    key: str,
) -> httpx.Response:
    body: Final[JsonObject] = {
        "model": model,
        "messages": _json_array({"role": "user", "content": marker}),
        "retrieval_config": {
            "vector_store_id": store_id,
            "custom_llm_provider": "openai",
            "top_k": 1,
        },
    }
    return gateway.request("POST", "/v1/rag/query", body, key=key)


def _searches_in(
    observations: tuple[Mapping[str, JsonValue], ...], marker: str, store_id: str
) -> tuple[Mapping[str, JsonValue], ...]:
    path: Final = f"/vector_stores/{store_id}/search"
    return tuple(
        observation
        for observation in observations
        if observation["path"] == path and marker in str(observation["body"])
    )


def _searches_for_marker(gateway: Gateway, marker: str, store_id: str) -> tuple[Mapping[str, JsonValue], ...]:
    return _searches_in(upstream_observations(gateway), marker, store_id)


def _closed_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port: Final = int(listener.getsockname()[1])
    return port


def _worker_processes(owned: OwnedProxy) -> tuple[psutil.Process, ...]:
    return tuple(
        child
        for child in psutil.Process(owned.process.pid).children(recursive=True)
        if child.is_running() and any("spawn_main" in argument for argument in child.cmdline())
    )


def _send_rag_query(
    gateway: Gateway,
    model: str,
    store_id: str,
    marker: str,
    key: str,
    started: threading.Event,
) -> httpx.Response | httpx.TransportError:
    started.set()
    try:
        return _rag_query(gateway, model, store_id, marker, key)
    except httpx.TransportError as error:
        return error


@pytest.mark.parametrize("scope", ("team", "key"))
def test_no_registry_rag_query_denies_unregistered_store_when_allowlist_excludes(
    gateway: Gateway,
    tmp_path: Path,
    scope: Literal["team", "key"],
) -> None:
    config: Final = _no_registry_config(tmp_path)
    with owned_proxy(
        gateway,
        tmp_path,
        _openai_environment(gateway),
        config=config,
        remove_environment=REMOVE_OPENAI_API_BASE,
        workers=2,
    ) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model()
            store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
            key: Final = _key_for_scope(scenario, model, scope, "vs_some_other_store")
            marker: Final = f"lit5610 no registry denied {scope} {uuid.uuid4().hex}"

            response: Final = _rag_query(candidate, model, store_id, marker, key)
            searches: Final = _searches_for_marker(gateway, marker, store_id)
            error_type: Final = (
                "team_vector_store_access_denied" if scope == "team" else "key_vector_store_access_denied"
            )

            assert response.status_code == 401, f"{response.text}; scripted_upstream_searches={searches!r}"
            assert response.json()["error"]["type"] == error_type, response.text
            assert searches == ()


def test_no_registry_rag_query_allows_team_allowlisted_unregistered_store(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _no_registry_config(tmp_path)
    with owned_proxy(
        gateway,
        tmp_path,
        _openai_environment(gateway),
        config=config,
        remove_environment=REMOVE_OPENAI_API_BASE,
        workers=2,
    ) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model()
            store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
            key: Final = _key_for_scope(scenario, model, "team", store_id)
            marker: Final = f"lit5610 no registry allowed {uuid.uuid4().hex}"

            response: Final = _rag_query(candidate, model, store_id, marker, key)
            assert response.status_code == 200, response.text

            searches: Final = _searches_for_marker(gateway, marker, store_id)
            assert len(searches) == 1, searches
            assert marker in str(object_value(searches[0]["body"])["query"]), searches


def test_no_registry_rag_query_denies_unregistered_5kb_store(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _no_registry_config(tmp_path)
    with owned_proxy(
        gateway,
        tmp_path,
        _openai_environment(gateway),
        config=config,
        remove_environment=REMOVE_OPENAI_API_BASE,
        workers=2,
    ) as candidate:
        with candidate.scenario() as scenario:
            model: Final = scenario.model()
            store_id: Final = "vs_unregistered_" + "x" * 5_000
            key: Final = _key_for_scope(scenario, model, "team", "vs_some_other_store")
            marker: Final = f"lit5610 no registry long id denied {uuid.uuid4().hex}"

            response: Final = _rag_query(candidate, model, store_id, marker, key)
            searches: Final = _searches_for_marker(gateway, marker, store_id)

            assert response.status_code == 401, f"{response.text}; scripted_upstream_searches={searches!r}"
            assert response.json()["error"]["type"] == "team_vector_store_access_denied", response.text
            assert searches == ()


def test_no_registry_rag_denial_skips_unavailable_vector_store_provider(gateway: Gateway, tmp_path: Path) -> None:
    config: Final = _no_registry_config(tmp_path)
    closed_url: Final = f"http://127.0.0.1:{_closed_local_port()}"
    with owned_proxy_process(
        gateway,
        tmp_path,
        _openai_environment(gateway, closed_url),
        config=config,
        remove_environment=REMOVE_OPENAI_API_BASE,
        workers=2,
    ) as owned:
        candidate: Final = owned.gateway
        with candidate.scenario() as scenario:
            model: Final = scenario.model()
            allowed_store: Final = f"vs_unregistered_allowed_{uuid.uuid4().hex}"
            denied_store: Final = f"vs_unregistered_denied_{uuid.uuid4().hex}"
            allowed_key: Final = _key_for_scope(scenario, model, "key", allowed_store)
            denied_key: Final = _key_for_scope(scenario, model, "key", "vs_some_other_store")
            allowed_marker: Final = f"lit5610 no registry closed port allowed {uuid.uuid4().hex}"
            denied_marker: Final = f"lit5610 no registry closed port denied {uuid.uuid4().hex}"

            allowed: Final = _rag_query(candidate, model, allowed_store, allowed_marker, allowed_key)
            denied: Final = _rag_query(candidate, model, denied_store, denied_marker, denied_key)
            allowed_detail: Final = object_value(object_value(allowed.json()).get("detail"))
            allowed_error: Final = allowed_detail.get("error")
            liveliness: Final = candidate.request("GET", "/health/liveliness")
            plain_chat_body: Final[JsonObject] = {
                "model": model,
                "messages": _json_array({"role": "user", "content": f"plain chat {uuid.uuid4().hex}"}),
            }
            plain_chat: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                plain_chat_body,
                key=allowed_key,
            )

            assert not 200 <= allowed.status_code < 300, allowed.text
            assert isinstance(allowed_error, str) and "Cannot connect to host" in allowed_error, allowed.text
            assert denied.status_code == 401, denied.text
            assert denied.json()["error"]["type"] == "key_vector_store_access_denied", denied.text
            assert liveliness.status_code == 200, liveliness.text
            assert plain_chat.status_code == 200, plain_chat.text


def test_no_registry_rag_allowlist_survives_worker_kill_during_burst(gateway: Gateway, tmp_path: Path) -> None:
    model: Final = f"lit5610-worker-{uuid.uuid4().hex}"
    config: Final = _no_registry_config(tmp_path, (model, f"{gateway.upstream_url}/v1"))
    with owned_proxy_process(
        gateway,
        tmp_path,
        _openai_environment(gateway),
        config=config,
        remove_environment=REMOVE_OPENAI_API_BASE,
        workers=2,
    ) as owned:
        candidate: Final = owned.gateway
        with candidate.scenario() as scenario:
            store_id: Final = f"vs_unregistered_{uuid.uuid4().hex}"
            allowed_key: Final = _key_for_scope(scenario, model, "key", store_id)
            denied_key: Final = _key_for_scope(scenario, model, "team", "vs_some_other_store")
            burst_markers: Final = tuple(
                f"lit5610 no registry worker burst {index} {uuid.uuid4().hex}" for index in range(30)
            )
            burst_keys: Final = tuple(allowed_key if index % 2 == 0 else denied_key for index in range(30))
            started: Final = tuple(threading.Event() for _ in range(30))
            workers: Final = eventually(
                lambda: _worker_processes(owned), lambda processes: len(processes) == 2, seconds=30
            )
            victim: Final = workers[0]

            with ThreadPoolExecutor(max_workers=30) as executor:
                futures: Final = tuple(
                    executor.submit(_send_rag_query, candidate, model, store_id, marker, key, event)
                    for marker, key, event in zip(burst_markers, burst_keys, started)
                )
                eventually(lambda: sum(event.is_set() for event in started), lambda count: count == 30)
                completed_at_kill: Final = frozenset(index for index, future in enumerate(futures) if future.done())
                in_flight_at_kill: Final = frozenset(range(30)) - completed_at_kill
                victim.send_signal(signal.SIGKILL)
                victim.wait(timeout=10)
                burst_outcomes: Final = tuple(future.result(timeout=30) for future in futures)

                post_markers: Final = tuple(
                    f"lit5610 no registry worker followup {index} {uuid.uuid4().hex}" for index in range(20)
                )
                post_keys: Final = tuple(allowed_key if index % 2 == 0 else denied_key for index in range(20))
                post_responses: Final = tuple(
                    executor.map(
                        _rag_query,
                        repeat(candidate),
                        repeat(model),
                        repeat(store_id),
                        post_markers,
                        post_keys,
                    )
                )

            connection_error_indexes: Final = tuple(
                index for index, outcome in enumerate(burst_outcomes) if isinstance(outcome, httpx.TransportError)
            )
            assert in_flight_at_kill, "all burst requests completed before the worker was killed"
            assert all(
                index in in_flight_at_kill
                and isinstance(burst_outcomes[index], (httpx.NetworkError, httpx.RemoteProtocolError))
                for index in connection_error_indexes
            ), tuple((index, str(burst_outcomes[index])) for index in connection_error_indexes)
            response_statuses: Final = tuple(
                (index, outcome.status_code)
                for index, outcome in enumerate(burst_outcomes)
                if isinstance(outcome, httpx.Response)
            )
            expected_response_statuses: Final = tuple(
                (index, 200 if index % 2 == 0 else 401)
                for index, outcome in enumerate(burst_outcomes)
                if isinstance(outcome, httpx.Response)
            )
            post_statuses: Final = tuple(response.status_code for response in post_responses)
            expected_post_statuses: Final = (200, 401) * 10
            observations: Final = upstream_observations(gateway)
            all_markers: Final = burst_markers + post_markers
            denied_search_counts: Final = tuple(
                len(_searches_in(observations, marker, store_id))
                for index, marker in enumerate(all_markers)
                if (index < 30 and index % 2 == 1) or (index >= 30 and (index - 30) % 2 == 1)
            )
            returned_allowed_markers: Final = tuple(
                marker
                for index, (marker, outcome) in enumerate(zip(burst_markers, burst_outcomes))
                if index % 2 == 0 and isinstance(outcome, httpx.Response) and outcome.status_code == 200
            ) + tuple(marker for index, marker in enumerate(post_markers) if index % 2 == 0)
            allowed_search_counts: Final = tuple(
                len(_searches_in(observations, marker, store_id)) for marker in returned_allowed_markers
            )
            returned_denied: Final = tuple(
                outcome
                for index, outcome in enumerate(burst_outcomes)
                if index % 2 == 1 and isinstance(outcome, httpx.Response)
            )
            response_texts: Final = tuple(
                outcome.text if isinstance(outcome, httpx.Response) else str(outcome) for outcome in burst_outcomes
            ) + tuple(response.text for response in post_responses)

            assert response_statuses == expected_response_statuses, (
                f"response_texts={response_texts!r}; connection_error_indexes={connection_error_indexes!r}; "
                f"denied_search_counts={denied_search_counts!r}; allowed_search_counts={allowed_search_counts!r}"
            )
            assert denied_search_counts == (0,) * len(denied_search_counts), denied_search_counts
            assert allowed_search_counts == (1,) * len(allowed_search_counts), allowed_search_counts
            assert post_statuses == expected_post_statuses, (
                f"response_texts={response_texts!r}; denied_search_counts={denied_search_counts!r}; "
                f"allowed_search_counts={allowed_search_counts!r}"
            )
            denied_error_types: Final = tuple(
                object_value(object_value(response.json()).get("error")).get("type") for response in returned_denied
            ) + tuple(
                object_value(object_value(response.json()).get("error")).get("type")
                for index, response in enumerate(post_responses)
                if index % 2 == 1
            )
            assert denied_error_types == ("team_vector_store_access_denied",) * (len(returned_denied) + 10), (
                response_texts
            )

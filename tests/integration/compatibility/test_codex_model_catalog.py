import asyncio
import json
import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from itertools import chain
from pathlib import Path
from typing import Final, TypeVar

import anthropic
import httpx
import openai
from pydantic import JsonValue

from litellm.constants import PROXY_CONFIG_RELOAD_INTERVAL_SECONDS
from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.wire import Reply, Request, Wire, wire_server

T = TypeVar("T")

CLIENT_VERSION: Final = "0.159.3"
CONSISTENT_READS: Final = 6
CONVERGENCE_SECONDS: Final = 60
RELOAD_MARGIN_SECONDS: Final = 5.0
PROXY_WORKERS: Final = int(os.environ.get("INTEGRATION_PROXY_WORKERS", "1"))
WORKER_SYNC_SECONDS: Final = 0.0 if PROXY_WORKERS == 1 else PROXY_CONFIG_RELOAD_INTERVAL_SECONDS + RELOAD_MARGIN_SECONDS
ROOT: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
BUNDLED_STOCK_PATH: Final = ROOT / "litellm" / "proxy" / "common_utils" / "codex_bundled_models_0.159.3.json"
BASE_INSTRUCTIONS_PATH: Final = ROOT / "litellm" / "proxy" / "common_utils" / "codex_base_instructions.md"
FRESH_CONNECTION: Final = {"Connection": "close"}
ANTHROPIC_HEADERS: Final = {"anthropic-version": "2023-06-01"}
USAGE: Final = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}


def _generic_tier(identity: str) -> dict[str, JsonValue]:
    return {"id": identity, "name": identity.capitalize(), "description": f"Sends service_tier={identity} upstream"}


def _bundled_stock() -> dict[str, dict[str, JsonValue]]:
    catalog: Final = JSON_OBJECT.validate_json(BUNDLED_STOCK_PATH.read_bytes())
    models: Final = catalog["models"]
    assert isinstance(models, list), catalog
    return {string_value(object_value(model)["slug"]): object_value(model) for model in models}


def _catalog_response(
    gateway: Gateway,
    *,
    key: str | None = None,
    params: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    path: str = "/v1/models",
) -> httpx.Response:
    return gateway.request(
        "GET",
        path,
        key=key,
        params={"client_version": CLIENT_VERSION, **(params or {})},
        headers={**FRESH_CONNECTION, **(headers or {})},
    )


def _entries(response: httpx.Response) -> list[dict[str, JsonValue]]:
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert set(body) == {"models"}, response.text
    models: Final = body["models"]
    assert isinstance(models, list), response.text
    return [object_value(entry) for entry in models]


def _catalog(
    gateway: Gateway, *, key: str | None = None, params: Mapping[str, str] | None = None
) -> list[dict[str, JsonValue]]:
    return _entries(_catalog_response(gateway, key=key, params=params))


def _by_slug(entries: Sequence[Mapping[str, JsonValue]]) -> dict[str, dict[str, JsonValue]]:
    return {string_value(entry["slug"]): dict(entry) for entry in entries}


def _tier_ids(entry: Mapping[str, JsonValue]) -> tuple[str, ...]:
    tiers: Final = entry["service_tiers"]
    assert isinstance(tiers, list), entry
    return tuple(string_value(object_value(tier)["id"]) for tier in tiers)


def _tiers_in(entries: Sequence[Mapping[str, JsonValue]]) -> dict[str, tuple[str, ...]]:
    return {slug: _tier_ids(entry) for slug, entry in _by_slug(entries).items()}


def _tiers_by_slug(
    gateway: Gateway, *, key: str | None = None, params: Mapping[str, str] | None = None
) -> dict[str, tuple[str, ...]]:
    return _tiers_in(_catalog(gateway, key=key, params=params))


def _converged_catalog(
    gateway: Gateway,
    satisfied: Callable[[dict[str, dict[str, JsonValue]]], bool],
    *,
    key: str | None = None,
    params: Mapping[str, str] | None = None,
) -> dict[str, dict[str, JsonValue]]:
    return eventually(
        lambda: _by_slug(_catalog(gateway, key=key, params=params)), satisfied, seconds=CONVERGENCE_SECONDS
    )


def _settled(read: Callable[[], T], satisfied: Callable[[T], bool], *, seconds: float = CONVERGENCE_SECONDS) -> T:
    reads: Final = eventually(
        lambda: tuple(read() for _ in range(CONSISTENT_READS)),
        lambda values: all(satisfied(value) for value in values),
        seconds=seconds,
    )
    return reads[-1]


def _settled_on_every_worker(read: Callable[[], T], satisfied: Callable[[T], bool], *, written_at: float) -> T:
    eventually(
        lambda: (time.monotonic() - written_at, read()),
        lambda stamped: stamped[0] >= WORKER_SYNC_SECONDS and satisfied(stamped[1]),
        seconds=WORKER_SYNC_SECONDS + CONVERGENCE_SECONDS,
    )
    return _settled(read, satisfied)


def _plain_ids(gateway: Gateway, *, key: str | None = None, params: Mapping[str, str] | None = None) -> list[str]:
    response: Final = gateway.request("GET", "/v1/models", key=key, params=params, headers=FRESH_CONNECTION)
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert set(body) == {"data", "object"} and body["object"] == "list", response.text
    data: Final = body["data"]
    assert isinstance(data, list), response.text
    return [string_value(object_value(entry)["id"]) for entry in data]


def _model_info_rows(gateway: Gateway, model_name: str) -> list[dict[str, JsonValue]]:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list), entries
    return [object_value(entry) for entry in entries if object_value(entry)["model_name"] == model_name]


def _delete_model_if_present(gateway: Gateway, identity: str) -> None:
    if read_rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (identity,)):
        gateway.post("/model/delete", {"id": identity})
    assert read_rows('SELECT model_id FROM "LiteLLM_ProxyModelTable" WHERE model_id = %s', (identity,)) == []


def _new_model(
    scenario: Scenario,
    name: str,
    *,
    model: str = "openai/gpt-4o-mini",
    model_info: Mapping[str, JsonValue] | None = None,
    api_base: str | None = None,
) -> str:
    created: Final = scenario.gateway.post(
        "/model/new",
        {
            "model_name": name,
            "litellm_params": {
                "model": model,
                "api_key": "integration-provider-key",
                "api_base": api_base or f"{scenario.gateway.upstream_url}/v1",
            },
            "model_info": dict(model_info) if model_info is not None else {},
        },
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(_delete_model_if_present, scenario.gateway, identity)
    return identity


def _model_name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def test_client_version_answers_codex_catalog_shape(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model_info={"max_input_tokens": 2048})
        catalog: Final = _converged_catalog(
            gateway, lambda value: model in value and value[model]["context_window"] == 2048
        )
        slugs: Final = list(catalog)
        entry: Final = catalog[model]
        assert entry["display_name"] == model, entry
        assert entry["priority"] == slugs.index(model), entry
        assert entry["visibility"] == "list" and entry["supported_in_api"] is True, entry
        assert entry["service_tiers"] == [], entry
        assert entry["context_window"] == 2048, entry
        assert entry["base_instructions"] == BASE_INSTRUCTIONS_PATH.read_text(encoding="utf-8"), entry["slug"]


def test_models_route_matches_v1_models(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        routes: Final = eventually(
            lambda: (_entries(_catalog_response(gateway, key=key, path="/models")), _catalog(gateway, key=key)),
            lambda pair: pair[0] == pair[1],
            seconds=CONVERGENCE_SECONDS,
        )
        assert [entry["slug"] for entry in routes[0]] == [model], routes


def test_plain_listing_keeps_openai_shape(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model_info={"service_tiers": ["priority"]})
        key: Final = scenario.key(models=[model])
        ids: Final = eventually(
            lambda: _plain_ids(gateway, key=key), lambda value: value == [model], seconds=CONVERGENCE_SECONDS
        )
        assert ids == [model], ids
        response: Final = gateway.request("GET", "/v1/models", key=key, headers=FRESH_CONNECTION)
        assert "service_tiers" not in response.text and "base_instructions" not in response.text, response.text


def test_client_version_wins_over_anthropic_headers(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        codex: Final = eventually(
            lambda: _catalog_response(gateway, key=key, headers={**ANTHROPIC_HEADERS, "x-api-key": key}),
            lambda response: [entry["slug"] for entry in _entries(response)] == [model],
            seconds=CONVERGENCE_SECONDS,
        )
        assert [entry["slug"] for entry in _entries(codex)] == [model], codex.text
        anthropic_listing: Final = gateway.request(
            "GET", "/v1/models", key=key, headers={**ANTHROPIC_HEADERS, "x-api-key": key, **FRESH_CONNECTION}
        )
        assert anthropic_listing.status_code == 200, anthropic_listing.text
        body: Final = JSON_OBJECT.validate_json(anthropic_listing.content)
        assert set(body) == {"data", "first_id", "has_more", "last_id"}, anthropic_listing.text


def test_cursor_models_route_ignores_client_version(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        response: Final = eventually(
            lambda: _catalog_response(gateway, key=key, path="/cursor/v1/models"),
            lambda value: value.status_code == 200 and model in value.text,
            seconds=CONVERGENCE_SECONDS,
        )
        assert response.status_code == 200, response.text
        body: Final = JSON_OBJECT.validate_json(response.content)
        assert set(body) == {"data", "object"}, response.text
        data: Final = body["data"]
        assert isinstance(data, list) and [object_value(entry)["id"] for entry in data] == [model], response.text


def test_client_version_value_shapes(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model(model_info={"service_tiers": ["priority"]})
        key: Final = scenario.key(models=[model])
        for value in ("", "1", "a" * 5000):
            response: Final = gateway.request(
                "GET", "/v1/models", key=key, params={"client_version": value}, headers=FRESH_CONNECTION
            )
            assert [entry["slug"] for entry in _entries(response)] == [model], (value[:20], response.text)
        repeated: Final = gateway.client.get(
            "/v1/models?client_version=1&client_version=2",
            headers={"Authorization": f"Bearer {key}", **FRESH_CONNECTION},
        )
        assert [entry["slug"] for entry in _entries(repeated)] == [model], repeated.text
        eventually(
            lambda: (_catalog(gateway, key=key), _catalog(gateway, key=key)),
            lambda pair: pair[0] == pair[1] and _tiers_in(pair[0]) == {model: ("priority",)},
            seconds=CONVERGENCE_SECONDS,
        )
        unauthenticated: Final = gateway.client.get(
            "/v1/models", params={"client_version": CLIENT_VERSION}, headers=FRESH_CONNECTION
        )
        assert unauthenticated.status_code == 401, unauthenticated.text


def test_configured_tiers_on_unknown_model(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        ultrafast: Final = {"id": "ultrafast", "name": "Ultra", "description": "fast lane"}
        model: Final = scenario.model(
            model_info={
                "service_tiers": ["priority", ultrafast],
                "display_name": "Probe Unknown",
                "max_input_tokens": 1234,
            }
        )
        entry: Final = _converged_catalog(
            gateway, lambda value: model in value and value[model]["context_window"] == 1234
        )[model]
        assert entry["service_tiers"] == [_generic_tier("priority"), ultrafast], entry
        assert entry["display_name"] == "Probe Unknown", entry
        assert entry["context_window"] == 1234, entry
        assert entry["upgrade"] is None, entry


def test_stock_model_keeps_codex_entry(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = _model_name("codex-stock")
        _new_model(scenario, name, model="openai/gpt-5.5")
        catalog: Final = _converged_catalog(gateway, lambda value: name in value)
        slugs: Final = list(catalog)
        stock: Final = _bundled_stock()["gpt-5.5"]
        assert stock["upgrade"] is not None and stock["service_tiers"] != [], stock
        expected: Final = {
            **stock,
            "slug": name,
            "display_name": name,
            "priority": slugs.index(name),
            "visibility": "list",
            "supported_in_api": True,
            "upgrade": None,
        }
        assert catalog[name] == expected, catalog[name]


def test_stock_model_configured_tiers_override(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        stock: Final = _bundled_stock()["gpt-6-sol"]
        stock_tiers: Final = stock["service_tiers"]
        assert isinstance(stock_tiers, list) and len(stock_tiers) == 1, stock
        stock_tier: Final = object_value(stock_tiers[0])
        assert stock["default_service_tier"] == stock_tier["id"], stock
        unknown_tier: Final = _model_name("codex-unknown-tier")
        known_tier: Final = _model_name("codex-known-tier")
        _new_model(scenario, unknown_tier, model="openai/gpt-6-sol", model_info={"service_tiers": ["ultrafast"]})
        _new_model(scenario, known_tier, model="openai/gpt-6-sol", model_info={"service_tiers": [stock_tier["id"]]})
        catalog: Final = _converged_catalog(gateway, lambda value: unknown_tier in value and known_tier in value)
        assert catalog[unknown_tier]["service_tiers"] == [_generic_tier("ultrafast")], catalog[unknown_tier]
        assert catalog[unknown_tier]["default_service_tier"] is None, catalog[unknown_tier]
        assert catalog[known_tier]["service_tiers"] == [stock_tier], catalog[known_tier]
        assert catalog[known_tier]["default_service_tier"] == stock_tier["id"], catalog[known_tier]
        assert catalog[known_tier]["display_name"] == known_tier, catalog[known_tier]


def test_stock_model_empty_tiers_disable(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = _model_name("codex-no-tiers")
        _new_model(scenario, name, model="openai/gpt-6-sol", model_info={"service_tiers": []})
        entry: Final = _converged_catalog(gateway, lambda value: name in value)[name]
        assert entry["service_tiers"] == [] and entry["default_service_tier"] is None, entry


def test_two_deployments_offer_only_shared_tiers(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        shared: Final = _model_name("codex-shared")
        half_set: Final = _model_name("codex-half-set")
        _new_model(scenario, shared, model_info={"service_tiers": ["a", "b"]})
        _new_model(scenario, shared, model_info={"service_tiers": ["b", "c"]})
        _new_model(scenario, half_set, model_info={"service_tiers": ["a"]})
        _new_model(scenario, half_set)
        tiers: Final = _settled(
            lambda: _tiers_by_slug(gateway),
            lambda value: value.get(shared) == ("b",) and value.get(half_set) == (),
        )
        assert tiers[shared] == ("b",) and tiers[half_set] == (), tiers


def test_invalid_tiers_offer_nothing_and_keep_listing(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        as_string: Final = scenario.model(model_info={"service_tiers": "priority"})
        stray_key: Final = scenario.model(
            model_info={"service_tiers": [{"id": "x", "name": "X", "description": "d", "stray": 1}]}
        )
        empty_id: Final = scenario.model(model_info={"service_tiers": [{"id": "", "name": "", "description": ""}]})
        control: Final = scenario.model(model_info={"service_tiers": ["priority"]})
        written: Final = (as_string, stray_key, empty_id, control)
        catalog: Final = _converged_catalog(gateway, lambda value: all(slug in value for slug in written))
        assert catalog[as_string]["service_tiers"] == [], catalog[as_string]
        assert catalog[stray_key]["service_tiers"] == [], catalog[stray_key]
        assert catalog[empty_id]["service_tiers"] == [], catalog[empty_id]
        assert catalog[control]["service_tiers"] == [_generic_tier("priority")], catalog[control]


def test_blocked_deployment_tiers_are_ignored(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = _model_name("codex-blockable")
        _new_model(scenario, name, model_info={"service_tiers": ["a", "b"]})
        narrowing: Final = _new_model(scenario, name, model_info={"service_tiers": ["b", "c"]})
        _settled(lambda: _tiers_by_slug(gateway).get(name), lambda value: value == ("b",))
        blocked: Final = gateway.request("PATCH", f"/model/{narrowing}/update", {"blocked": True})
        assert blocked.status_code == 200, blocked.text
        _settled(lambda: _tiers_by_slug(gateway).get(name), lambda value: value == ("a", "b"))
        unblocked: Final = gateway.request("PATCH", f"/model/{narrowing}/update", {"blocked": False})
        assert unblocked.status_code == 200, unblocked.text
        _settled(lambda: _tiers_by_slug(gateway).get(name), lambda value: value == ("b",))


def test_restricted_key_lists_only_its_models(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        tiered: Final = scenario.model(model_info={"service_tiers": ["priority"]})
        other: Final = scenario.model(model_info={"service_tiers": ["flex"]})
        key: Final = scenario.key(models=[tiered])
        tiers: Final = eventually(
            lambda: _tiers_by_slug(gateway, key=key),
            lambda value: value == {tiered: ("priority",)},
            seconds=CONVERGENCE_SECONDS,
        )
        assert tiers == {tiered: ("priority",)}, tiers
        admin_tiers: Final = eventually(
            lambda: _tiers_by_slug(gateway), lambda value: value.get(other) == ("flex",), seconds=CONVERGENCE_SECONDS
        )
        assert admin_tiers[other] == ("flex",), admin_tiers


def test_team_scoped_model_tiers(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        owning_team: Final = scenario.team()
        other_team: Final = scenario.team()
        public_name: Final = _model_name("codex-team-model")
        _new_model(scenario, public_name, model_info={"team_id": owning_team, "service_tiers": ["priority"]})
        owning_key: Final = scenario.key(team_id=owning_team)
        other_key: Final = scenario.key(team_id=other_team)
        owning_tiers: Final = _settled(
            lambda: _tiers_by_slug(gateway, key=owning_key), lambda value: value.get(public_name) == ("priority",)
        )
        assert owning_tiers[public_name] == ("priority",), owning_tiers
        admin_tiers: Final = _settled(
            lambda: _tiers_by_slug(gateway, params={"team_id": owning_team}),
            lambda value: value.get(public_name) == ("priority",),
        )
        assert admin_tiers[public_name] == ("priority",), admin_tiers
        assert public_name not in _tiers_by_slug(gateway, key=other_key), other_key[:8]
        assert public_name not in _tiers_by_slug(gateway)


def test_wildcard_deployment_never_listed(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        _new_model(scenario, "openai/*", model="openai/*", model_info={"service_tiers": ["priority"]})
        flags: Final = {"return_wildcard_routes": "true"}
        plain: Final = eventually(
            lambda: _plain_ids(gateway, params=flags), lambda ids: "openai/*" in ids and model in ids
        )
        assert "openai/*" in plain, plain
        slugs: Final = eventually(
            lambda: [string_value(entry["slug"]) for entry in _catalog(gateway, params=flags)],
            lambda values: model in values,
        )
        assert all("*" not in slug for slug in slugs), slugs


def test_a_tiered_model_listed_last_survives_the_byte_cut(gateway: Gateway) -> None:
    """Codex 0.159.3 rejects a `model_catalog_url` body over `MAX_MODEL_CATALOG_BYTES` (1 MiB,
    `codex-rs/model-provider/src/models_endpoint.rs`, read on 2026-10-03), so the proxy keeps the
    entries offering a tier first and then listing order; fallback entries carry Codex's base
    prompt, so some fifty of them overrun the limit."""
    with gateway.scenario() as scenario:
        plain: Final = tuple(scenario.model() for _ in range(52))
        tiered: Final = scenario.model(model_info={"service_tiers": ["ultrafast"]})
        key: Final = scenario.key(models=[*plain, tiered])
        listing: Final = eventually(
            lambda: _plain_ids(gateway, key=key), lambda ids: set(ids) == {*plain, tiered}, seconds=CONVERGENCE_SECONDS
        )
        assert listing.index(tiered) == len(plain), listing
        response: Final = eventually(
            lambda: _catalog_response(gateway, key=key),
            lambda value: value.status_code == 200 and tiered in {entry["slug"] for entry in _entries(value)},
            seconds=CONVERGENCE_SECONDS,
        )
        entries: Final = _entries(response)
        slugs: Final = [string_value(entry["slug"]) for entry in entries]
        assert len(response.content) <= 1024 * 1024, len(response.content)
        assert 0 < len(slugs) < len(listing), (len(slugs), len(listing))
        assert slugs == [slug for slug in listing if slug in slugs], slugs
        assert slugs[-1] == tiered and _tier_ids(entries[-1]) == ("ultrafast",), entries[-1]
        assert slugs[:-1] == list(plain[: len(slugs) - 1]), slugs
        assert [entry["priority"] for entry in entries] == [listing.index(slug) for slug in slugs], slugs


def test_an_oversized_tiered_model_is_left_out_alone(gateway: Gateway) -> None:
    """A tier description longer than Codex's whole byte limit makes an entry that can never fit;
    it is taken first as a tiered entry, so the cut passes over it and still keeps the models after it."""
    with gateway.scenario() as scenario:
        plain: Final = scenario.model()
        oversized_tier: Final = {"id": "huge", "name": "Huge", "description": "x" * (1024 * 1024)}
        oversized: Final = scenario.model(model_info={"service_tiers": [oversized_tier]})
        tiered: Final = scenario.model(model_info={"service_tiers": ["ultrafast"]})
        key: Final = scenario.key(models=[plain, oversized, tiered])
        listing: Final = eventually(
            lambda: _plain_ids(gateway, key=key),
            lambda ids: set(ids) == {plain, oversized, tiered},
            seconds=CONVERGENCE_SECONDS,
        )
        assert listing == [plain, oversized, tiered], listing

        def kept() -> tuple[int, tuple[tuple[JsonValue, JsonValue, tuple[str, ...]], ...]]:
            response: Final = _catalog_response(gateway, key=key)
            return len(response.content), tuple(
                (entry["slug"], entry["priority"], _tier_ids(entry)) for entry in _entries(response)
            )

        expected: Final = ((plain, 0, ()), (tiered, 2, ("ultrafast",)))
        size, entries = _settled(kept, lambda value: value[1] == expected)
        assert entries == expected, entries
        assert size <= 1024 * 1024, size


def test_model_info_echoes_configured_tiers(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        configured: Final = ["priority", {"id": "ultrafast", "name": "Ultra", "description": "fast lane"}]
        model: Final = scenario.model(model_info={"service_tiers": configured})
        matching: Final = eventually(
            lambda: _model_info_rows(gateway, model), lambda rows: len(rows) == 1, seconds=CONVERGENCE_SECONDS
        )
        assert len(matching) == 1, model
        assert object_value(matching[0]["model_info"])["service_tiers"] == configured, matching[0]


def test_tier_update_propagates(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = _model_name("codex-updatable")
        identity: Final = _new_model(scenario, name, model_info={"service_tiers": ["priority"]})
        _settled(lambda: _tiers_by_slug(gateway).get(name), lambda value: value == ("priority",))
        updated: Final = gateway.request(
            "PATCH", f"/model/{identity}/update", {"model_info": {"service_tiers": ["flex", "ultrafast"]}}
        )
        assert updated.status_code == 200, updated.text
        _settled(lambda: _tiers_by_slug(gateway).get(name), lambda value: value == ("flex", "ultrafast"))


def test_model_delete_drops_the_slug(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        name: Final = _model_name("codex-deletable")
        identity: Final = _new_model(scenario, name, model_info={"service_tiers": ["priority"]})
        _settled(lambda: _tiers_by_slug(gateway).get(name), lambda value: value == ("priority",))
        gateway.post("/model/delete", {"id": identity})
        _settled(lambda: _tiers_by_slug(gateway).get(name), lambda value: value is None)


def _sse(events: Sequence[object]) -> tuple[bytes, ...]:
    return tuple(b"data: " + json.dumps(event).encode() + b"\n\n" for event in events) + (b"data: [DONE]\n\n",)


def _chat_reply(stream: bool) -> Reply:
    identity: Final = "chatcmpl-" + uuid.uuid4().hex
    if not stream:
        return Reply(
            body=json.dumps(
                {
                    "id": identity,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "tier probe"}, "finish_reason": "stop"}
                    ],
                    "usage": USAGE,
                }
            ).encode()
        )
    head: Final = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini"}
    return Reply(
        content_type="text/event-stream",
        chunks=_sse(
            (
                {**head, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "tier "}}]},
                {**head, "choices": [{"index": 0, "delta": {"content": "probe"}}]},
                {**head, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
                {**head, "choices": [], "usage": USAGE},
            )
        ),
    )


def _responses_reply() -> Reply:
    identity: Final = uuid.uuid4().hex
    return Reply(
        body=json.dumps(
            {
                "id": "resp_" + identity,
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "id": "msg_" + identity,
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "tier probe", "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
                "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


def _upstream(request: Request) -> Reply:
    if request.method == "GET" and request.target.endswith("/models"):
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    if request.target.endswith("/responses"):
        return _responses_reply()
    return _chat_reply(json.loads(request.body).get("stream") is True)


def _v1(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/") + "/v1"


def _llm_requests(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if request.method == "POST")


def _request_bodies(requests: Sequence[Request]) -> list[dict[str, JsonValue]]:
    return [JSON_OBJECT.validate_json(request.body) for request in requests]


def _texts(value: JsonValue) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, list):
        return tuple(chain.from_iterable(_texts(item) for item in value))
    if isinstance(value, dict):
        return tuple(chain.from_iterable(_texts(value.get(key)) for key in ("content", "text", "input", "messages")))
    return ()


def _prompt_of(body: Mapping[str, JsonValue]) -> str:
    return " ".join(_texts(body.get("input", body.get("messages"))))


def _sent_prompt(body: Mapping[str, JsonValue], sent: Sequence[str]) -> str:
    text: Final = _prompt_of(body)
    matches: Final = [prompt for prompt in sent if prompt in text]
    assert len(matches) == 1, (matches, text)
    return matches[0]


def _spend_ids_for_key(key: str) -> list[str]:
    rows: Final = read_rows(
        'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (sha256(key.encode()).hexdigest(),)
    )
    return [string_value(row["request_id"]) for row in rows]


def test_advertised_tier_reaches_the_upstream(gateway: Gateway) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        written_at: Final = time.monotonic()
        model: Final = scenario.model(api_base=f"{wire.url}/v1", model_info={"service_tiers": ["priority"]})
        key: Final = scenario.key(models=[model])
        marker: Final = uuid.uuid4().hex
        advertised: Final = _settled_on_every_worker(
            lambda: _tiers_by_slug(gateway, key=key),
            lambda tiers: tiers == {model: ("priority",)},
            written_at=written_at,
        )
        assert advertised == {model: ("priority",)}, advertised
        sync_client: Final = openai.OpenAI(
            api_key=key, base_url=_v1(gateway), max_retries=0, http_client=httpx.Client(timeout=15, trust_env=False)
        )
        responses_id: Final = sync_client.responses.create(
            model=model, input=f"{marker} responses probe", service_tier="priority"
        ).id
        chat_id: Final = sync_client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": f"{marker} chat probe"}], service_tier="priority"
        ).id

        async def streamed() -> str:
            client: Final = openai.AsyncOpenAI(
                api_key=key,
                base_url=_v1(gateway),
                max_retries=0,
                http_client=httpx.AsyncClient(timeout=15, trust_env=False),
            )
            stream: Final = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": f"{marker} stream probe"}],
                service_tier="priority",
                stream=True,
            )
            chunks: Final = [chunk async for chunk in stream]
            assert chunks, "no chunks streamed"
            return chunks[0].id

        stream_id: Final = asyncio.run(streamed())
        messages_id: Final = (
            anthropic.Anthropic(
                api_key=key,
                base_url=str(gateway.client.base_url).rstrip("/"),
                max_retries=0,
                http_client=httpx.Client(timeout=15, trust_env=False),
            )
            .messages.create(
                model=model, max_tokens=64, messages=[{"role": "user", "content": f"{marker} messages probe"}]
            )
            .id
        )
        received: Final = _llm_requests(wire)
        assert len(received) == 4, [request.target for request in received]
        targets: Final = sorted(request.target for request in received)
        assert targets == ["/v1/chat/completions"] * 2 + ["/v1/responses"] * 2, targets
        openai_shaped: Final = [
            body for body in _request_bodies(received) if f"{marker} messages probe" not in _prompt_of(body)
        ]
        assert len(openai_shaped) == 3, _request_bodies(received)
        assert all(body["service_tier"] == "priority" for body in openai_shaped), openai_shaped
        ids: Final = sorted((responses_id, chat_id, stream_id, messages_id))
        assert len(set(ids)) == 4, ids
        landed: Final = eventually(lambda: _spend_ids_for_key(key), lambda values: len(values) == 4, seconds=70)
        assert sorted(landed) == ids, (landed, ids)


BURST_CATALOG: Final = 12
BURST_CHAT: Final = 8
BURST_STREAM: Final = 8
BURST_RESPONSES: Final = 4
BURST_MESSAGES: Final = 4
BURST_SIZES: Final = (
    ("chat", BURST_CHAT),
    ("stream", BURST_STREAM),
    ("responses", BURST_RESPONSES),
    ("messages", BURST_MESSAGES),
)


def _llm_tasks() -> tuple[tuple[str, int], ...]:
    return tuple(
        (kind, index) for kind, size in BURST_SIZES for index in range(size)
    )  # comprehension-ok: a flat task list of two small axes


def _stream_id(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    events: Final = [
        object_value(json.loads(line.removeprefix("data:")))
        for line in response.iter_lines()
        if line.startswith("data:") and line.removeprefix("data:").strip() != "[DONE]"
    ]
    assert events, response.text
    return string_value(events[0]["id"])


def _json_id(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return string_value(JSON_OBJECT.validate_json(response.content)["id"])


def test_catalog_burst_with_model_churn(gateway: Gateway) -> None:
    with wire_server(_upstream) as wire, gateway.scenario() as scenario:
        written_at: Final = time.monotonic()
        model: Final = scenario.model(api_base=f"{wire.url}/v1", model_info={"service_tiers": ["priority"]})
        key: Final = scenario.key(models=[model])
        marker: Final = uuid.uuid4().hex
        _settled_on_every_worker(
            lambda: _tiers_by_slug(gateway, key=key),
            lambda tiers: tiers == {model: ("priority",)},
            written_at=written_at,
        )

        def prompt(kind: str, index: int) -> str:
            return f"{marker} {kind} {index}"

        def catalog(index: int) -> tuple[str, str]:
            tiers: Final = _tiers_by_slug(gateway, key=key)
            assert tiers == {model: ("priority",)}, (index, tiers)
            return ("catalog", "")

        def chat(index: int) -> tuple[str, str]:
            body: Final = {"model": model, "messages": [{"role": "user", "content": prompt("chat", index)}]}
            return ("chat", _json_id(gateway.request("POST", "/v1/chat/completions", body, key=key)))

        def stream(index: int) -> tuple[str, str]:
            body: Final = {
                "model": model,
                "messages": [{"role": "user", "content": prompt("stream", index)}],
                "stream": True,
            }
            return ("stream", _stream_id(gateway.request("POST", "/v1/chat/completions", body, key=key)))

        def responses(index: int) -> tuple[str, str]:
            body: Final = {"model": model, "input": prompt("responses", index), "service_tier": "priority"}
            return ("responses", _json_id(gateway.request("POST", "/v1/responses", body, key=key)))

        def messages(index: int) -> tuple[str, str]:
            body: Final = {
                "model": model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": prompt("messages", index)}],
            }
            return ("messages", _json_id(gateway.request("POST", "/v1/messages", body, key=key)))

        def churn(index: int) -> tuple[str, str]:
            churned: Final = _new_model(
                scenario, _model_name(f"codex-churn-{index}"), model_info={"service_tiers": ["flex"]}
            )
            gateway.post("/model/delete", {"id": churned})
            return ("churn", churned)

        tasks: Final[tuple[tuple[Callable[[int], tuple[str, str]], int], ...]] = (
            *((catalog, index) for index in range(BURST_CATALOG)),
            *((chat, index) for index in range(BURST_CHAT)),
            *((stream, index) for index in range(BURST_STREAM)),
            *((responses, index) for index in range(BURST_RESPONSES)),
            *((messages, index) for index in range(BURST_MESSAGES)),
            (churn, 0),
            (churn, 1),
        )
        with ThreadPoolExecutor(max_workers=len(tasks)) as pool:
            outcomes: Final = tuple(pool.map(lambda task: task[0](task[1]), tasks))
        llm_kinds: Final = frozenset({"chat", "stream", "responses", "messages"})
        llm_ids: Final = sorted(identity for kind, identity in outcomes if kind in llm_kinds)
        assert len(llm_ids) == BURST_CHAT + BURST_STREAM + BURST_RESPONSES + BURST_MESSAGES, outcomes
        assert len(set(llm_ids)) == len(llm_ids), llm_ids
        received: Final = _request_bodies(_llm_requests(wire))
        assert len(received) == len(llm_ids), len(received)
        prompts_sent: Final = sorted(prompt(kind, index) for kind, index in _llm_tasks())
        prompts_seen: Final = sorted(_sent_prompt(body, prompts_sent) for body in received)
        assert prompts_seen == prompts_sent, (prompts_seen, prompts_sent)
        landed: Final = eventually(
            lambda: _spend_ids_for_key(key), lambda values: len(values) == len(llm_ids), seconds=70
        )
        assert sorted(landed) == llm_ids, (landed, llm_ids)
        final_tiers: Final = _settled(
            lambda: _tiers_by_slug(gateway, key=key), lambda value: value == {model: ("priority",)}
        )
        assert final_tiers == {model: ("priority",)}, final_tiers

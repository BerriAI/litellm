"""Partial wildcards like ``cloudflare/@cf/*`` list each callable id once on /v1/models and /model/info,
and a partial wildcard that matches no known id, like ``databricks/system.ai.*``, lists no guessed ids.
Every id a wildcard lists reaches the upstream as the model its /model/info row names, priced from that model."""

import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, list_value, object_value, string_value
from integration._support.openai_wire import answering_model_discovery, chat_reply, responses_reply
from integration._support.timing import assert_within_factor_of_control, timed
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

PATTERNS: Final = ("cloudflare/@cf/*", "databricks/system.ai.*", "databricks/*")
UNREACHABLE_API_BASE: Final = "https://example.invalid"
COST_MAP: Final = JSON_OBJECT.validate_json(
    (Path(__file__).resolve().parents[3] / "litellm" / "model_prices_and_context_window_backup.json").read_bytes()
)
ROUTES: Final = MappingProxyType(
    {
        "alias_over_partial_model": ("{namespace}/*", "openai/gpt-4*"),
        "alias_over_one_fixed_model": ("{namespace}/*", "openai/gpt-4o-mini"),
        "catch_all_over_provider": ("*", "openai/*"),
    }
)
SAMPLED_IDS_PER_ROUTE: Final = 6
MANY_STARS: Final = "*" * 7 + "!"
CONTROL_LISTINGS: Final = 3
LISTING_SLOWDOWN_FACTOR: Final = 25
LISTING_TIMEOUT_SECONDS: Final = 120


def _register(scenario: Scenario, model_name: str, model: str, api_base: str) -> str:
    created: Final = object_value(
        scenario.gateway.post(
            "/model/new",
            {
                "model_name": model_name,
                "litellm_params": {"model": model, "api_key": "integration-wildcard-key", "api_base": api_base},
            },
        )
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return identity


@pytest.fixture
def wildcard_deployments(gateway: Gateway) -> Iterator[None]:
    with gateway.scenario() as scenario:
        for pattern in PATTERNS:
            _register(scenario, pattern, pattern, UNREACHABLE_API_BASE)
        yield


def _rows(gateway: Gateway, path: str) -> tuple[dict[str, JsonValue], ...]:
    data: Final = gateway.get(path)["data"]
    assert isinstance(data, list)
    return tuple(object_value(entry) for entry in data)


def _listed_model_ids(gateway: Gateway) -> tuple[str, ...]:
    return tuple(string_value(entry["id"]) for entry in _rows(gateway, "/v1/models"))


def _model_info_names(gateway: Gateway) -> tuple[str, ...]:
    return tuple(string_value(entry["model_name"]) for entry in _rows(gateway, "/model/info"))


def _model_group_names(gateway: Gateway) -> tuple[str, ...]:
    return tuple(string_value(entry["model_group"]) for entry in _rows(gateway, "/model_group/info"))


def _openai_reply(request: Request) -> Reply:
    model: Final = string_value(JSON_OBJECT.validate_json(request.body)["model"])
    identity: Final = f"wildcard-{uuid.uuid4().hex}"
    if request.target.endswith("/responses"):
        return responses_reply(identity, model, "listed id answered", stream=False)
    return chat_reply(identity, model, "listed id answered", stream=False)


OPENAI_UPSTREAM: Final = answering_model_discovery(_openai_reply)


def _call(gateway: Gateway, model: str) -> httpx.Response:
    response: Final = gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "which model am I"}], "cache": {"no-cache": True}},
    )
    assert response.status_code == 200, f"{model}: {response.status_code} {response.text}"
    return response


def _upstream_model(wire: Wire) -> str:
    (request,) = tuple(request for request in wire.drain() if request.method == "POST")
    return string_value(JSON_OBJECT.validate_json(request.body)["model"])


def _upstream_model_of_call(gateway: Gateway, wire: Wire, model: str) -> str:
    _call(gateway, model)
    return _upstream_model(wire)


def _spread(ids: Sequence[str], count: int) -> tuple[str, ...]:
    ordered: Final = sorted(ids)
    return tuple(ordered[:: max(1, len(ordered) // count)][:count])


def _openai_cost_entry(model: str) -> dict[str, JsonValue] | None:
    entry: Final = COST_MAP.get(model)
    if not isinstance(entry, dict) or entry.get("litellm_provider") != "openai":
        return None
    return entry


def _assert_row_names_priced_model(
    rows: Sequence[dict[str, JsonValue]], name: str, deployment_id: str, upstream_model: str
) -> None:
    (row,) = tuple(
        row for row in rows if row["model_name"] == name and object_value(row["model_info"])["id"] == deployment_id
    )
    cost_entry: Final = _openai_cost_entry(upstream_model)
    assert cost_entry is not None, f"{name} reached {upstream_model}, which no openai price names"
    assert object_value(row["litellm_params"])["model"] == f"openai/{upstream_model}", name
    assert object_value(row["model_info"])["input_cost_per_token"] == cost_entry.get("input_cost_per_token"), name


def test_partial_wildcard_lists_provider_ids_once_on_both_endpoints(
    gateway: Gateway, wildcard_deployments: None
) -> None:
    listed: Final = tuple(model_id for model_id in _listed_model_ids(gateway) if model_id.startswith("cloudflare/"))
    model_info: Final = tuple(name for name in _model_info_names(gateway) if name.startswith("cloudflare/"))

    assert listed
    assert all(model_id.startswith("cloudflare/@cf/") and model_id.count("@cf/") == 1 for model_id in listed)
    assert len(listed) == len(set(listed))
    assert sorted(model_info) == sorted(listed)


def test_partial_wildcard_matching_no_known_id_lists_no_guessed_ids(
    gateway: Gateway, wildcard_deployments: None
) -> None:
    assert [model_id for model_id in _listed_model_ids(gateway) if model_id.startswith("databricks/system.ai.")] == []
    assert [name for name in _model_info_names(gateway) if name.startswith("databricks/system.ai.")] == [
        "databricks/system.ai.*"
    ]


def test_provider_wildcard_still_expands_to_cost_map_names(gateway: Gateway, wildcard_deployments: None) -> None:
    assert any(model_id.startswith("databricks/databricks-") for model_id in _listed_model_ids(gateway))


@pytest.mark.parametrize("route", tuple(ROUTES))
def test_listed_wildcard_ids_reach_the_upstream_as_the_priced_model_their_row_names(
    gateway: Gateway, route: str
) -> None:
    model_name_template, deployment_model = ROUTES[route]
    model_name: Final = model_name_template.format(namespace=f"listing-{uuid.uuid4().hex[:12]}")
    already_listed: Final = frozenset(_listed_model_ids(gateway))

    with wire_server(OPENAI_UPSTREAM) as wire, gateway.scenario() as scenario:
        deployment_id: Final = _register(scenario, model_name, deployment_model, f"{wire.url}/v1")
        listed: Final = tuple(model_id for model_id in _listed_model_ids(gateway) if model_id not in already_listed)
        rows: Final = _rows(gateway, "/model/info")
        groups: Final = _model_group_names(gateway)
        sampled: Final = _spread(listed, SAMPLED_IDS_PER_ROUTE)
        wire.drain()

        assert sampled, f"{model_name} over {deployment_model} listed no ids"
        assert all(groups.count(model_id) == 1 for model_id in sampled), sampled
        upstream_models: Final = tuple(_upstream_model_of_call(gateway, wire, model_id) for model_id in sampled)

    for model_id, upstream_model in zip(sampled, upstream_models, strict=True):
        _assert_row_names_priced_model(rows, model_id, deployment_id, upstream_model)
    assert len(set(upstream_models)) == len(sampled), dict(zip(sampled, upstream_models, strict=True))


@pytest.mark.parametrize(
    ("narrow_route", "narrow_model"),
    (("gpt-4o*", "openai/gpt-4o*"), ("gpt-4o-mini", "openai/gpt-4o-mini")),
    ids=("narrower_wildcard", "exact_name"),
)
def test_each_wildcard_row_names_the_deployment_its_id_routes_to(
    gateway: Gateway, narrow_route: str, narrow_model: str
) -> None:
    namespace: Final = f"shadow-{uuid.uuid4().hex[:12]}"

    with wire_server(OPENAI_UPSTREAM) as wire, gateway.scenario() as scenario:
        narrow_id: Final = _register(scenario, f"{namespace}/{narrow_route}", narrow_model, f"{wire.url}/v1")
        broad_id: Final = _register(scenario, f"{namespace}/*", "openai/*", f"{wire.url}/v1")
        rows: Final = tuple(
            (string_value(row["model_name"]), string_value(object_value(row["model_info"])["id"]))
            for row in _rows(gateway, "/model/info")
            if string_value(row["model_name"]).startswith(f"{namespace}/")
        )
        contested: Final = tuple(row for row in rows if row[0].startswith(f"{namespace}/gpt-4o"))
        broad_only: Final = tuple(row for row in rows if not row[0].startswith(f"{namespace}/gpt-4o"))
        sampled_broad: Final = frozenset(_spread([name for name, _ in broad_only], 3))
        checked: Final = contested + tuple(row for row in broad_only if row[0] in sampled_broad)
        served: Final = tuple(
            (name, row_deployment, _call(gateway, name).headers.get("x-litellm-model-id"))
            for name, row_deployment in checked
        )

    assert any(row_deployment == narrow_id for _, row_deployment in contested)
    assert any(row_deployment == broad_id for _, row_deployment in broad_only)
    assert [entry for entry in served if entry[1] != entry[2]] == []


def test_key_scoped_to_a_wildcard_lists_an_id_a_narrower_route_sends_to_the_same_model(gateway: Gateway) -> None:
    namespace: Final = f"keyscope-{uuid.uuid4().hex[:12]}"
    narrowly_served: Final = f"{namespace}/gpt-4.1"

    with wire_server(OPENAI_UPSTREAM) as wire, gateway.scenario() as scenario:
        _register(scenario, f"{namespace}/*", "openai/*", f"{wire.url}/v1")
        _register(scenario, f"{namespace}/gpt-4.1*", "openai/gpt-4.1*", f"{wire.url}/v1")
        key: Final = scenario.key(models=[f"{namespace}/*"])
        listing: Final = gateway.request("GET", "/v1/models", key=key)
        wire.drain()
        call: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": narrowly_served, "messages": [{"role": "user", "content": "hi"}], "cache": {"no-cache": True}},
            key=key,
        )
        upstream_model: Final = _upstream_model(wire)

    assert call.status_code == 200, call.text
    assert upstream_model == "gpt-4.1"
    assert listing.status_code == 200, listing.text
    listed: Final = tuple(
        string_value(object_value(entry)["id"])
        for entry in list_value(JSON_OBJECT.validate_json(listing.content)["data"])
    )
    assert narrowly_served in listed, sorted(model_id for model_id in listed if model_id.startswith(namespace))


def test_overlapping_deployments_of_one_wildcard_list_each_id_once(gateway: Gateway) -> None:
    namespace: Final = f"overlap-{uuid.uuid4().hex[:12]}"

    with gateway.scenario() as scenario:
        for _ in range(2):
            _register(scenario, f"{namespace}/gpt-4o*", "openai/gpt-4o*", UNREACHABLE_API_BASE)
        _register(scenario, f"{namespace}/gpt-4o-mini", "openai/gpt-4o-mini", UNREACHABLE_API_BASE)
        listed: Final = tuple(
            model_id for model_id in _listed_model_ids(gateway) if model_id.startswith(f"{namespace}/")
        )
        groups: Final = tuple(name for name in _model_group_names(gateway) if name.startswith(f"{namespace}/"))

    assert f"{namespace}/gpt-4o-mini" in listed
    assert len(listed) == len(set(listed)), sorted(listed)
    assert len(groups) == len(set(groups)), sorted(groups)
    assert set(groups) == set(listed)


@pytest.mark.parametrize(("pattern", "probes_match"), (("gpt-4o*", True), ("no-such-model-*", False)))
def test_wildcard_health_check_probes_a_model_its_pattern_names(
    gateway: Gateway, pattern: str, probes_match: bool
) -> None:
    model_name: Final = f"health-{uuid.uuid4().hex[:12]}/{pattern}"

    with wire_server(OPENAI_UPSTREAM) as wire, gateway.scenario() as scenario:
        _register(scenario, model_name, f"openai/{pattern}", f"{wire.url}/v1")
        wire.drain()
        health: Final = gateway.get("/health", {"model": model_name})
        probe: Final = _upstream_model(wire)

    assert (health["healthy_count"], health["unhealthy_count"]) == (1, 0), health
    assert probe.startswith(pattern.removesuffix("*")) is probes_match, probe


def _namespace_listing_seconds(gateway: Gateway, namespace: str) -> float:
    listing: Final = timed(
        lambda: gateway.client.get(
            "/v1/models", headers={"Authorization": f"Bearer {gateway.key}"}, timeout=LISTING_TIMEOUT_SECONDS
        )
    )
    assert listing.value.status_code == 200, listing.value.text
    assert any(
        string_value(object_value(entry)["id"]).startswith(f"{namespace}/")
        for entry in list_value(JSON_OBJECT.validate_json(listing.value.content)["data"])
    )
    return listing.seconds


def test_wildcard_listing_answers_promptly_beside_a_route_with_many_stars(gateway: Gateway) -> None:
    namespace: Final = f"stars-{uuid.uuid4().hex[:12]}"

    with gateway.scenario() as scenario:
        _register(scenario, f"{namespace}/*", "openai/*", UNREACHABLE_API_BASE)
        control: Final = max(_namespace_listing_seconds(gateway, namespace) for _ in range(CONTROL_LISTINGS))
        _register(scenario, f"{namespace}/{MANY_STARS}", f"openai/{MANY_STARS}", UNREACHABLE_API_BASE)
        observed: Final = _namespace_listing_seconds(gateway, namespace)

    assert_within_factor_of_control(observed, control=control, factor=LISTING_SLOWDOWN_FACTOR)

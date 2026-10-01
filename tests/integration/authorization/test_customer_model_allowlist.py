import json
import os
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import JsonValue, TypeAdapter
from redis import Redis

from litellm.proxy.common_utils.user_api_key_cache import end_user_restricted_registry_cache_key
from tests.integration._support.client import Gateway, Scenario, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_REGISTRY_IDS: Final = TypeAdapter(list[str])
_CHAT_RESPONSE: Final = {
    "object": "chat.completion",
    "created": 1700000000,
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "scripted response"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 2, "total_tokens": 4},
}


def _customer(scenario: Scenario, **fields: JsonValue) -> str:
    identity: Final = f"integration-customer-{uuid.uuid4().hex}"
    scenario.gateway.post("/customer/new", {"user_id": identity, **fields})
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [identity]})
    return identity


def _chat_reply(request: Request) -> Reply:
    body: Final = _JSON_OBJECT.validate_json(request.body)
    messages: Final = body.get("messages")
    assert isinstance(messages, list) and messages, request.body.decode()
    message: Final = object_value(messages[-1])
    marker: Final = string_value(message["content"])
    response: Final = {
        "id": f"chatcmpl-{marker}",
        "model": string_value(body["model"]),
        **_CHAT_RESPONSE,
    }
    return Reply(body=json.dumps(response).encode())


def _chat(
    gateway: Gateway,
    key: str,
    model: str,
    *,
    customer: str | None,
    headers: Mapping[str, str] | None = None,
    text: str,
) -> httpx.Response:
    body: Final = {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        **({"user": customer} if customer is not None else {}),
    }
    return gateway.request("POST", "/v1/chat/completions", body, key=key, headers=headers)


def _denial_type(response: httpx.Response) -> JsonValue:
    assert response.status_code == 403, response.text
    return object_value(object_value(response.json())["error"])["type"]


def _assert_one_request(wire: Wire, marker: str) -> None:
    requests: Final = wire.drain()
    matching: Final = tuple(request for request in requests if marker.encode() in request.body)
    assert len(matching) == 1, matching


def _assert_no_request(wire: Wire, marker: str) -> None:
    requests: Final = wire.drain()
    matching: Final = tuple(request for request in requests if marker.encode() in request.body)
    assert matching == (), matching


def _customer_request(
    gateway: Gateway,
    key: str,
    model: str,
    customer: str | None,
    wire: Wire,
) -> httpx.Response:
    marker: Final = uuid.uuid4().hex
    response: Final = _chat(gateway, key, model, customer=customer, text=marker)
    if response.status_code == 200:
        assert response.json()["id"] == f"chatcmpl-{marker}", response.text
        _assert_one_request(wire, marker)
    else:
        _assert_no_request(wire, marker)
    return response


def _observe_registry_after_normal_load(
    gateway: Gateway,
    key: str,
    model: str,
    restricted_customer: str,
    wire: Wire,
    redis_cache: Redis,
) -> tuple[str, str, tuple[str, ...]]:
    probe_customer: Final = f"integration-registry-probe-{uuid.uuid4().hex}"
    baseline: Final = _customer_request(gateway, key, model, probe_customer, wire)
    assert baseline.status_code == 200, baseline.text

    registry_suffix: Final = end_user_restricted_registry_cache_key()
    registry_keys: Final = tuple(
        redis_key for redis_key in redis_cache.scan_iter() if redis_key.endswith(registry_suffix)
    )
    assert len(registry_keys) == 1, registry_keys
    redis_key: Final = registry_keys[0]
    serialized: Final = redis_cache.get(redis_key)
    assert serialized is not None, redis_key
    registry_ids: Final = tuple(_REGISTRY_IDS.validate_json(serialized))
    assert restricted_customer in registry_ids, registry_ids
    assert probe_customer not in registry_ids, registry_ids
    return redis_key, serialized, registry_ids


def test_customer_models_allowlist_rejects_model_outside_it_for_the_same_key(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        disallowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, disallowed])
        customer: Final = _customer(scenario, models=[allowed])

        info: Final = gateway.get("/customer/info", {"end_user_id": customer})
        assert info["models"] == [allowed], info

        permitted: Final = _customer_request(gateway, key, allowed, customer, wire)
        assert permitted.status_code == 200, permitted.text
        denied: Final = _customer_request(gateway, key, disallowed, customer, wire)
        assert _denial_type(denied) == "customer_model_access_denied", denied.text
        assert "not in the allowed models for this customer" in denied.text, denied.text
        marker: Final = uuid.uuid4().hex
        via_header: Final = _chat(
            gateway,
            key,
            disallowed,
            customer=None,
            headers={"x-litellm-customer-id": customer},
            text=marker,
        )
        assert _denial_type(via_header) == "customer_model_access_denied", via_header.text
        _assert_no_request(wire, marker)
        without_customer: Final = _customer_request(gateway, key, disallowed, None, wire)
        assert without_customer.status_code == 200, without_customer.text


def test_customer_create_without_models_reads_back_empty_list(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        customer: Final = _customer(scenario)
        info: Final = gateway.get("/customer/info", {"end_user_id": customer})
        assert info["models"] == [], info


def test_customer_update_moves_and_clears_the_models_allowlist(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        first: Final = scenario.model(api_base=f"{wire.url}/v1")
        second: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[first, second])
        customer: Final = _customer(scenario)
        initial: Final = _customer_request(gateway, key, second, customer, wire)
        assert initial.status_code == 200, initial.text

        gateway.post("/customer/update", {"user_id": customer, "models": [second]})
        assert gateway.get("/customer/info", {"end_user_id": customer})["models"] == [second]
        moved_denial: Final = _customer_request(gateway, key, first, customer, wire)
        assert _denial_type(moved_denial) == "customer_model_access_denied", moved_denial.text
        moved_allowed: Final = _customer_request(gateway, key, second, customer, wire)
        assert moved_allowed.status_code == 200, moved_allowed.text

        gateway.post("/customer/update", {"user_id": customer, "models": []})
        assert gateway.get("/customer/info", {"end_user_id": customer})["models"] == []
        cleared: Final = _customer_request(gateway, key, first, customer, wire)
        assert cleared.status_code == 200, cleared.text


def test_customer_update_omitting_models_preserves_the_list(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        customer: Final = _customer(scenario, models=[model])
        updated: Final = gateway.request("POST", "/customer/update", {"user_id": customer, "alias": "updated"})
        assert updated.status_code == 200, updated.text
        assert gateway.get("/customer/info", {"end_user_id": customer})["models"] == [model]


def test_customer_update_with_null_models_preserves_the_list(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        customer: Final = _customer(scenario, models=[model])
        updated: Final = gateway.request("POST", "/customer/update", {"user_id": customer, "models": None})
        assert updated.status_code == 200, updated.text
        assert gateway.get("/customer/info", {"end_user_id": customer})["models"] == [model]


def test_customer_list_returns_models_for_each_customer(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        customer: Final = _customer(scenario, models=[model])
        response: Final = gateway.request("GET", "/customer/list")
        assert response.status_code == 200, response.text
        customers: Final = response.json()
        assert isinstance(customers, list), response.text
        matching: Final = tuple(entry for entry in customers if object_value(entry)["user_id"] == customer)
        assert len(matching) == 1, response.text
        assert object_value(matching[0])["models"] == [model], response.text


def test_customer_model_validation_does_not_break_other_customers(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model])
        unrelated: Final = _customer(scenario, models=[model])
        invalid_string_id: Final = f"integration-invalid-{uuid.uuid4().hex}"
        invalid_members_id: Final = f"integration-invalid-{uuid.uuid4().hex}"
        invalid_string: Final = gateway.request("POST", "/customer/new", {"user_id": invalid_string_id, "models": "m1"})
        invalid_members: Final = gateway.request(
            "POST",
            "/customer/new",
            {"user_id": invalid_members_id, "models": [1, 2]},
        )
        assert invalid_string.status_code == 422, invalid_string.text
        assert invalid_members.status_code == 422, invalid_members.text
        empty_string: Final = gateway.request(
            "POST",
            "/customer/new",
            {"user_id": f"integration-empty-{uuid.uuid4().hex}", "models": [""]},
        )
        assert empty_string.status_code == 200, empty_string.text
        empty_string_info: Final = gateway.get(
            "/customer/info", {"end_user_id": string_value(empty_string.json()["user_id"])}
        )
        assert empty_string_info["models"] == [""], empty_string_info
        served: Final = _customer_request(gateway, key, model, unrelated, wire)
        assert served.status_code == 200, served.text


def test_customer_models_round_trip_a_five_kilobyte_model_name(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = "m" * 5_000
        customer: Final = _customer(scenario, models=[model])
        info: Final = gateway.get("/customer/info", {"end_user_id": customer})
        assert info["models"] == [model], info


def test_customer_without_models_can_call_every_key_model(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        first: Final = scenario.model(api_base=f"{wire.url}/v1")
        second: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[first, second])
        without_models: Final = _customer(scenario)
        empty_models: Final = _customer(scenario, models=[])

        for customer in (without_models, empty_models):
            for model in (first, second):
                for headers in (None, {"x-litellm-customer-id": customer}):
                    marker: Final = uuid.uuid4().hex
                    response: Final = _chat(
                        gateway,
                        key,
                        model,
                        customer=customer if headers is None else None,
                        headers=headers,
                        text=marker,
                    )
                    assert response.status_code == 200, response.text
                    assert response.json()["id"] == f"chatcmpl-{marker}", response.text
                    _assert_one_request(wire, marker)


def test_customer_without_identifier_keeps_key_model_access(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        first: Final = scenario.model(api_base=f"{wire.url}/v1")
        second: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[first, second])
        response: Final = _customer_request(gateway, key, second, None, wire)
        assert response.status_code == 200, response.text


def test_unknown_customer_identifier_is_unrestricted(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        disallowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, disallowed])
        response: Final = _chat(
            gateway,
            key,
            disallowed,
            customer=None,
            headers={"x-litellm-customer-id": f"missing-{uuid.uuid4().hex}"},
            text=uuid.uuid4().hex,
        )
        assert response.status_code == 200, response.text
        assert response.json()["id"].startswith("chatcmpl-"), response.text
        requests: Final = wire.drain()
        assert len(requests) == 1, requests


def test_end_user_id_header_identifies_the_customer_for_model_access(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        disallowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, disallowed])
        customer: Final = _customer(scenario, models=[allowed])
        marker: Final = uuid.uuid4().hex
        response: Final = _chat(
            gateway,
            key,
            disallowed,
            customer=None,
            headers={"x-litellm-end-user-id": customer},
            text=marker,
        )
        assert _denial_type(response) == "customer_model_access_denied", response.text
        _assert_no_request(wire, marker)


def test_customer_allowlist_does_not_widen_key_access(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        key_model: Final = scenario.model(api_base=f"{wire.url}/v1")
        customer_only: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[key_model])
        customer: Final = _customer(scenario, models=[key_model, customer_only])

        allowed: Final = _customer_request(gateway, key, key_model, customer, wire)
        assert allowed.status_code == 200, allowed.text
        denied: Final = _customer_request(gateway, key, customer_only, customer, wire)
        assert _denial_type(denied) == "key_model_access_denied", denied.text


def test_team_restriction_remains_additive_to_customer_allowlist(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        denied: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[allowed])
        key: Final = scenario.key(team_id=team)
        customer: Final = _customer(scenario, models=[allowed, denied])

        served: Final = _customer_request(gateway, key, allowed, customer, wire)
        assert served.status_code == 200, served.text
        blocked: Final = _customer_request(gateway, key, denied, customer, wire)
        assert _denial_type(blocked) == "team_model_access_denied", blocked.text


def test_customer_restriction_remains_additive_to_key_and_team(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        denied: Final = scenario.model(api_base=f"{wire.url}/v1")
        team: Final = scenario.team(models=[allowed, denied])
        key: Final = scenario.key(team_id=team, models=[allowed, denied])
        customer: Final = _customer(scenario, models=[allowed])

        blocked: Final = _customer_request(gateway, key, denied, customer, wire)
        assert _denial_type(blocked) == "customer_model_access_denied", blocked.text


def test_key_alias_outside_customer_allowlist_is_denied(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        hidden: Final = scenario.model(api_base=f"{wire.url}/v1")
        alias: Final = f"integration-alias-{uuid.uuid4().hex}"
        key: Final = scenario.key(models=[allowed, hidden], aliases={alias: hidden})
        customer: Final = _customer(scenario, models=[allowed])
        response: Final = _customer_request(gateway, key, alias, customer, wire)
        assert _denial_type(response) == "customer_model_access_denied", response.text


def test_team_alias_to_customer_allowed_model_is_served(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        target: Final = scenario.model(api_base=f"{wire.url}/v1")
        alias: Final = f"integration-team-alias-{uuid.uuid4().hex}"
        team: Final = scenario.team(models=[target], model_aliases={alias: target})
        key: Final = scenario.key(team_id=team)
        customer: Final = _customer(scenario, models=[target])
        response: Final = _customer_request(gateway, key, alias, customer, wire)
        assert response.status_code == 200, response.text


def test_customer_wildcard_matches_provider_prefix_and_key_semantics(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        provider_model: Final = "openai/gpt-4o-mini"
        created: Final = gateway.post(
            "/model/new",
            {
                "model_name": provider_model,
                "litellm_params": {
                    "model": provider_model,
                    "api_key": "integration-provider-key",
                    "api_base": f"{wire.url}/v1",
                },
            },
        )
        scenario.cleanups.callback(scenario.delete_model, string_value(object_value(created["model_info"])["id"]))
        alias: Final = scenario.model(api_base=f"{wire.url}/v1")
        wildcard_key: Final = scenario.key(models=["openai/*"])

        key_provider: Final = _customer_request(gateway, wildcard_key, provider_model, None, wire)
        assert key_provider.status_code == 200, key_provider.text
        key_alias: Final = _customer_request(gateway, wildcard_key, alias, None, wire)
        assert _denial_type(key_alias) == "key_model_access_denied", key_alias.text

        customer_key: Final = scenario.key(models=[provider_model, alias])
        customer: Final = _customer(scenario, models=["openai/*"])
        customer_provider: Final = _customer_request(gateway, customer_key, provider_model, customer, wire)
        assert customer_provider.status_code == 200, customer_provider.text
        customer_alias: Final = _customer_request(gateway, customer_key, alias, customer, wire)
        assert (customer_provider.status_code, customer_alias.status_code) == (
            key_provider.status_code,
            key_alias.status_code,
        ), f"key: {key_provider.text} {key_alias.text}; customer: {customer_provider.text} {customer_alias.text}"
        assert _denial_type(customer_alias) == "customer_model_access_denied", customer_alias.text


def test_customer_access_group_matches_key_access_group_semantics(gateway: Gateway) -> None:
    with wire_server(_chat_reply) as wire, gateway.scenario() as scenario:
        access_group: Final = f"integration-access-group-{uuid.uuid4().hex}"
        member: Final = scenario.model(api_base=f"{wire.url}/v1", model_info={"access_groups": [access_group]})
        other: Final = scenario.model(api_base=f"{wire.url}/v1")
        group_key: Final = scenario.key(models=[access_group])
        key_member: Final = _customer_request(gateway, group_key, member, None, wire)
        assert key_member.status_code == 200, key_member.text
        key_other: Final = _customer_request(gateway, group_key, other, None, wire)
        assert _denial_type(key_other) == "key_model_access_denied", key_other.text

        customer_key: Final = scenario.key(models=[member, other])
        customer: Final = _customer(scenario, models=[access_group])
        customer_member: Final = _customer_request(gateway, customer_key, member, customer, wire)
        assert customer_member.status_code == 200, customer_member.text
        customer_other: Final = _customer_request(gateway, customer_key, other, customer, wire)
        assert (customer_member.status_code, customer_other.status_code) == (
            key_member.status_code,
            key_other.status_code,
        ), f"key: {key_member.text} {key_other.text}; customer: {customer_member.text} {customer_other.text}"
        assert _denial_type(customer_other) == "customer_model_access_denied", customer_other.text


def test_new_customer_allowlist_is_enforced_when_registry_cache_is_stale(gateway: Gateway) -> None:
    with (
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as redis_cache,
        wire_server(_chat_reply) as wire,
        gateway.scenario() as scenario,
    ):
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        disallowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, disallowed])
        baseline_customer: Final = _customer(scenario, models=[allowed])
        registry_key, stale_registry, registry_ids = _observe_registry_after_normal_load(
            gateway,
            key,
            allowed,
            baseline_customer,
            wire,
            redis_cache,
        )

        customer: Final = _customer(scenario, models=[allowed])
        assert customer not in registry_ids, registry_ids
        assert redis_cache.set(registry_key, stale_registry)
        scenario.cleanups.callback(redis_cache.delete, registry_key)

        denied: Final = _customer_request(gateway, key, disallowed, customer, wire)
        assert _denial_type(denied) == "customer_model_access_denied", denied.text
        served: Final = _customer_request(gateway, key, allowed, customer, wire)
        assert served.status_code == 200, served.text


def test_updated_customer_allowlist_is_enforced_when_registry_cache_is_stale(gateway: Gateway) -> None:
    with (
        Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]), decode_responses=True) as redis_cache,
        wire_server(_chat_reply) as wire,
        gateway.scenario() as scenario,
    ):
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        disallowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, disallowed])
        customer: Final = _customer(scenario, models=[])
        baseline_customer: Final = _customer(scenario, models=[allowed])
        registry_key, stale_registry, registry_ids = _observe_registry_after_normal_load(
            gateway,
            key,
            allowed,
            baseline_customer,
            wire,
            redis_cache,
        )

        updated: Final = gateway.post("/customer/update", {"user_id": customer, "models": [allowed]})
        assert updated["models"] == [allowed], updated
        assert customer not in registry_ids, registry_ids
        assert redis_cache.set(registry_key, stale_registry)
        scenario.cleanups.callback(redis_cache.delete, registry_key)

        denied: Final = _customer_request(gateway, key, disallowed, customer, wire)
        assert _denial_type(denied) == "customer_model_access_denied", denied.text
        served: Final = _customer_request(gateway, key, allowed, customer, wire)
        assert served.status_code == 200, served.text

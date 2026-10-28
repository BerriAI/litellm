import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final, Literal

import httpx
import pytest
import yaml
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, Scenario, object_value, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
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
    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
}


def _customer(scenario: Scenario, models: list[str] | None = None) -> str:
    identity: Final = f"integration-customer-{uuid.uuid4().hex}"
    body: Final[dict[str, JsonValue]] = {
        "user_id": identity,
        **({"models": models} if models is not None else {}),
    }
    scenario.gateway.post("/customer/new", body)
    scenario.cleanups.callback(scenario.gateway.post, "/customer/delete", {"user_ids": [identity]})
    return identity


def _reply(request: Request) -> Reply:
    if request.method != "POST" or not request.body:
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = _JSON_OBJECT.validate_json(request.body)
    response: Final = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "model": string_value(body["model"]),
        **_CHAT_RESPONSE,
    }
    return Reply(body=json.dumps(response).encode())


def _fallback_reply(request: Request) -> Reply:
    if request.method == "POST" and request.body:
        body: Final = _JSON_OBJECT.validate_json(request.body)
        if "scripted-primary-" in string_value(body["model"]):
            return Reply(status=500, body=b'{"error":{"message":"scripted primary failure","type":"server_error"}}')
    return _reply(request)


def _chat(
    gateway: Gateway,
    key: str,
    model: str,
    marker: str,
    *,
    customer: str | None = None,
    headers: Mapping[str, str] | None = None,
    fallbacks: tuple[str, ...] | None = None,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": [{"role": "user", "content": marker}],
        **({"user": customer} if customer is not None else {}),
        **({"fallbacks": list(fallbacks)} if fallbacks is not None else {}),
    }
    return gateway.request("POST", "/v1/chat/completions", body, key=key, headers=headers)


def _error_type(response: httpx.Response) -> str:
    error: Final = object_value(_JSON_OBJECT.validate_json(response.content)["error"])
    return string_value(error["type"])


def _assert_upstream(wire: Wire, marker: str, expected: int) -> tuple[Request, ...]:
    matching: Final = tuple(request for request in wire.drain() if marker.encode() in request.body)
    assert len(matching) == expected, matching
    return matching


@pytest.mark.parametrize(
    "customer_state",
    ("without_models", "empty_models", "no_customer_id", "unknown_customer_id"),
)
def test_unrestricted_customers_keep_key_model_access(
    gateway: Gateway,
    customer_state: Literal["without_models", "empty_models", "no_customer_id", "unknown_customer_id"],
) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        first: Final = scenario.model(api_base=f"{wire.url}/v1")
        second: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[first, second])
        customer: Final = (
            _customer(scenario)
            if customer_state == "without_models"
            else _customer(scenario, [])
            if customer_state == "empty_models"
            else f"missing-{uuid.uuid4().hex}"
            if customer_state == "unknown_customer_id"
            else None
        )

        for model in (first, second):
            marker: Final = uuid.uuid4().hex
            response: Final = _chat(gateway, key, model, marker, customer=customer)
            assert response.status_code == 200, response.text
            assert len(_assert_upstream(wire, marker, 1)) == 1


def test_customer_and_key_model_lists_are_both_enforced(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model_a: Final = scenario.model(api_base=f"{wire.url}/v1")
        model_b: Final = scenario.model(api_base=f"{wire.url}/v1")
        model_c: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model_a, model_b])
        headers: Final = {"x-litellm-end-user-id": _customer(scenario, [model_b, model_c])}

        marker_a: Final = uuid.uuid4().hex
        denied_by_customer: Final = _chat(gateway, key, model_a, marker_a, headers=headers)
        assert denied_by_customer.status_code == 403, denied_by_customer.text
        assert _error_type(denied_by_customer) == "customer_model_access_denied"
        assert _assert_upstream(wire, marker_a, 0) == ()

        marker_b: Final = uuid.uuid4().hex
        allowed: Final = _chat(gateway, key, model_b, marker_b, headers=headers)
        assert allowed.status_code == 200, allowed.text
        assert len(_assert_upstream(wire, marker_b, 1)) == 1

        marker_c: Final = uuid.uuid4().hex
        denied_by_key: Final = _chat(gateway, key, model_c, marker_c, headers=headers)
        assert denied_by_key.status_code == 403, denied_by_key.text
        assert _error_type(denied_by_key) == "key_model_access_denied"
        assert _assert_upstream(wire, marker_c, 0) == ()


def test_request_body_fallback_outside_customer_allowlist_is_denied(gateway: Gateway) -> None:
    with wire_server(_fallback_reply) as wire, gateway.scenario() as scenario:
        primary: Final = scenario.model(api_base=f"{wire.url}/v1")
        fallback: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[primary, fallback])
        customer: Final = _customer(scenario, [primary])
        marker: Final = uuid.uuid4().hex

        response: Final = _chat(
            gateway,
            key,
            primary,
            marker,
            customer=customer,
            fallbacks=(fallback,),
        )
        assert response.status_code == 403, response.text
        assert _error_type(response) == "customer_model_access_denied"
        assert _assert_upstream(wire, marker, 0) == ()


def _router_config(
    directory: Path,
    wire: Wire,
    primary: str,
    fallback: str,
    allowed_fallback: str,
    primary_provider: str,
    fallback_provider: str,
    allowed_provider: str,
    *,
    fallback_target: str,
    enforce: bool,
) -> Path:
    models: Final = (
        (primary, primary_provider),
        (fallback, fallback_provider),
        (allowed_fallback, allowed_provider),
    )
    config: Final = {
        "model_list": [
            {
                "model_name": name,
                "litellm_params": {
                    "model": provider_model,
                    "api_key": "integration-provider-key",
                    "api_base": f"{wire.url}/v1",
                },
            }
            for name, provider_model in models
        ],
        "general_settings": {
            "master_key": "os.environ/LITELLM_MASTER_KEY",
            "store_model_in_db": False,
            "enforce_fallback_model_access": enforce,
        },
        "litellm_settings": {"cache": False},
        "router_settings": {
            "num_retries": 0,
            "disable_cooldowns": True,
            "fallbacks": [{primary: [fallback_target]}],
        },
    }
    path: Final = directory / f"customer-fallback-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@pytest.mark.parametrize(
    ("enforce", "allow_fallback", "expected_status", "expected_upstream_count"),
    (
        (True, False, 500, 1),
        (False, False, 200, 2),
        (True, True, 200, 2),
    ),
)
def test_router_config_fallback_customer_allowlist(
    gateway: Gateway,
    tmp_path: Path,
    enforce: bool,
    allow_fallback: bool,
    expected_status: int,
    expected_upstream_count: int,
) -> None:
    with wire_server(_fallback_reply) as wire, gateway.scenario() as model_scenario:
        primary_provider: Final = f"openai/scripted-primary-{uuid.uuid4().hex}"
        fallback_provider: Final = f"openai/scripted-fallback-{uuid.uuid4().hex}"
        allowed_provider: Final = f"openai/scripted-allowed-{uuid.uuid4().hex}"
        primary: Final = model_scenario.model(api_base=f"{wire.url}/v1", model=primary_provider)
        fallback: Final = model_scenario.model(api_base=f"{wire.url}/v1", model=fallback_provider)
        allowed_fallback: Final = model_scenario.model(api_base=f"{wire.url}/v1", model=allowed_provider)
        fallback_target: Final = allowed_fallback if allow_fallback else fallback
        config: Final = _router_config(
            tmp_path,
            wire,
            primary,
            fallback,
            allowed_fallback,
            primary_provider,
            fallback_provider,
            allowed_provider,
            fallback_target=fallback_target,
            enforce=enforce,
        )

        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            with candidate.scenario() as scenario:
                key: Final = scenario.key(models=[primary, fallback, allowed_fallback])
                customer_models: Final = [primary, fallback_target] if allow_fallback else [primary]
                customer: Final = _customer(scenario, customer_models)
                marker: Final = uuid.uuid4().hex
                response: Final = _chat(candidate, key, primary, marker, customer=customer)

                assert response.status_code == expected_status, response.text
                requests: Final = _assert_upstream(wire, marker, expected_upstream_count)
                models: Final = tuple(
                    string_value(_JSON_OBJECT.validate_json(request.body)["model"]) for request in requests
                )
                assert "scripted-primary-" in models[0]
                if expected_upstream_count == 2:
                    assert ("scripted-allowed-" in models[1]) is allow_fallback
                    assert ("scripted-fallback-" in models[1]) is not allow_fallback


def test_customer_crud_sets_and_clears_models_immediately(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        model_a: Final = scenario.model(api_base=f"{wire.url}/v1")
        model_b: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model_a, model_b])
        customer: Final = _customer(scenario)

        assert gateway.get("/customer/info", {"end_user_id": customer})["models"] == []
        gateway.post("/customer/update", {"user_id": customer, "models": [model_a]})
        assert gateway.get("/customer/info", {"end_user_id": customer})["models"] == [model_a]

        denied_marker: Final = uuid.uuid4().hex
        denied: Final = _chat(gateway, key, model_b, denied_marker, customer=customer)
        assert denied.status_code == 403, denied.text
        assert _error_type(denied) == "customer_model_access_denied"
        assert _assert_upstream(wire, denied_marker, 0) == ()

        gateway.post("/customer/update", {"user_id": customer, "models": []})
        assert gateway.get("/customer/info", {"end_user_id": customer})["models"] == []
        allowed_marker: Final = uuid.uuid4().hex
        allowed: Final = _chat(gateway, key, model_b, allowed_marker, customer=customer)
        assert allowed.status_code == 200, allowed.text
        assert len(_assert_upstream(wire, allowed_marker, 1)) == 1

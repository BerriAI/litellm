import json
import os
import uuid
from pathlib import Path
from typing import Final

import httpx
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
            "message": {"role": "assistant", "content": "scripted fallback response"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
}


def _customer(scenario: Scenario, models: tuple[str, ...]) -> str:
    identity: Final = f"integration-customer-{uuid.uuid4().hex}"
    scenario.gateway.post("/customer/new", {"user_id": identity, "models": list(models)})
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


def _assert_requests(wire: Wire, marker: str, expected: int) -> tuple[Request, ...]:
    requests: Final = wire.drain()
    matching: Final = tuple(request for request in requests if marker.encode() in request.body)
    assert len(matching) == expected, matching
    return matching


def _chat(
    gateway: Gateway,
    key: str,
    model: str | None,
    customer: str,
    marker: str,
    fallbacks: tuple[str, ...],
) -> httpx.Response:
    body: Final = {
        "messages": [{"role": "user", "content": marker}],
        "user": customer,
        "fallbacks": list(fallbacks),
        **({"model": model} if model is not None else {}),
    }
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


def test_client_fallback_outside_customer_allowlist_is_denied_up_front(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        primary: Final = scenario.model(api_base=f"{wire.url}/v1")
        fallback: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[primary, fallback])
        customer: Final = _customer(scenario, (primary,))
        marker: Final = uuid.uuid4().hex
        response: Final = _chat(gateway, key, primary, customer, marker, (fallback,))
        assert response.status_code == 403, response.text
        assert "customer_model_access_denied" in response.text, response.text
        _assert_requests(wire, marker, 0)


def test_client_fallback_inside_customer_allowlist_is_served(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        primary: Final = scenario.model(api_base=f"{wire.url}/v1")
        fallback: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[primary, fallback])
        customer: Final = _customer(scenario, (primary,))
        marker: Final = uuid.uuid4().hex
        response: Final = _chat(gateway, key, primary, customer, marker, (primary,))
        assert response.status_code == 200, response.text
        _assert_requests(wire, marker, 1)


def test_fallback_without_primary_has_distinct_base_and_head_errors(gateway: Gateway) -> None:
    with wire_server(_reply) as wire, gateway.scenario() as scenario:
        allowed: Final = scenario.model(api_base=f"{wire.url}/v1")
        fallback: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[allowed, fallback])
        customer: Final = _customer(scenario, (allowed,))
        marker: Final = uuid.uuid4().hex
        response: Final = _chat(gateway, key, None, customer, marker, (fallback,))
        error: Final = object_value(object_value(response.json())["error"])
        audit_leg: Final = os.environ.get("LITELLM_CUSTOMER_ALLOWLIST_AUDIT_LEG", "head")
        if audit_leg == "base":
            assert response.status_code == 400, response.text
            assert error["type"] == "invalid_request_error", response.text
        else:
            assert response.status_code == 403, response.text
            assert error["type"] == "customer_model_access_denied", response.text
        _assert_requests(wire, marker, 0)


def _router_config(
    directory: Path,
    wire: Wire,
    *,
    enforce: bool,
    include_allowed_target: bool,
) -> tuple[Path, str, str, str | None]:
    primary: Final = f"m1-{uuid.uuid4().hex}"
    fallback: Final = f"m2-{uuid.uuid4().hex}"
    allowed_target: Final = f"m3-{uuid.uuid4().hex}" if include_allowed_target else None
    models: Final = (
        (primary, "fallback-primary-" + uuid.uuid4().hex),
        (fallback, "fallback-target-" + uuid.uuid4().hex),
    ) + (((allowed_target, "fallback-allowed-" + uuid.uuid4().hex),) if allowed_target is not None else ())
    config: Final = {
        "model_list": [
            {
                "model_name": name,
                "litellm_params": {
                    "model": "openai/" + upstream,
                    "api_key": "integration-provider-key",
                    "api_base": f"{wire.url}/v1",
                },
            }
            for name, upstream in models
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
            "fallbacks": [{primary: [fallback]}],
        },
    }
    path: Final = directory / f"customer-fallback-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path, primary, fallback, allowed_target


def _router_reply(request: Request) -> Reply:
    if request.method != "POST" or not request.body:
        return Reply(body=b'{"object":"list","data":[]}')
    body: Final = _JSON_OBJECT.validate_json(request.body)
    if "fallback-primary-" in string_value(body["model"]):
        return Reply(status=500, body=b'{"error":{"message":"scripted primary failure","type":"server_error"}}')
    return _reply(request)


def _router_request(
    candidate: Gateway,
    key: str,
    primary: str,
    customer: str,
    marker: str,
) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": primary,
            "messages": [{"role": "user", "content": marker}],
            "user": customer,
        },
        key=key,
    )


def test_router_fallback_is_not_attempted_when_customer_access_is_enforced(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_router_reply) as wire:
        config, primary, fallback, _ = _router_config(tmp_path, wire, enforce=True, include_allowed_target=False)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            with candidate.scenario() as scenario:
                key: Final = scenario.key(models=[primary, fallback])
                customer: Final = _customer(scenario, (primary,))
                marker: Final = uuid.uuid4().hex
                response: Final = _router_request(candidate, key, primary, customer, marker)
                assert response.status_code == 500, response.text
                assert "scripted primary failure" in response.text, response.text
                requests: Final = wire.drain()
                matching: Final = tuple(request for request in requests if marker.encode() in request.body)
                assert len(matching) == 1, matching
                assert "fallback-primary-" in string_value(_JSON_OBJECT.validate_json(matching[0].body)["model"])


def test_router_fallback_is_attempted_when_customer_access_enforcement_is_disabled(
    gateway: Gateway, tmp_path: Path
) -> None:
    with wire_server(_router_reply) as wire:
        config, primary, fallback, _ = _router_config(tmp_path, wire, enforce=False, include_allowed_target=False)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            with candidate.scenario() as scenario:
                key: Final = scenario.key(models=[primary, fallback])
                customer: Final = _customer(scenario, (primary,))
                marker: Final = uuid.uuid4().hex
                response: Final = _router_request(candidate, key, primary, customer, marker)
                assert response.status_code == 200, response.text
                requests: Final = wire.drain()
                matching: Final = tuple(request for request in requests if marker.encode() in request.body)
                models: Final = tuple(
                    string_value(_JSON_OBJECT.validate_json(request.body)["model"]) for request in matching
                )
                assert len(models) == 2, models
                assert all(marker.encode() in request.body for request in matching), matching
                assert "fallback-primary-" in models[0], models
                assert "fallback-target-" in models[1], models


def test_router_fallback_to_customer_allowed_model_is_served(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_router_reply) as wire:
        config, primary, fallback, allowed_target = _router_config(
            tmp_path, wire, enforce=True, include_allowed_target=True
        )
        assert allowed_target is not None
        config_data: Final = _JSON_OBJECT.validate_json(json.dumps(yaml.safe_load(config.read_text())))
        router_settings: Final = object_value(config_data["router_settings"])
        config.write_text(
            yaml.safe_dump(
                {
                    **config_data,
                    "router_settings": {
                        **router_settings,
                        "fallbacks": [{primary: [allowed_target]}],
                    },
                }
            )
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate:
            with candidate.scenario() as scenario:
                key: Final = scenario.key(models=[primary, fallback, allowed_target])
                customer: Final = _customer(scenario, (primary,))
                candidate.post("/customer/update", {"user_id": customer, "models": [primary, allowed_target]})
                marker: Final = uuid.uuid4().hex
                response: Final = _router_request(candidate, key, primary, customer, marker)
                assert response.status_code == 200, response.text
                requests: Final = wire.drain()
                matching: Final = tuple(request for request in requests if marker.encode() in request.body)
                models: Final = tuple(
                    string_value(_JSON_OBJECT.validate_json(request.body)["model"]) for request in matching
                )
                assert len(models) == 2, models
                assert all(marker.encode() in request.body for request in matching), matching
                assert "fallback-primary-" in models[0], models
                assert "fallback-allowed-" in models[1], models

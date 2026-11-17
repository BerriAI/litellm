from __future__ import annotations

import json
import uuid
from types import MappingProxyType
from typing import Final

from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.wire import Reply, Request, Wire, wire_server

FRESH_CONNECTION: Final = MappingProxyType({"Connection": "close"})
IMAGE_RESPONSE: Final = {
    "created": 1,
    "background": None,
    "data": [
        {
            "b64_json": None,
            "revised_prompt": None,
            "url": "https://images.example/generated.png",
            "provider_specific_fields": None,
        }
    ],
    "output_format": None,
    "quality": None,
    "size": None,
    "usage": {
        "total_tokens": 0,
        "input_tokens": 0,
        "input_tokens_details": {"image_tokens": 0, "text_tokens": 0},
        "output_tokens": 0,
    },
}


def _primary_unavailable(_: Request) -> Reply:
    return Reply(
        status=503,
        body=json.dumps({"error": {"message": "primary unavailable", "type": "server_error"}}).encode(),
    )


def _image_reply(_: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "created": 1,
                "data": [{"url": "https://images.example/generated.png"}],
            }
        ).encode()
    )


def _image_model(scenario: Scenario, group: str, wire: Wire) -> str:
    return scenario.model(
        name=group,
        model="openai/dall-e-3",
        api_base=f"{wire.url}/v1",
        api_key="integration-provider-key",
        num_retries=0,
    )


def _assert_image_fallback(
    gateway: Gateway,
    key: str,
    primary: Wire,
    fallback: Wire,
    primary_model: str,
) -> None:
    response: Final = gateway.request(
        "POST",
        "/v1/images/generations",
        {"model": primary_model, "prompt": "draw a tree"},
        key=key,
        headers=FRESH_CONNECTION,
    )

    assert response.status_code == 200, response.text
    assert response.json() == IMAGE_RESPONSE, response.text
    assert tuple((request.method, request.target) for request in primary.drain()) == (
        ("POST", "/v1/images/generations"),
    ), response.text
    assert tuple((request.method, request.target) for request in fallback.drain()) == (
        ("POST", "/v1/images/generations"),
    ), response.text


def test_key_router_settings_fallback_routes_image_generation() -> None:
    with (
        wire_server(_primary_unavailable) as primary_wire,
        wire_server(_image_reply) as fallback_wire,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
    ):
        primary_model: Final = _image_model(scenario, f"image-primary-{uuid.uuid4().hex}", primary_wire)
        fallback_model: Final = _image_model(scenario, f"image-fallback-{uuid.uuid4().hex}", fallback_wire)
        key: Final = scenario.key(router_settings={"fallbacks": [{primary_model: [fallback_model]}], "num_retries": 0})

        _assert_image_fallback(gateway, key, primary_wire, fallback_wire, primary_model)


def test_team_router_settings_fallback_routes_image_generation() -> None:
    with (
        wire_server(_primary_unavailable) as primary_wire,
        wire_server(_image_reply) as fallback_wire,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
    ):
        primary_model: Final = _image_model(scenario, f"image-primary-{uuid.uuid4().hex}", primary_wire)
        fallback_model: Final = _image_model(scenario, f"image-fallback-{uuid.uuid4().hex}", fallback_wire)
        team: Final = scenario.team(
            router_settings={"fallbacks": [{primary_model: [fallback_model]}], "num_retries": 0}
        )
        key: Final = scenario.key(team_id=team)

        _assert_image_fallback(gateway, key, primary_wire, fallback_wire, primary_model)


def test_key_model_group_alias_resolves_once_for_image_generation() -> None:
    with (
        wire_server(_image_reply) as primary_wire,
        wire_server(_image_reply) as target_wire,
        wire_server(_image_reply) as chained_target_wire,
        gateway_from_environment() as gateway,
        gateway.scenario() as scenario,
    ):
        primary_model: Final = _image_model(scenario, f"image-primary-{uuid.uuid4().hex}", primary_wire)
        target_model: Final = _image_model(scenario, f"image-target-{uuid.uuid4().hex}", target_wire)
        chained_target_model: Final = _image_model(
            scenario, f"image-chained-target-{uuid.uuid4().hex}", chained_target_wire
        )
        alias_name: Final = f"image-alias-{uuid.uuid4().hex}"
        key: Final = scenario.key(
            router_settings={
                "model_group_alias": {
                    alias_name: target_model,
                    target_model: chained_target_model,
                }
            }
        )

        response: Final = gateway.request(
            "POST",
            "/v1/images/generations",
            {"model": alias_name, "prompt": "draw a tree"},
            key=key,
            headers=FRESH_CONNECTION,
        )
        primary_requests: Final = tuple((request.method, request.target) for request in primary_wire.drain())
        target_requests: Final = tuple((request.method, request.target) for request in target_wire.drain())
        chained_target_requests: Final = tuple(
            (request.method, request.target) for request in chained_target_wire.drain()
        )

        assert alias_name not in {primary_model, target_model, chained_target_model}, response.text
        assert response.status_code == 200, response.text
        assert response.json() == IMAGE_RESPONSE, response.text
        assert primary_requests == (), response.text
        assert target_requests == (("POST", "/v1/images/generations"),), (
            f"{response.text}; chained target requests: {chained_target_requests}"
        )
        assert chained_target_requests == (), response.text

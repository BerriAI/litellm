"""Encrypted-content affinity when the origin deployment has no encryption-boundary peer.

A multi-region model group has several deployments sharing one upstream api_key but a
distinct api_base each, with ``optional_pre_call_checks: [encrypted_content_affinity]``
and ``disable_cooldowns: true`` (the integration proxy config). A follow-up
``POST /v1/responses`` that replays a reasoning item must never fail at the proxy while
sibling deployments in the same group are healthy: when the origin cannot serve the turn
its encrypted reasoning should be stripped and the request dispatched to a sibling.
"""

import uuid
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, eventually, object_value, string_value
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import JsonResponse
from pydantic import JsonValue

AFFINITY_CHECK: Final = "encrypted_content_affinity"
PROVIDER_MODEL: Final = "openai/gpt-5"
PROVIDER_KEY: Final = "integration-provider-key"
DEPLOYMENT_COUNT: Final = 3


def _responses_payload() -> dict[str, JsonValue]:
    return {
        "id": "resp_$REQUEST_ID",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5-scripted",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_$REQUEST_ID",
                "summary": [],
                "encrypted_content": "ZHNra2RrZA==",
            },
            {
                "type": "message",
                "id": "msg_$REQUEST_ID",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "scripted answer"}],
            },
        ],
        "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
    }


def _enable_affinity_check(scenario: Scenario) -> None:
    gateway: Final = scenario.gateway
    current: Final = object_value(gateway.get("/router/settings")["current_values"]).get("optional_pre_call_checks")
    original: Final = list(current) if isinstance(current, list) else []
    scenario.cleanups.callback(
        lambda: gateway.post("/config/update", {"router_settings": {"optional_pre_call_checks": original}})
    )
    gateway.post(
        "/config/update",
        {"router_settings": {"optional_pre_call_checks": [*original, AFFINITY_CHECK], "num_retries": 0}},
    )


def _multi_region_group(scenario: Scenario) -> tuple[str, tuple[str, ...]]:
    """Three deployments in one model group: one shared api_key, a distinct api_base each."""
    gateway: Final = scenario.gateway
    group: Final = f"enc-affinity-{uuid.uuid4().hex}"
    handles: Final = tuple(
        register_scenario(
            f"{group}-{index}",
            JsonResponse(content_type="application/json", body=_responses_payload()),
        )
        for index in range(DEPLOYMENT_COUNT)
    )
    for handle in handles:
        scenario.cleanups.callback(delete_scenario, handle)

    def delete_model_if_present(model_id: str) -> None:
        entries: Final = gateway.get("/model/info")["data"]
        assert isinstance(entries, list)
        if any(object_value(object_value(entry)["model_info"])["id"] == model_id for entry in entries):
            scenario.delete_model(model_id)

    deployment_ids: Final = tuple(
        string_value(
            object_value(
                gateway.post(
                    "/model/new",
                    {
                        "model_name": group,
                        "litellm_params": {
                            "model": PROVIDER_MODEL,
                            "api_key": PROVIDER_KEY,
                            "api_base": handle.api_base(),
                        },
                        "model_info": {},
                    },
                )["model_info"]
            )["id"]
        )
        for handle in handles
    )
    for model_id in deployment_ids:
        scenario.cleanups.callback(delete_model_if_present, model_id)
    return group, deployment_ids


def _user_message(text: str) -> dict[str, JsonValue]:
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _responses_turn(gateway: Gateway, group: str, request_input: JsonValue) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/responses",
        {"model": group, "input": request_input, "store": False, "include": ["reasoning.encrypted_content"]},
    )


def _turn_one(gateway: Gateway, group: str, deployment_ids: tuple[str, ...]) -> tuple[str, list[JsonValue]]:
    # the proxy caches responses, so the turn-one prompt needs a unique marker or a
    # stale body carrying another run's encoded model_id would replay instead
    response: Final = _responses_turn(gateway, group, f"hello affinity {uuid.uuid4().hex}")
    assert response.status_code == 200, response.text
    origin: Final = str(response.headers["x-litellm-model-id"])
    assert origin in deployment_ids, f"turn one served by unknown deployment {origin}: {response.text}"
    output: Final = object_value(response.json()).get("output")
    assert isinstance(output, list), f"turn one returned no output items: {response.text}"
    reasoning: Final = next((item for item in output if object_value(item).get("type") == "reasoning"), None)
    assert isinstance(reasoning, dict), f"turn one returned no reasoning item: {response.text}"
    assert str(object_value(reasoning)["id"]).startswith("encitem_"), (
        f"affinity encoding did not run on turn one: {reasoning}"
    )
    assert isinstance(object_value(reasoning).get("encrypted_content"), str), (
        f"turn one reasoning item has no encrypted_content: {reasoning}"
    )
    message: Final = next((item for item in output if object_value(item).get("type") == "message"), None)
    assert isinstance(message, dict), f"turn one returned no message item: {response.text}"
    return origin, [reasoning, message]


def _replay(gateway: Gateway, group: str, items: list[JsonValue]) -> httpx.Response:
    return _responses_turn(
        gateway,
        group,
        [_user_message("hello affinity"), *items, _user_message("continue the conversation")],
    )


def _model_blocked(gateway: Gateway, model_id: str) -> bool:
    entries: Final = gateway.get("/model/info")["data"]
    assert isinstance(entries, list)
    entry: Final = next(
        (entry for entry in entries if object_value(object_value(entry)["model_info"])["id"] == model_id),
        None,
    )
    return entry is not None and object_value(object_value(entry)["model_info"]).get("blocked") is True


def test_replayed_encrypted_content_serves_from_sibling_when_origin_blocked(gateway: Gateway) -> None:
    """Origin excluded from healthy deployments (admin-blocked, no cooldown) must not 503.

    On unfixed code the origin is a routed-group candidate with no (api_base, api_key)
    peer, so the affinity check raises a proxy-level 503 instead of stripping the
    encrypted reasoning and dispatching to a healthy sibling.
    """
    with gateway.scenario() as scenario:
        _enable_affinity_check(scenario)
        group, deployment_ids = _multi_region_group(scenario)
        origin, items = _turn_one(gateway, group, deployment_ids)

        gateway.post("/model/block", {"model_id": origin})
        eventually(lambda: _model_blocked(gateway, origin), lambda blocked: blocked)

        response: Final = _replay(gateway, group, items)
        assert response.status_code == 200, response.text
        siblings: Final = tuple(model_id for model_id in deployment_ids if model_id != origin)
        assert response.headers.get("x-litellm-model-id") in siblings, (
            f"turn two served by {response.headers.get('x-litellm-model-id')}, "
            f"expected a sibling of blocked origin {origin}: {response.text}"
        )


def test_replayed_encrypted_content_serves_from_sibling_when_origin_deleted(gateway: Gateway) -> None:
    """Origin permanently removed: strip-and-dispatch to a sibling, the shipped behavior."""
    with gateway.scenario() as scenario:
        _enable_affinity_check(scenario)
        group, deployment_ids = _multi_region_group(scenario)
        origin, items = _turn_one(gateway, group, deployment_ids)

        gateway.post("/model/delete", {"id": origin})
        eventually(
            lambda: gateway.get("/model/info")["data"],
            lambda entries: (
                isinstance(entries, list)
                and all(object_value(object_value(entry)["model_info"])["id"] != origin for entry in entries)
            ),
        )

        response: Final = _replay(gateway, group, items)
        assert response.status_code == 200, response.text
        siblings: Final = tuple(model_id for model_id in deployment_ids if model_id != origin)
        assert response.headers.get("x-litellm-model-id") in siblings, (
            f"turn two served by {response.headers.get('x-litellm-model-id')}, "
            f"expected a sibling of deleted origin {origin}: {response.text}"
        )

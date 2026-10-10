from __future__ import annotations

import asyncio
import json
import uuid
from typing import Final

import websockets
from integration._support.client import Gateway, Scenario, gateway_from_environment
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import RealtimeResponse
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
OBSERVATIONS: Final = TypeAdapter(list[dict[str, JsonValue]])
BLOCKED_PHRASE: Final = "realtime-blocked-phrase"
MODEL: Final = "openai/gpt-realtime"


def _register_guardrail(gateway: Gateway, scenario: Scenario, name: str) -> None:
    response: Final = gateway.request(
        "POST",
        "/guardrails",
        {
            "guardrail": {
                "guardrail_name": name,
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": "pre_call",
                    "default_on": True,
                    "realtime_violation_message": "That request was blocked by the content filter.",
                    "blocked_words": [{"keyword": BLOCKED_PHRASE, "action": "BLOCK"}],
                },
            }
        },
    )
    assert response.status_code == 200, response.text
    guardrail_id: Final = JSON_OBJECT.validate_json(response.content).get("guardrail_id")
    assert isinstance(guardrail_id, str), response.text
    scenario.cleanups.callback(gateway.request, "DELETE", f"/guardrails/{guardrail_id}")


def _register_model(gateway: Gateway, scenario: Scenario) -> tuple[str, str]:
    provider_key: Final = f"realtime-{uuid.uuid4().hex}"
    handle: Final = register_scenario(
        provider_key,
        RealtimeResponse(
            content_type="application/x-realtime",
            events=(
                {
                    "type": "response.done",
                    "response": {"id": "resp-scripted", "status": "completed", "output": []},
                },
            ),
        ),
        control_url=gateway.upstream_url,
    )
    scenario.cleanups.callback(delete_scenario, handle)
    model: Final = scenario.model(model=MODEL, api_key=provider_key)
    return model, provider_key


async def _exchange(
    gateway: Gateway,
    model: str,
    key: str,
    text: str,
) -> tuple[dict[str, JsonValue], ...]:
    url: Final = f"{str(gateway.client.base_url).replace('http://', 'ws://').rstrip('/')}/v1/realtime?model={model}"
    async with websockets.connect(url, additional_headers={"Authorization": f"Bearer {key}"}) as socket:
        created: Final = JSON_OBJECT.validate_json(await socket.recv())
        assert created.get("type") == "session.created", created
        await socket.send(
            json.dumps(
                {
                    "type": "conversation.item.create",
                    "item": {"role": "user", "content": [{"type": "input_text", "text": text}]},
                }
            )
        )
        await socket.send(json.dumps({"type": "response.create"}))
        return await _collect(socket, ())


async def _collect(
    socket: websockets.ClientConnection,
    events: tuple[dict[str, JsonValue], ...],
) -> tuple[dict[str, JsonValue], ...]:
    event: Final = JSON_OBJECT.validate_json(await asyncio.wait_for(socket.recv(), timeout=15))
    collected: Final = (*events, event)
    return (
        collected if event.get("type") == "response.done" or len(collected) >= 20 else await _collect(socket, collected)
    )


def _provider_frames(gateway: Gateway, provider_key: str) -> tuple[dict[str, JsonValue], ...]:
    response: Final = gateway.client.get(f"{gateway.upstream_url}/__observations")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    requests: Final = OBSERVATIONS.validate_python(body.get("requests"))
    return tuple(
        JSON_OBJECT.validate_python(request["body"])
        for request in requests
        if isinstance(request, dict)
        and request.get("method") == "WEBSOCKET_FRAME"
        and request.get("authorization") == f"Bearer {provider_key}"
        and isinstance(request.get("body"), dict)
    )


def test_blocked_realtime_text_is_replaced_before_forwarding() -> None:
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        guardrail: Final = f"realtime-guardrail-{uuid.uuid4().hex}"
        _register_guardrail(gateway, scenario, guardrail)
        model, provider_key = _register_model(gateway, scenario)
        key: Final = scenario.key(models=[model])
        events: Final = asyncio.run(_exchange(gateway, model, key, f"hello {BLOCKED_PHRASE}"))

        errors: Final = tuple(
            event
            for event in events
            if event.get("type") == "error"
            and isinstance(event.get("error"), dict)
            and event["error"].get("type") == "guardrail_violation"
        )
        assert errors, events
        assert any(event.get("type") == "response.done" for event in events), events
        frames: Final = _provider_frames(gateway, provider_key)
        assert all(BLOCKED_PHRASE not in json.dumps(frame) for frame in frames), frames
        frame_types: Final = tuple(frame.get("type") for frame in frames)
        assert frame_types.count("conversation.item.create") == 1, frames
        assert frame_types.count("response.create") == 1, frames


def test_clean_realtime_text_and_response_create_reach_provider() -> None:
    with gateway_from_environment() as gateway, gateway.scenario() as scenario:
        guardrail: Final = f"realtime-guardrail-{uuid.uuid4().hex}"
        _register_guardrail(gateway, scenario, guardrail)
        model, provider_key = _register_model(gateway, scenario)
        key: Final = scenario.key(models=[model])
        text: Final = f"clean-realtime-{uuid.uuid4().hex}"
        events: Final = asyncio.run(_exchange(gateway, model, key, text))

        assert any(event.get("type") == "response.done" for event in events), events
        frames: Final = _provider_frames(gateway, provider_key)
        assert any(frame.get("type") == "conversation.item.create" and text in json.dumps(frame) for frame in frames), (
            frames
        )
        assert any(frame.get("type") == "response.create" for frame in frames), frames

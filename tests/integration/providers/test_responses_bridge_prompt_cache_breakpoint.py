from __future__ import annotations

import uuid
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from integration._support.client import Gateway
from integration._support.wire import wire_server
from integration.providers._responses_bridge_prompt_cache_breakpoint import (
    _BREAKPOINT,
    _Call,
    _ClientKind,
    _JSON_OBJECT,
    _MODEL,
    _UNSUPPORTED_MODEL,
    _assert_spend_for_result,
    _contains_breakpoint,
    _expected_multimodal_input,
    _multimodal_chat_body,
    _prompt,
    _raw_call,
    _request_body,
    _responses_reply,
    _serve_chat,
    _simple_chat_body,
    _simple_expected_input,
)

@pytest.mark.parametrize("client_kind", ("openai_sync", "openai_async", "httpx"))
@pytest.mark.parametrize("stream", (False, True))
async def test_caller_prompt_cache_breakpoints_survive_chat_to_responses_bridge(
    gateway: Gateway,
    client_kind: _ClientKind,
    stream: bool,
) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_responses_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url + "/v1")
        body: Final = _multimodal_chat_body(model, marker, stream)
        served: Final = await _serve_chat(gateway, body, client_kind, stream)
        assert served.status == 200, served.text
        _assert_spend_for_result(served, model)
        (peer_request,) = wire.drain()
        peer_body: Final = _request_body(peer_request)
        assert peer_body["input"] == _expected_multimodal_input(marker), peer_body
        assert peer_body["prompt_cache_options"] == {"mode": "explicit"}, peer_body

def _expected_uninjected_system_bridge_body(
    marker: str,
    prompt_cache_options: dict[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {
        "input": _simple_expected_input(marker, marked=False),
        "instructions": _prompt(marker, "system"),
        "model": "gpt-5.6",
        "reasoning": {"effort": "low"},
        "stream": False,
        "tools": [
            {
                "type": "function",
                "name": "synthetic_tool",
                "parameters": {"type": "object", "properties": {}},
                "strict": None,
                "description": "Synthetic bridge test tool",
            }
        ],
        **({"prompt_cache_options": prompt_cache_options} if prompt_cache_options is not None else {}),
    }

async def test_deployment_cache_control_injection_without_options_is_unchanged(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_responses_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_base=wire.url + "/v1",
            cache_control_injection_points=[{"location": "message", "role": "system"}],
        )
        body: Final = _simple_chat_body(model, marker, system_as_string=True, marked=False)
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            served: Final = await _raw_call(client, "/v1/chat/completions", body, _Call("chat", False, marker))
        assert served.status == 200, served.text
        _assert_spend_for_result(served, model)
        (peer_request,) = wire.drain()
        peer_body: Final = _request_body(peer_request)
        assert peer_body == _expected_uninjected_system_bridge_body(marker), peer_body
        assert not _contains_breakpoint(peer_body), peer_body
        assert "prompt_cache_options" not in peer_body, peer_body

async def test_deployment_prompt_cache_options_override_is_unchanged(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    options: Final[dict[str, JsonValue]] = {"mode": "implicit", "ttl": "30m"}
    with wire_server(_responses_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=_MODEL,
            api_base=wire.url + "/v1",
            prompt_cache_options=options,
        )
        body: Final = _simple_chat_body(model, marker, system_as_string=True, marked=False)
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            served: Final = await _raw_call(client, "/v1/chat/completions", body, _Call("chat", False, marker))
        assert served.status == 200, served.text
        _assert_spend_for_result(served, model)
        (peer_request,) = wire.drain()
        peer_body: Final = _request_body(peer_request)
        assert peer_body == _expected_uninjected_system_bridge_body(marker, options), peer_body
        assert not _contains_breakpoint(peer_body), peer_body

async def test_unmarked_bridge_and_direct_responses_marker_are_forwarded_unchanged(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(_responses_reply) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=_MODEL, api_base=wire.url + "/v1")
        unmarked_body: Final = _simple_chat_body(model, marker, marked=False)
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            unmarked: Final = await _raw_call(
                client,
                "/v1/chat/completions",
                unmarked_body,
                _Call("chat", False, marker),
            )
        assert unmarked.status == 200, unmarked.text
        _assert_spend_for_result(unmarked, model)
        (unmarked_peer,) = wire.drain()
        unmarked_body_at_peer: Final = _request_body(unmarked_peer)
        assert not _contains_breakpoint(unmarked_body_at_peer), unmarked_body_at_peer
        assert "prompt_cache_options" not in unmarked_body_at_peer, unmarked_body_at_peer

        direct_marker: Final = uuid.uuid4().hex
        direct_input: Final = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": _prompt(direct_marker, "direct"),
                        "prompt_cache_breakpoint": _BREAKPOINT,
                    }
                ],
            }
        ]
        direct_body: Final = _JSON_OBJECT.validate_python({"model": model, "input": direct_input, "store": False})
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            direct: Final = await _raw_call(
                client,
                "/v1/responses",
                direct_body,
                _Call("responses", False, direct_marker),
            )
        assert direct.status == 200, direct.text
        _assert_spend_for_result(direct, model)
        (direct_peer,) = wire.drain()
        assert _request_body(direct_peer)["input"] == direct_input, _request_body(direct_peer)

@pytest.mark.parametrize("stream", (False, True))
async def test_unsupported_model_drops_breakpoints_without_rejecting_the_request(
    gateway: Gateway,
    stream: bool,
) -> None:
    marker: Final = uuid.uuid4().hex
    with (
        wire_server(lambda request: _responses_reply(request, reject_breakpoints=True)) as wire,
        gateway.scenario() as scenario,
    ):
        model: Final = scenario.model(model=_UNSUPPORTED_MODEL, api_base=wire.url + "/v1")
        body: Final = _simple_chat_body(model, marker, stream=stream)
        async with httpx.AsyncClient(
            base_url=str(gateway.client.base_url),
            headers={"Authorization": f"Bearer {gateway.key}"},
            timeout=20,
            trust_env=False,
        ) as client:
            served: Final = await _raw_call(client, "/v1/chat/completions", body, _Call("chat", stream, marker))
        assert served.status == 200, served.text
        _assert_spend_for_result(served, model)
        (peer_request,) = wire.drain()
        peer_body: Final = _request_body(peer_request)
        assert peer_body["input"] == _simple_expected_input(marker, marked=False), peer_body
        assert not _contains_breakpoint(peer_body), peer_body

import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.anthropic_sse import message_json
from integration._support.client import Gateway
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

GUARDRAIL: Final = "reject-unregistered-registered-guardrail"
OPENAI_ROUTE: Final = "/openai/v1/chat/completions"
ANTHROPIC_ROUTE: Final = "/anthropic/v1/messages"
GEMINI_ROUTE: Final = "/gemini/v1beta/models/gemini-2.5-flash:generateContent"
ROUTES: Final = (OPENAI_ROUTE, ANTHROPIC_ROUTE, GEMINI_ROUTE)
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_UPSTREAM_BODIES: Final = MappingProxyType(
    {
        OPENAI_ROUTE: chat_reply("chatcmpl-reject-unregistered", "gpt-5.6", "Hi", stream=False).body,
        ANTHROPIC_ROUTE: message_json("msg_reject_unregistered", "claude-sonnet-5-5", "Hi"),
        GEMINI_ROUTE: json.dumps(
            {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1, "totalTokenCount": 5},
            }
        ).encode(),
    }
)


def _route_body(route: str) -> dict[str, JsonValue]:
    match route:
        case "/anthropic/v1/messages":
            return {"model": "claude-sonnet-5-5", "max_tokens": 16, "messages": [{"role": "user", "content": "Hello"}]}
        case "/gemini/v1beta/models/gemini-2.5-flash:generateContent":
            return {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
        case _:
            return {"model": "gpt-5.6", "messages": [{"role": "user", "content": "Hello"}]}


def _allow(_request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _replying(body: bytes) -> Callable[[Request], Reply]:
    def respond(_request: Request) -> Reply:
        return Reply(body=body)

    return respond


@dataclass(frozen=True, slots=True)
class RejectUnregisteredRig:
    owned: OwnedProxy
    vendor: Wire
    upstreams: Mapping[str, Wire]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[RejectUnregisteredRig]:
    directory: Final = tmp_path_factory.mktemp("passthrough-reject-unregistered-guardrails")
    with ExitStack() as stack:
        vendor: Final = stack.enter_context(wire_server(_allow))
        upstreams: Final = MappingProxyType(
            {route: stack.enter_context(wire_server(_replying(body))) for route, body in _UPSTREAM_BODIES.items()}
        )
        config: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        path: Final = directory / "passthrough-reject-unregistered-guardrails.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "general_settings": {
                        **_JSON.validate_python(config.get("general_settings") or {}),
                        "reject_unregistered_guardrails": True,
                    },
                    "guardrails": [
                        {
                            "guardrail_name": GUARDRAIL,
                            "litellm_params": {
                                "guardrail": "generic_guardrail_api",
                                "mode": "pre_call",
                                "default_on": False,
                                "api_base": vendor.url,
                                "api_key": "synthetic-guardrail-key",
                            },
                        }
                    ],
                }
            )
        )
        environment: Final = {
            "OPENAI_API_BASE": upstreams[OPENAI_ROUTE].url,
            "OPENAI_API_KEY": "synthetic-openai-key",
            "ANTHROPIC_API_BASE": upstreams[ANTHROPIC_ROUTE].url,
            "ANTHROPIC_API_KEY": "synthetic-anthropic-key",
            "GEMINI_API_BASE": upstreams[GEMINI_ROUTE].url,
            "GEMINI_API_KEY": "synthetic-gemini-key",
        }
        owned: Final = stack.enter_context(owned_proxy_process(session_gateway, directory, environment, config=path))
        yield RejectUnregisteredRig(owned, vendor, upstreams)


def _send(rig: RejectUnregisteredRig, route: str, body: Mapping[str, JsonValue], key: str) -> httpx.Response:
    rig.vendor.drain()
    for upstream in rig.upstreams.values():
        upstream.drain()
    return rig.owned.gateway.request("POST", route, body, key=key, headers={"x-goog-api-key": key})


def _unregistered_name() -> str:
    return f"not-registered-{uuid.uuid4().hex}"


@pytest.mark.parametrize("route", ROUTES)
def test_a_body_guardrails_string_naming_an_unregistered_guardrail_is_rejected(
    rig: RejectUnregisteredRig, route: str
) -> None:
    name: Final = _unregistered_name()
    with rig.owned.gateway.scenario() as scenario:
        response: Final = _send(rig, route, {**_route_body(route), "guardrails": name}, scenario.key())

    assert response.status_code == 400, response.text
    assert f"Unknown guardrail(s): {name}" in response.text, response.text
    assert rig.upstreams[route].drain() == ()


@pytest.mark.parametrize("route", ROUTES)
def test_a_guarded_keys_metadata_guardrails_string_naming_an_unregistered_guardrail_is_rejected(
    rig: RejectUnregisteredRig, route: str
) -> None:
    name: Final = _unregistered_name()
    with rig.owned.gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [GUARDRAIL]})
        response: Final = _send(rig, route, {**_route_body(route), "metadata": {"guardrails": name}}, key)

    assert response.status_code == 400, response.text
    assert f"Unknown guardrail(s): {name}" in response.text, response.text
    assert rig.vendor.drain() == ()
    assert rig.upstreams[route].drain() == ()


@pytest.mark.parametrize("route", ROUTES)
def test_a_body_guardrails_list_naming_an_unregistered_guardrail_is_rejected(
    rig: RejectUnregisteredRig, route: str
) -> None:
    name: Final = _unregistered_name()
    with rig.owned.gateway.scenario() as scenario:
        response: Final = _send(rig, route, {**_route_body(route), "guardrails": [name]}, scenario.key())

    assert response.status_code == 400, response.text
    assert f"Unknown guardrail(s): {name}" in response.text, response.text
    assert rig.upstreams[route].drain() == ()


@pytest.mark.parametrize("route", ROUTES)
def test_a_body_guardrails_string_naming_a_registered_guardrail_is_dropped(
    rig: RejectUnregisteredRig, route: str
) -> None:
    with rig.owned.gateway.scenario() as scenario:
        response: Final = _send(rig, route, {**_route_body(route), "guardrails": GUARDRAIL}, scenario.key())

    assert response.status_code == 200, response.text
    assert rig.vendor.drain() == ()
    assert [b'"guardrails"' in request.body for request in rig.upstreams[route].drain()] == [False]

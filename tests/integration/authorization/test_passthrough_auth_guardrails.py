import json
import uuid
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import httpx
import pytest
import yaml
from integration._support.anthropic_sse import message_json
from integration._support.client import Gateway, Scenario
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

GUARDRAIL: Final = "passthrough-auth-guardrail"
GLOBAL_POST_CALL: Final = "passthrough-global-post-call"
KEY_POST_CALL: Final = "passthrough-key-post-call"
GUARDED_MODEL: Final = "passthrough-guarded-model"
BLOCK_MARKER: Final = "block-this-prompt"
OPENAI_ROUTE: Final = "/openai/v1/chat/completions"
ANTHROPIC_ROUTE: Final = "/anthropic/v1/messages"
GEMINI_ROUTE: Final = "/gemini/v1beta/models/gemini-2.5-flash:generateContent"
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])
_JSON_LIST: Final = TypeAdapter(list[JsonValue])

Level = Literal["key", "team", "project"]


def _route_body(route: str, prompt: str) -> dict[str, JsonValue]:
    match route:
        case "/anthropic/v1/messages":
            return {"model": "claude-sonnet-5-5", "max_tokens": 16, "messages": [{"role": "user", "content": prompt}]}
        case "/gemini/v1beta/models/gemini-2.5-flash:generateContent":
            return {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        case _:
            return {"model": "gpt-5.6", "messages": [{"role": "user", "content": prompt}]}


def _verdict(request: Request) -> Reply:
    action: Final = "BLOCKED" if BLOCK_MARKER.encode() in request.body else "NONE"
    return Reply(body=json.dumps({"action": action, "blocked_reason": "synthetic block"}).encode())


_OPENAI_BODY: Final = json.dumps(
    _JSON.validate_json(chat_reply("chatcmpl-passthrough-auth", "gpt-5.6", "Hi", stream=False).body), indent=2
).encode()


def _openai_reply(_request: Request) -> Reply:
    return Reply(body=_OPENAI_BODY)


def _lakera_clean(_request: Request) -> Reply:
    return Reply(body=json.dumps({"flagged": False, "breakdown": [], "payload": []}).encode())


def _lakera_flagged(_request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "flagged": True,
                "breakdown": [{"detector_type": "prompt_attack", "detected": True, "message_id": 0}],
                "payload": [],
            }
        ).encode()
    )


def _anthropic_reply(_request: Request) -> Reply:
    return Reply(body=message_json("msg_passthrough_auth", "claude-sonnet-5-5", "Hi"))


def _gemini_reply(_request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1, "totalTokenCount": 5},
            }
        ).encode()
    )


@dataclass(frozen=True, slots=True)
class AuthGuardrailRig:
    owned: OwnedProxy
    vendor: Wire
    post_call_vendor: Wire
    key_post_call_vendor: Wire
    upstreams: Mapping[str, Wire]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[AuthGuardrailRig]:
    directory: Final = tmp_path_factory.mktemp("passthrough-auth-guardrails")
    with ExitStack() as stack:
        vendor: Final = stack.enter_context(wire_server(_verdict))
        post_call_vendor: Final = stack.enter_context(wire_server(_lakera_clean))
        key_post_call_vendor: Final = stack.enter_context(wire_server(_lakera_flagged))
        upstreams: Final = MappingProxyType(
            {
                OPENAI_ROUTE: stack.enter_context(wire_server(_openai_reply)),
                ANTHROPIC_ROUTE: stack.enter_context(wire_server(_anthropic_reply)),
                GEMINI_ROUTE: stack.enter_context(wire_server(_gemini_reply)),
            }
        )
        config: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        path: Final = directory / "passthrough-auth-guardrails.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "model_list": [
                        *_JSON_LIST.validate_python(config.get("model_list") or []),
                        {
                            "model_name": GUARDED_MODEL,
                            "litellm_params": {
                                "model": "openai/gpt-5.6",
                                "api_key": "synthetic-openai-key",
                                "guardrails": [GUARDRAIL],
                            },
                        },
                    ],
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
                        },
                        {
                            "guardrail_name": KEY_POST_CALL,
                            "litellm_params": {
                                "guardrail": "lakera_v2",
                                "mode": "post_call",
                                "default_on": False,
                                "api_base": key_post_call_vendor.url,
                                "api_key": "synthetic-guardrail-key",
                            },
                        },
                        {
                            "guardrail_name": GLOBAL_POST_CALL,
                            "litellm_params": {
                                "guardrail": "lakera_v2",
                                "mode": "post_call",
                                "default_on": True,
                                "api_base": post_call_vendor.url,
                                "api_key": "synthetic-guardrail-key",
                            },
                        },
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
        yield AuthGuardrailRig(owned, vendor, post_call_vendor, key_post_call_vendor, upstreams)


def _key_with_guardrail(scenario: Scenario, level: Level) -> str:
    attached: Final[dict[str, JsonValue]] = {"metadata": {"guardrails": [GUARDRAIL]}}
    team: Final = scenario.team(**(attached if level == "team" else {}))
    match level:
        case "project":
            return scenario.key(team_id=team, project_id=scenario.project(team, **attached))
        case "key":
            return scenario.key(team_id=team, **attached)
        case _:
            return scenario.key(team_id=team)


def _send(gateway: Gateway, route: str, body: Mapping[str, JsonValue], key: str) -> httpx.Response:
    return gateway.request("POST", route, body, key=key, headers={"x-goog-api-key": key})


def _drain(rig: AuthGuardrailRig) -> None:
    rig.vendor.drain()
    rig.post_call_vendor.drain()
    rig.key_post_call_vendor.drain()
    for upstream in rig.upstreams.values():
        upstream.drain()


@pytest.mark.parametrize("level", ["key", "team", "project"])
@pytest.mark.parametrize("route", [OPENAI_ROUTE, ANTHROPIC_ROUTE, GEMINI_ROUTE])
def test_auth_guardrail_blocks_the_prompt_before_the_provider(rig: AuthGuardrailRig, route: str, level: Level) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = _key_with_guardrail(scenario, level)
        _drain(rig)

        response: Final = _send(gateway, route, _route_body(route, BLOCK_MARKER), key)

        assert response.status_code == 400, response.text
        assert [BLOCK_MARKER.encode() in request.body for request in rig.vendor.drain()] == [True]
        assert rig.upstreams[route].drain() == ()


@pytest.mark.parametrize("route", [OPENAI_ROUTE, ANTHROPIC_ROUTE, GEMINI_ROUTE])
def test_auth_guardrail_lets_a_clean_prompt_reach_the_provider(rig: AuthGuardrailRig, route: str) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = _key_with_guardrail(scenario, "key")
        marker: Final = f"clean-{uuid.uuid4().hex}"
        _drain(rig)

        response: Final = _send(gateway, route, _route_body(route, marker), key)

        assert response.status_code == 200, response.text
        assert [marker.encode() in request.body for request in rig.vendor.drain()] == [True]
        assert [marker.encode() in request.body for request in rig.upstreams[route].drain()] == [True]


def test_empty_request_guardrails_list_does_not_drop_the_key_guardrail(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = _key_with_guardrail(scenario, "key")
        _drain(rig)

        response: Final = gateway.request(
            "POST", OPENAI_ROUTE, {**_route_body(OPENAI_ROUTE, BLOCK_MARKER), "guardrails": []}, key=key
        )

        assert response.status_code == 400, response.text
        assert rig.upstreams[OPENAI_ROUTE].drain() == ()


def test_request_without_a_body_is_not_sent_to_the_guardrail(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = _key_with_guardrail(scenario, "key")
        _drain(rig)

        response: Final = gateway.request("GET", "/openai/v1/models", key=key)

        assert response.status_code == 200, response.text
        assert rig.vendor.drain() == ()
        assert len(rig.upstreams[OPENAI_ROUTE].drain()) == 1


def test_pre_call_guardrail_relays_the_provider_reply_untouched(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = _key_with_guardrail(scenario, "key")
        _drain(rig)

        response: Final = _send(gateway, OPENAI_ROUTE, _route_body(OPENAI_ROUTE, "What is the capital of France?"), key)

        assert response.status_code == 200, response.text
        assert response.content == _OPENAI_BODY
        assert len(rig.vendor.drain()) == 1
        assert rig.post_call_vendor.drain() == ()


def test_key_post_call_guardrail_runs_on_a_model_with_its_own_guardrails(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [KEY_POST_CALL]})
        _drain(rig)

        response: Final = _send(
            gateway,
            OPENAI_ROUTE,
            {**_route_body(OPENAI_ROUTE, "What is the capital of France?"), "model": GUARDED_MODEL},
            key,
        )

        assert response.status_code == 400, response.text
        assert len(rig.key_post_call_vendor.drain()) == 1
        assert len(rig.upstreams[OPENAI_ROUTE].drain()) == 1

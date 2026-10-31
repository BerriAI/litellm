import json
from collections.abc import Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Final

import httpx
import pytest
import yaml
from integration._support.anthropic_sse import message_json
from integration._support.client import Gateway, Scenario
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

PRE_CALL: Final = "passthrough-header-pre"
POST_CALL: Final = "passthrough-header-post"
NON_LATIN1: Final = "passthrough-header-pre\u2014em"
BLOCKER: Final = "passthrough-header-block"
MISSING_MODEL: Final = "passthrough-header-missing-model"
PILLAR: Final = "passthrough-header-pillar"
TAG_POLICY: Final = "passthrough-header-tag-policy"
POLICY_TAG_PREFIX: Final = "passthrough-header-tag-"
HEADER: Final = "x-litellm-applied-guardrails"
OPENAI_ROUTE: Final = "/openai/v1/chat/completions"
ANTHROPIC_ROUTE: Final = "/anthropic/v1/messages"
GEMINI_ROUTE: Final = "/gemini/v1beta/models/gemini-2.5-flash:generateContent"
_CONFIG: Final = Path(__file__).resolve().parents[1] / "proxy_config.yaml"
_JSON: Final = TypeAdapter(dict[str, JsonValue])


def _route_body(route: str) -> dict[str, JsonValue]:
    match route:
        case "/anthropic/v1/messages":
            return {"model": "claude-sonnet-5-5", "max_tokens": 16, "messages": [{"role": "user", "content": "Hi"}]}
        case "/gemini/v1beta/models/gemini-2.5-flash:generateContent":
            return {"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]}
        case _:
            return {"model": "gpt-5.6", "messages": [{"role": "user", "content": "Hi"}]}


def _allow(_request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "NONE"}).encode())


def _block(_request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic block"}).encode())


def _pillar_allow(_request: Request) -> Reply:
    return Reply(body=json.dumps({"flagged": False, "session_id": "pillar-vendor-session"}).encode())


def _openai_reply(request: Request) -> Reply:
    payload: Final = _JSON.validate_json(request.body)
    if payload.get("model") == MISSING_MODEL:
        return Reply(
            status=404,
            body=json.dumps({"error": {"message": "model not found", "type": "invalid_request_error"}}).encode(),
        )
    return chat_reply("chatcmpl-passthrough-header", "gpt-5.6", "Hello", stream=payload.get("stream") is True)


def _anthropic_reply(_request: Request) -> Reply:
    return Reply(body=message_json("msg_passthrough_header", "claude-sonnet-5-5", "Hello"))


def _gemini_reply(_request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "candidates": [{"content": {"role": "model", "parts": [{"text": "Hello"}]}, "finishReason": "STOP"}],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
            }
        ).encode()
    )


@dataclass(frozen=True, slots=True)
class HeaderRig:
    owned: OwnedProxy
    upstreams: Mapping[str, Wire]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[HeaderRig]:
    directory: Final = tmp_path_factory.mktemp("passthrough-applied-guardrails-header")
    with ExitStack() as stack:
        vendor: Final = stack.enter_context(wire_server(_allow))
        blocking_vendor: Final = stack.enter_context(wire_server(_block))
        pillar_vendor: Final = stack.enter_context(wire_server(_pillar_allow))
        upstreams: Final = MappingProxyType(
            {
                OPENAI_ROUTE: stack.enter_context(wire_server(_openai_reply)),
                ANTHROPIC_ROUTE: stack.enter_context(wire_server(_anthropic_reply)),
                GEMINI_ROUTE: stack.enter_context(wire_server(_gemini_reply)),
            }
        )
        config: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        path: Final = directory / "passthrough-applied-guardrails-header.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "guardrails": [
                        {
                            "guardrail_name": name,
                            "litellm_params": {
                                "guardrail": "generic_guardrail_api",
                                "mode": mode,
                                "default_on": False,
                                "api_base": api_base,
                                "api_key": "synthetic-guardrail-key",
                            },
                        }
                        for name, mode, api_base in (
                            (PRE_CALL, "pre_call", vendor.url),
                            (POST_CALL, "post_call", vendor.url),
                            (NON_LATIN1, "pre_call", vendor.url),
                            (BLOCKER, "pre_call", blocking_vendor.url),
                        )
                    ]
                    + [
                        {
                            "guardrail_name": PILLAR,
                            "litellm_params": {
                                "guardrail": "pillar",
                                "mode": "pre_call",
                                "default_on": False,
                                "api_base": pillar_vendor.url,
                                "api_key": "synthetic-pillar-key",
                            },
                        }
                    ],
                    "policies": {TAG_POLICY: {"guardrails": {"add": [PRE_CALL]}}},
                    "policy_attachments": [{"policy": TAG_POLICY, "tags": [f"{POLICY_TAG_PREFIX}*"]}],
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
        yield HeaderRig(owned, upstreams)


def _send(
    gateway: Gateway, scenario: Scenario, route: str, body: Mapping[str, JsonValue], *guardrails: str
) -> httpx.Response:
    key: Final = scenario.key(metadata={"guardrails": list(guardrails)}) if guardrails else scenario.key()
    return gateway.request("POST", route, body, key=key, headers={"x-goog-api-key": key})


@pytest.mark.parametrize("route", [OPENAI_ROUTE, ANTHROPIC_ROUTE, GEMINI_ROUTE])
def test_response_names_the_guardrail_that_ran(rig: HeaderRig, route: str) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        response: Final = _send(gateway, scenario, route, _route_body(route), PRE_CALL)

    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER) == PRE_CALL


def test_streamed_response_names_the_guardrail_that_ran(rig: HeaderRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        response: Final = _send(
            gateway, scenario, OPENAI_ROUTE, {**_route_body(OPENAI_ROUTE), "stream": True}, PRE_CALL
        )

    assert response.status_code == 200, response.text
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    assert response.headers.get(HEADER) == PRE_CALL


def test_response_names_pre_and_post_call_guardrails(rig: HeaderRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        response: Final = _send(gateway, scenario, OPENAI_ROUTE, _route_body(OPENAI_ROUTE), PRE_CALL, POST_CALL)

    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER) == f"{PRE_CALL},{POST_CALL}"


@pytest.mark.parametrize("field", ["metadata", "litellm_params"])
def test_client_cannot_claim_a_guardrail_ran(rig: HeaderRig, field: str) -> None:
    gateway: Final = rig.owned.gateway
    claim: Final[dict[str, JsonValue]] = {
        "applied_guardrails": ["forged-guardrail"],
        "guardrail_scan_ids": ["forged-scan"],
    }
    forged: Final = {**_route_body(OPENAI_ROUTE), field: claim if field == "metadata" else {"metadata": claim}}
    with gateway.scenario() as scenario:
        response: Final = _send(gateway, scenario, OPENAI_ROUTE, forged)

    assert response.status_code == 200, response.text
    assert HEADER not in response.headers
    assert "x-litellm-guardrail-scan-id" not in response.headers


def test_guardrail_name_a_header_cannot_carry_is_left_out(rig: HeaderRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        response: Final = _send(gateway, scenario, OPENAI_ROUTE, _route_body(OPENAI_ROUTE), NON_LATIN1)

    assert response.status_code == 200, response.text
    assert HEADER not in response.headers


@pytest.mark.parametrize("stream", [False, True])
def test_upstream_error_names_the_guardrail_that_ran(rig: HeaderRig, stream: bool) -> None:
    gateway: Final = rig.owned.gateway
    body: Final = {**_route_body(OPENAI_ROUTE), "model": MISSING_MODEL, "stream": stream}
    with gateway.scenario() as scenario:
        response: Final = _send(gateway, scenario, OPENAI_ROUTE, body, PRE_CALL)

    assert response.status_code == 404, response.text
    assert response.headers.get(HEADER) == PRE_CALL


def test_blocked_request_names_the_blocking_guardrail(rig: HeaderRig) -> None:
    gateway: Final = rig.owned.gateway
    upstream: Final = rig.upstreams[OPENAI_ROUTE]
    upstream.drain()
    with gateway.scenario() as scenario:
        response: Final = _send(gateway, scenario, OPENAI_ROUTE, _route_body(OPENAI_ROUTE), PRE_CALL, BLOCKER)

    assert response.status_code == 400, response.text
    assert response.headers.get(HEADER) == f"{PRE_CALL},{BLOCKER}"
    assert upstream.drain() == ()


@pytest.mark.parametrize("field", ["metadata", "litellm_metadata"])
def test_client_claims_beside_a_real_guardrail_are_dropped(rig: HeaderRig, field: str) -> None:
    gateway: Final = rig.owned.gateway
    claim: Final[dict[str, JsonValue]] = {
        "applied_guardrails": ["forged-guardrail"],
        "guardrail_scan_ids": ["forged-scan"],
    }
    with gateway.scenario() as scenario:
        response: Final = _send(gateway, scenario, OPENAI_ROUTE, {**_route_body(OPENAI_ROUTE), field: claim}, PRE_CALL)

    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER) == PRE_CALL
    assert "forged-scan" not in response.headers.get("x-litellm-guardrail-scan-id", "")


@pytest.mark.parametrize("claim", ["none", 7, ""])
def test_non_list_client_claim_does_not_fail_a_guarded_request(rig: HeaderRig, claim: JsonValue) -> None:
    gateway: Final = rig.owned.gateway
    body: Final = {**_route_body(OPENAI_ROUTE), "metadata": {"applied_guardrails": claim}}
    with gateway.scenario() as scenario:
        response: Final = _send(gateway, scenario, OPENAI_ROUTE, body, PRE_CALL)

    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER) == PRE_CALL


def test_pillar_headers_stay_off_pass_through_responses(rig: HeaderRig) -> None:
    gateway: Final = rig.owned.gateway
    forged: Final[dict[str, JsonValue]] = {"pillar_session_id_response": "client-session", "pillar_scanners": {"x": 1}}
    with gateway.scenario() as scenario:
        response: Final = _send(
            gateway, scenario, OPENAI_ROUTE, {**_route_body(OPENAI_ROUTE), "metadata": forged}, PILLAR
        )

    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER) == PILLAR
    assert [name for name, _ in response.headers.multi_items() if name.startswith("x-pillar-")] == []


@pytest.mark.parametrize("suffix", ["team\n", "x\r\nX-Injected: yes"])
def test_header_value_with_a_line_break_is_left_out(rig: HeaderRig, suffix: str) -> None:
    gateway: Final = rig.owned.gateway
    body: Final[dict[str, JsonValue]] = {
        **_route_body(OPENAI_ROUTE),
        "metadata": {"tags": [f"{POLICY_TAG_PREFIX}{suffix}"]},
    }
    with gateway.scenario() as scenario:
        response: Final = _send(gateway, scenario, OPENAI_ROUTE, body)

    assert response.status_code == 200, response.text
    assert response.headers.get(HEADER) == PRE_CALL
    assert response.headers.get("x-litellm-applied-policies") == TAG_POLICY
    assert "x-litellm-policy-sources" not in response.headers
    assert "x-injected" not in response.headers

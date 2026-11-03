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
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually
from integration._support.openai_wire import chat_reply
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.safeguards_rig import key_hash
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

GUARDRAIL: Final = "passthrough-auth-guardrail"
GLOBAL_POST_CALL: Final = "passthrough-global-post-call"
KEY_POST_CALL: Final = "passthrough-key-post-call"
GENERIC_POST_CALL: Final = "passthrough-generic-post-call"
BLOCKING_POST_CALL: Final = "passthrough-post-call-blocks-every-reply"
TAG_SCOPED_POST_CALL: Final = "passthrough-tag-scoped-post-call"
POST_CALL_TAG: Final = "scan-the-reply"
MODEL_POST_CALL: Final = "passthrough-model-post-call"
GUARDED_MODEL: Final = "passthrough-guarded-model"
POST_CALL_GUARDED_MODEL: Final = "passthrough-post-call-guarded-model"
BLOCK_MARKER: Final = "block-this-prompt"
RAISING_HOOK_MODEL: Final = "passthrough-model-a-logger-hook-cannot-read"
_RAISING_HOOK_MODULE: Final = f"""
from litellm.integrations.custom_logger import CustomLogger


class RaisingHook(CustomLogger):
    async def async_post_call_success_hook(self, data, user_api_key_dict, response):
        if data.get("model") == "{RAISING_HOOK_MODEL}":
            raise ValueError("this hook only reads native responses")


proxy_handler_instance = RaisingHook()
"""
OPENAI_ROUTE: Final = "/openai/v1/chat/completions"
ANTHROPIC_ROUTE: Final = "/anthropic/v1/messages"
GEMINI_ROUTE: Final = "/gemini/v1beta/models/gemini-2.5-flash:generateContent"
CUSTOM_ROUTE: Final = "/passthrough-auth-custom"
CUSTOM_GUARDED_ROUTE: Final = "/passthrough-auth-custom-guarded"
CUSTOM_PRE_CALL_ROUTE: Final = "/passthrough-auth-custom-pre-call"
GLOBAL_UNIFIED_POST_CALL: Final = "passthrough-global-unified-post-call"
UNPRICED_MODEL: Final = "passthrough-auth-unpriced-model"
MONITOR_POST_CALL: Final = "passthrough-auth-monitor-post-call"
MONITORED_PROMPT: Final = "reply-with-the-monitored-id"
MONITORED_REPLY_ID: Final = "msg_passthrough_auth_monitored"
PHONE_POST_CALL: Final = "passthrough-auth-phone-post-call"
PIPELINE_BLOCKS: Final = "passthrough-pipeline-blocks-the-response"
PIPELINE_ALLOWS: Final = "passthrough-pipeline-allows-the-response"
PIPELINE_BEFORE_THE_CALL: Final = "passthrough-pipeline-before-the-call"
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


def _pipeline_policy(mode: str, guardrail: str, on_fail: str) -> dict[str, JsonValue]:
    return {
        "guardrails": {"add": [guardrail]},
        "pipeline": {"mode": mode, "steps": [{"guardrail": guardrail, "on_pass": "allow", "on_fail": on_fail}]},
    }


def _verdict(request: Request) -> Reply:
    action: Final = "BLOCKED" if BLOCK_MARKER.encode() in request.body else "NONE"
    return Reply(body=json.dumps({"action": action, "blocked_reason": "synthetic block"}).encode())


def _block_every_reply(_request: Request) -> Reply:
    return Reply(body=json.dumps({"action": "BLOCKED", "blocked_reason": "synthetic block"}).encode())


_OPENAI_BODY: Final = json.dumps(
    {
        **_JSON.validate_json(chat_reply("chatcmpl-passthrough-auth", "gpt-5.6", "Hi", stream=False).body),
        "created": 1793600123,
    },
    indent=2,
).encode()


def _openai_reply(_request: Request) -> Reply:
    return Reply(body=_OPENAI_BODY)


def _lakera_pii_only(_request: Request) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "flagged": True,
                "breakdown": [{"detector_type": "pii/email", "detected": True, "message_id": 0}],
                "payload": [],
            }
        ).encode()
    )


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


_ANTHROPIC_BODY: Final = json.dumps(
    _JSON.validate_json(message_json("msg_passthrough_auth", "claude-sonnet-5-5", "Hi")), indent=2
).encode()
_GEMINI_BODY: Final = json.dumps(
    {
        "candidates": [{"content": {"role": "model", "parts": [{"text": "Hi"}]}, "finishReason": "STOP"}],
        "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 1, "totalTokenCount": 5},
    },
    indent=2,
).encode()
_UPSTREAM_BODIES: Final = MappingProxyType(
    {OPENAI_ROUTE: _OPENAI_BODY, ANTHROPIC_ROUTE: _ANTHROPIC_BODY, GEMINI_ROUTE: _GEMINI_BODY}
)


_MONITORED_ANTHROPIC_BODY: Final = json.dumps(
    _JSON.validate_json(message_json(MONITORED_REPLY_ID, "claude-sonnet-5-5", "Hi"))
).encode()


_UNPRICED_ANTHROPIC_BODY: Final = json.dumps(
    {
        **JSON_OBJECT.validate_json(message_json("msg_passthrough_auth_unpriced", UNPRICED_MODEL, "Hi")),
        "usage": {"input_tokens": 300, "output_tokens": 200},
    }
).encode()


def _anthropic_reply(request: Request) -> Reply:
    if UNPRICED_MODEL.encode() in request.body:
        return Reply(body=_UNPRICED_ANTHROPIC_BODY)
    return Reply(body=_MONITORED_ANTHROPIC_BODY if MONITORED_PROMPT.encode() in request.body else _ANTHROPIC_BODY)


def _gemini_reply(_request: Request) -> Reply:
    return Reply(body=_GEMINI_BODY)


@dataclass(frozen=True, slots=True)
class AuthGuardrailRig:
    owned: OwnedProxy
    vendor: Wire
    post_call_vendor: Wire
    key_post_call_vendor: Wire
    generic_post_call_vendor: Wire
    global_unified_vendor: Wire
    model_post_call_vendor: Wire
    upstreams: Mapping[str, Wire]


@pytest.fixture(scope="module")
def rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[AuthGuardrailRig]:
    directory: Final = tmp_path_factory.mktemp("passthrough-auth-guardrails")
    with ExitStack() as stack:
        vendor: Final = stack.enter_context(wire_server(_verdict))
        post_call_vendor: Final = stack.enter_context(wire_server(_lakera_clean))
        key_post_call_vendor: Final = stack.enter_context(wire_server(_lakera_flagged))
        generic_post_call_vendor: Final = stack.enter_context(wire_server(_verdict))
        global_unified_vendor: Final = stack.enter_context(wire_server(_verdict))
        blocking_post_call_vendor: Final = stack.enter_context(wire_server(_block_every_reply))
        monitor_post_call_vendor: Final = stack.enter_context(wire_server(_lakera_pii_only))
        model_post_call_vendor: Final = stack.enter_context(wire_server(_verdict))
        upstreams: Final = MappingProxyType(
            {
                OPENAI_ROUTE: stack.enter_context(wire_server(_openai_reply)),
                ANTHROPIC_ROUTE: stack.enter_context(wire_server(_anthropic_reply)),
                GEMINI_ROUTE: stack.enter_context(wire_server(_gemini_reply)),
            }
        )
        config: Final = _JSON.validate_python(yaml.safe_load(_CONFIG.read_text()))
        general_settings: Final = _JSON.validate_python(config.get("general_settings") or {})
        (directory / "passthrough_raising_hook.py").write_text(_RAISING_HOOK_MODULE)
        path: Final = directory / "passthrough-auth-guardrails.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    **config,
                    "litellm_settings": {
                        **_JSON.validate_python(config.get("litellm_settings") or {}),
                        "callbacks": ["passthrough_raising_hook.proxy_handler_instance"],
                    },
                    "general_settings": {
                        **general_settings,
                        "pass_through_endpoints": [
                            {
                                "path": CUSTOM_ROUTE,
                                "target": f"{upstreams[OPENAI_ROUTE].url}/v1/chat/completions",
                                "auth": True,
                            },
                            {
                                "path": CUSTOM_GUARDED_ROUTE,
                                "target": f"{upstreams[OPENAI_ROUTE].url}/v1/chat/completions",
                                "auth": True,
                                "cost_per_request": 0.25,
                                "guardrails": {BLOCKING_POST_CALL: None},
                            },
                            {
                                "path": CUSTOM_PRE_CALL_ROUTE,
                                "target": f"{upstreams[OPENAI_ROUTE].url}/v1/chat/completions",
                                "auth": True,
                                "guardrails": {GUARDRAIL: None},
                            },
                        ],
                    },
                    "policies": {
                        PIPELINE_BLOCKS: _pipeline_policy("post_call", KEY_POST_CALL, "block"),
                        PIPELINE_ALLOWS: _pipeline_policy("post_call", KEY_POST_CALL, "allow"),
                        PIPELINE_BEFORE_THE_CALL: _pipeline_policy("pre_call", GUARDRAIL, "block"),
                    },
                    "policy_attachments": [
                        {"policy": policy, "tags": [policy]}
                        for policy in (PIPELINE_BLOCKS, PIPELINE_ALLOWS, PIPELINE_BEFORE_THE_CALL)
                    ],
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
                        {
                            "model_name": POST_CALL_GUARDED_MODEL,
                            "litellm_params": {
                                "model": "openai/gpt-5.6",
                                "api_key": "synthetic-openai-key",
                                "guardrails": [MODEL_POST_CALL],
                            },
                        },
                    ],
                    "guardrails": [
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
                            "guardrail_name": BLOCKING_POST_CALL,
                            "litellm_params": {
                                "guardrail": "generic_guardrail_api",
                                "mode": "post_call",
                                "default_on": False,
                                "api_base": blocking_post_call_vendor.url,
                                "api_key": "synthetic-guardrail-key",
                            },
                        },
                        {
                            "guardrail_name": TAG_SCOPED_POST_CALL,
                            "litellm_params": {
                                "guardrail": "generic_guardrail_api",
                                "mode": {"tags": {POST_CALL_TAG: "post_call"}, "default": "logging_only"},
                                "default_on": False,
                                "api_base": blocking_post_call_vendor.url,
                                "api_key": "synthetic-guardrail-key",
                            },
                        },
                        {
                            "guardrail_name": MONITOR_POST_CALL,
                            "litellm_params": {
                                "guardrail": "lakera_v2",
                                "mode": "post_call",
                                "default_on": False,
                                "on_flagged": "monitor",
                                "api_base": monitor_post_call_vendor.url,
                                "api_key": "synthetic-guardrail-key",
                            },
                        },
                        {
                            "guardrail_name": GLOBAL_UNIFIED_POST_CALL,
                            "litellm_params": {
                                "guardrail": "generic_guardrail_api",
                                "mode": "post_call",
                                "default_on": True,
                                "api_base": global_unified_vendor.url,
                                "api_key": "synthetic-guardrail-key",
                            },
                        },
                        {
                            "guardrail_name": GENERIC_POST_CALL,
                            "litellm_params": {
                                "guardrail": "generic_guardrail_api",
                                "mode": "post_call",
                                "default_on": False,
                                "api_base": generic_post_call_vendor.url,
                                "api_key": "synthetic-guardrail-key",
                            },
                        },
                        {
                            "guardrail_name": PHONE_POST_CALL,
                            "litellm_params": {
                                "guardrail": "litellm_content_filter",
                                "mode": "post_call",
                                "default_on": False,
                                "patterns": [
                                    {"pattern_type": "prebuilt", "pattern_name": "us_phone", "action": "BLOCK"}
                                ],
                            },
                        },
                        {
                            "guardrail_name": MODEL_POST_CALL,
                            "litellm_params": {
                                "guardrail": "generic_guardrail_api",
                                "mode": "post_call",
                                "default_on": False,
                                "api_base": model_post_call_vendor.url,
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
        yield AuthGuardrailRig(
            owned,
            vendor,
            post_call_vendor,
            key_post_call_vendor,
            generic_post_call_vendor,
            global_unified_vendor,
            model_post_call_vendor,
            upstreams,
        )


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
    rig.generic_post_call_vendor.drain()
    rig.model_post_call_vendor.drain()
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


@pytest.mark.parametrize("route", [ANTHROPIC_ROUTE, GEMINI_ROUTE])
def test_pre_call_guardrail_relays_every_provider_reply_untouched(rig: AuthGuardrailRig, route: str) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = _key_with_guardrail(scenario, "team")
        _drain(rig)

        response: Final = _send(gateway, route, _route_body(route, "What is the capital of France?"), key)

        assert response.status_code == 200, response.text
        assert response.content == _UPSTREAM_BODIES[route]
        assert len(rig.vendor.drain()) == 1
        assert rig.post_call_vendor.drain() == ()


def test_custom_pass_through_endpoint_keeps_its_opt_in(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(
            team_id=scenario.team(),
            metadata={"guardrails": [GUARDRAIL], "allowed_passthrough_routes": [CUSTOM_ROUTE]},
        )
        _drain(rig)

        response: Final = gateway.request("POST", CUSTOM_ROUTE, _route_body(OPENAI_ROUTE, BLOCK_MARKER), key=key)

        assert response.status_code == 200, response.text
        assert rig.vendor.drain() == ()
        assert [BLOCK_MARKER.encode() in request.body for request in rig.upstreams[OPENAI_ROUTE].drain()] == [True]


@pytest.mark.parametrize("route", [OPENAI_ROUTE, ANTHROPIC_ROUTE, GEMINI_ROUTE])
def test_key_apply_guardrail_post_call_scans_the_provider_reply(rig: AuthGuardrailRig, route: str) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [GENERIC_POST_CALL]})
        _drain(rig)

        response: Final = _send(gateway, route, _route_body(route, "What is the capital of France?"), key)

        assert response.status_code == 200, response.text
        assert [b'"Hi' in request.body for request in rig.generic_post_call_vendor.drain()] == [True]


def test_post_call_guardrail_does_not_read_a_reply_timestamp_as_a_phone_number(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [PHONE_POST_CALL]})
        _drain(rig)

        response: Final = _send(gateway, OPENAI_ROUTE, _route_body(OPENAI_ROUTE, "What is the capital of France?"), key)

        assert response.status_code == 200, response.text
        assert response.json()["created"] == 1793600123


@pytest.mark.parametrize("route", [OPENAI_ROUTE, ANTHROPIC_ROUTE, GEMINI_ROUTE])
def test_guardrail_the_client_names_in_metadata_still_runs_for_a_guarded_key(rig: AuthGuardrailRig, route: str) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [GENERIC_POST_CALL]})
        _drain(rig)

        response: Final = _send(
            gateway, route, {**_route_body(route, BLOCK_MARKER), "metadata": {"guardrails": [GUARDRAIL]}}, key
        )

        assert response.status_code == 400, response.text
        assert [BLOCK_MARKER.encode() in request.body for request in rig.vendor.drain()] == [True]
        assert rig.upstreams[route].drain() == ()


@pytest.mark.parametrize("settings", [{}, True])
@pytest.mark.parametrize("route", [OPENAI_ROUTE, ANTHROPIC_ROUTE, GEMINI_ROUTE])
def test_guardrail_the_client_names_in_the_dict_form_still_runs_for_a_guarded_key(
    rig: AuthGuardrailRig, route: str, settings: JsonValue
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [GENERIC_POST_CALL]})
        _drain(rig)

        response: Final = _send(
            gateway, route, {**_route_body(route, BLOCK_MARKER), "metadata": {"guardrails": {GUARDRAIL: settings}}}, key
        )

        assert response.status_code == 400, response.text
        assert [BLOCK_MARKER.encode() in request.body for request in rig.vendor.drain()] == [True]
        assert rig.upstreams[route].drain() == ()


def _key_with_opt_out(scenario: Scenario, opt_out: str) -> str:
    guarded: Final[dict[str, JsonValue]] = {"guardrails": [GENERIC_POST_CALL]}
    match opt_out:
        case "key_disable":
            return scenario.key(metadata=guarded, disable_global_guardrails=True)
        case "key_opted_out":
            return scenario.key(metadata={**guarded, "opted_out_global_guardrails": [GLOBAL_POST_CALL]})
        case "team_disable":
            return scenario.key(team_id=scenario.team(disable_global_guardrails=True), metadata=guarded)
        case "team_opted_out":
            return scenario.key(
                team_id=scenario.team(metadata={"opted_out_global_guardrails": [GLOBAL_POST_CALL]}), metadata=guarded
            )
        case _:
            return scenario.key(metadata=guarded)


@pytest.mark.parametrize(
    ("opt_out", "global_vendor_calls"),
    [("key_disable", 0), ("key_opted_out", 0), ("team_disable", 0), ("team_opted_out", 0), ("none", 1)],
)
@pytest.mark.parametrize("route", [OPENAI_ROUTE, "/v1/chat/completions"])
def test_post_call_pass_honors_a_global_guardrail_opt_out_as_the_native_route_does(
    rig: AuthGuardrailRig, route: str, opt_out: str, global_vendor_calls: int
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = _key_with_opt_out(scenario, opt_out)
        _drain(rig)

        response: Final = _send(
            gateway,
            route,
            {**_route_body(OPENAI_ROUTE, "What is the capital of France?"), "model": GUARDED_MODEL},
            key,
        )

        assert response.status_code == 200, response.text
        assert len(rig.generic_post_call_vendor.drain()) == 1
        assert len(rig.post_call_vendor.drain()) == global_vendor_calls


def test_model_guardrail_does_not_run_on_pass_through_traffic_that_names_the_model(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [GENERIC_POST_CALL]})
        _drain(rig)

        response: Final = _send(
            gateway,
            OPENAI_ROUTE,
            {**_route_body(OPENAI_ROUTE, "What is the capital of France?"), "model": POST_CALL_GUARDED_MODEL},
            key,
        )

        assert response.status_code == 200, response.text
        assert len(rig.generic_post_call_vendor.drain()) == 1
        assert rig.model_post_call_vendor.drain() == ()


def test_post_call_pass_leaves_a_logger_hook_for_native_responses_out(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [GENERIC_POST_CALL]})
        _drain(rig)

        response: Final = _send(
            gateway,
            OPENAI_ROUTE,
            {**_route_body(OPENAI_ROUTE, "What is the capital of France?"), "model": RAISING_HOOK_MODEL},
            key,
        )

        assert response.status_code == 200, response.text
        assert len(rig.generic_post_call_vendor.drain()) == 1
        assert len(rig.upstreams[OPENAI_ROUTE].drain()) == 1


def _charged_rows(gateway: Gateway, key: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        eventually(
            lambda: gateway.rows(
                'SELECT status, spend, total_tokens FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (key_hash(key),)
            ),
            bool,
        )
    )


def _tagged_body(route: str, tag: str) -> dict[str, JsonValue]:
    return {**_route_body(route, "What is the capital of France?"), "metadata": {"tags": [tag]}}


@pytest.mark.parametrize("route", [OPENAI_ROUTE, ANTHROPIC_ROUTE])
def test_post_call_policy_pipeline_blocks_a_flagged_pass_through_response(rig: AuthGuardrailRig, route: str) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        _drain(rig)

        response: Final = _send(gateway, route, _tagged_body(route, PIPELINE_BLOCKS), key)

        assert response.status_code == 400, response.text
        assert len(rig.key_post_call_vendor.drain()) == 1
        assert len(rig.upstreams[route].drain()) == 1
        assert [row["status"] for row in _charged_rows(gateway, key)] == ["failure"]


def test_post_call_policy_pipeline_decides_what_a_flagged_step_does(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        _drain(rig)

        response: Final = _send(gateway, OPENAI_ROUTE, _tagged_body(OPENAI_ROUTE, PIPELINE_ALLOWS), key)

        assert response.status_code == 200, response.text
        assert len(rig.key_post_call_vendor.drain()) == 1


def test_pre_call_policy_pipeline_relays_the_provider_bytes(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        _drain(rig)

        response: Final = _send(gateway, OPENAI_ROUTE, _tagged_body(OPENAI_ROUTE, PIPELINE_BEFORE_THE_CALL), key)

        assert response.status_code == 200, response.text
        assert response.content == _OPENAI_BODY
        assert len(rig.vendor.drain()) == 1


def test_lakera_post_call_has_nothing_to_read_on_a_gemini_body(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [KEY_POST_CALL]})
        _drain(rig)

        response: Final = _send(gateway, GEMINI_ROUTE, _route_body(GEMINI_ROUTE, "What is the capital of France?"), key)

        assert response.status_code == 200, response.text
        assert rig.key_post_call_vendor.drain() == ()
        assert len(rig.upstreams[GEMINI_ROUTE].drain()) == 1


def test_post_call_policy_pipeline_lakera_step_has_nothing_to_read_on_a_gemini_body(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        _drain(rig)

        response: Final = _send(gateway, GEMINI_ROUTE, _tagged_body(GEMINI_ROUTE, PIPELINE_BLOCKS), key)

        assert response.status_code == 200, response.text
        assert rig.key_post_call_vendor.drain() == ()
        assert len(rig.upstreams[GEMINI_ROUTE].drain()) == 1


@pytest.mark.parametrize(("route", "guardrail"), [(ANTHROPIC_ROUTE, KEY_POST_CALL), (GEMINI_ROUTE, BLOCKING_POST_CALL)])
def test_response_a_post_call_guardrail_blocks_still_charges_the_provider_call(
    rig: AuthGuardrailRig, route: str, guardrail: str
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [guardrail]})
        _drain(rig)

        response: Final = _send(gateway, route, _route_body(route, "What is the capital of France?"), key)

        assert response.status_code == 400, response.text
        assert len(rig.upstreams[route].drain()) == 1
        rows: Final = _charged_rows(gateway, key)
        assert [row["status"] for row in rows] == ["failure"], rows
        assert float(str(rows[0]["spend"])) > 0, rows
        assert int(str(rows[0]["total_tokens"])) > 0, rows


def _row_fields(gateway: Gateway, key: str) -> tuple[Mapping[str, object], ...]:
    return tuple(
        eventually(
            lambda: gateway.rows(
                'SELECT status, spend, total_tokens, api_base, request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s',
                (key_hash(key),),
            ),
            bool,
        )
    )


@pytest.mark.parametrize("route", [OPENAI_ROUTE, ANTHROPIC_ROUTE, GEMINI_ROUTE])
@pytest.mark.parametrize("value", [None, False, 7, "not-a-list"])
def test_a_body_guardrails_that_is_not_a_list_is_dropped_for_a_guarded_key(
    rig: AuthGuardrailRig, route: str, value: JsonValue
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [GUARDRAIL]})
        _drain(rig)

        response: Final = _send(gateway, route, {**_route_body(route, "Hello"), "guardrails": value}, key)

        assert response.status_code == 200, response.text
        assert len(rig.vendor.drain()) == 1
        assert [b'"guardrails"' in request.body for request in rig.upstreams[route].drain()] == [False]


def test_a_blocked_response_with_no_cost_result_keeps_its_usage_on_the_failure_row(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [BLOCKING_POST_CALL]})
        _drain(rig)

        response: Final = _send(
            gateway, ANTHROPIC_ROUTE, {**_route_body(ANTHROPIC_ROUTE, "Hello"), "model": UNPRICED_MODEL}, key
        )

        assert response.status_code == 400, response.text
        assert len(rig.upstreams[ANTHROPIC_ROUTE].drain()) == 1
        rows: Final = _row_fields(gateway, key)
        assert [row["status"] for row in rows] == ["failure"], rows
        assert float(str(rows[0]["spend"])) == 0, rows
        assert int(str(rows[0]["total_tokens"])) > 0, rows
        assert rows[0]["api_base"], rows


def test_a_tpm_limit_counts_blocked_responses_with_no_cost_result(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [BLOCKING_POST_CALL]}, tpm_limit=600)
        _drain(rig)
        body: Final = {**_route_body(ANTHROPIC_ROUTE, "Hello"), "model": UNPRICED_MODEL}

        limited: Final = eventually(
            lambda: _send(gateway, ANTHROPIC_ROUTE, body, key),
            lambda response: response.status_code != 400,
            seconds=70,
        )

        assert limited.status_code == 429, limited.text


def test_a_tag_scoped_post_call_guardrail_scans_a_reply_for_a_request_with_its_tag(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [TAG_SCOPED_POST_CALL]})
        _drain(rig)

        response: Final = _send(
            gateway,
            OPENAI_ROUTE,
            {**_route_body(OPENAI_ROUTE, "Hello"), "metadata": {"tags": [POST_CALL_TAG]}},
            key,
        )

        assert response.status_code == 400, response.text
        assert len(rig.upstreams[OPENAI_ROUTE].drain()) == 1


def test_custom_endpoint_writes_no_charge_for_a_response_its_post_call_guardrail_blocks(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(
            team_id=scenario.team(), metadata={"allowed_passthrough_routes": [CUSTOM_GUARDED_ROUTE]}
        )
        _drain(rig)

        response: Final = gateway.request("POST", CUSTOM_GUARDED_ROUTE, _route_body(OPENAI_ROUTE, "Hello"), key=key)

        assert response.status_code == 400, response.text
        assert len(rig.upstreams[OPENAI_ROUTE].drain()) == 1
        assert [float(str(row["spend"])) for row in _row_fields(gateway, key)] == [0], response.text


def test_a_custom_endpoint_with_only_a_pre_call_guardrail_never_calls_a_global_unified_guardrail(
    rig: AuthGuardrailRig,
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        custom_key: Final = scenario.key(metadata={"allowed_passthrough_routes": [CUSTOM_PRE_CALL_ROUTE]})
        built_in_key: Final = scenario.key(metadata={"guardrails": [GENERIC_POST_CALL]})
        _drain(rig)
        rig.global_unified_vendor.drain()

        custom_response: Final = gateway.request(
            "POST", CUSTOM_PRE_CALL_ROUTE, _route_body(OPENAI_ROUTE, "Hello"), key=custom_key
        )

        assert custom_response.status_code == 200, custom_response.text
        assert len(rig.vendor.drain()) == 1
        assert rig.global_unified_vendor.drain() == ()

        built_in_response: Final = _send(gateway, OPENAI_ROUTE, _route_body(OPENAI_ROUTE, "Hello"), built_in_key)

        assert built_in_response.status_code == 200, built_in_response.text
        assert len(rig.global_unified_vendor.drain()) == 1


@pytest.mark.parametrize(
    "field", ["Hello there", 7, ["a", "b"], {"guardrails": 7}], ids=["text", "int", "list", "dict"]
)
def test_a_custom_endpoints_own_body_field_still_gets_its_reply_scanned(
    rig: AuthGuardrailRig, field: JsonValue
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"allowed_passthrough_routes": [CUSTOM_GUARDED_ROUTE]})

        response: Final = gateway.request(
            "POST", CUSTOM_GUARDED_ROUTE, {**_route_body(OPENAI_ROUTE, "Hello"), "body": field}, key=key
        )

        assert response.status_code == 400, response.text
        assert "synthetic block" in response.text


@pytest.mark.parametrize(
    ("guardrail", "fields"),
    [
        (GUARDRAIL, {"proxy_server_request": 5}),
        (GUARDRAIL, {"metadata": {"user_api_key_alias": 5}}),
        (TAG_SCOPED_POST_CALL, {"metadata": {"tags": [1, {"a": 1}]}}),
        (
            None,
            {
                "metadata": {"_pipeline_managed_guardrails": []},
                "litellm_metadata": {"_guardrail_pipelines": [["p", "q"]]},
            },
        ),
    ],
    ids=["proxy-server-request", "key-alias", "tags", "pipeline-slots"],
)
def test_a_malformed_proxy_field_in_a_body_still_gets_the_provider_reply(
    rig: AuthGuardrailRig, guardrail: str | None, fields: Mapping[str, JsonValue]
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [guardrail]} if guardrail else {})
        _drain(rig)

        response: Final = _send(gateway, OPENAI_ROUTE, {**_route_body(OPENAI_ROUTE, "Hello"), **fields}, key)

        assert response.status_code == 200, response.text
        assert len(rig.upstreams[OPENAI_ROUTE].drain()) == 1


def test_a_monitored_lakera_flag_keeps_the_provider_id_on_the_row(rig: AuthGuardrailRig) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key(metadata={"guardrails": [MONITOR_POST_CALL]})
        _drain(rig)

        response: Final = _send(gateway, ANTHROPIC_ROUTE, _route_body(ANTHROPIC_ROUTE, MONITORED_PROMPT), key)

        assert response.status_code == 200, response.text
        assert response.content == _MONITORED_ANTHROPIC_BODY
        assert [row["request_id"] for row in _row_fields(gateway, key)] == [MONITORED_REPLY_ID]


@pytest.mark.parametrize(
    ("policy", "forwarded"), [(PIPELINE_ALLOWS, True), (PIPELINE_BEFORE_THE_CALL, False)], ids=["post-call", "pre-call"]
)
def test_only_a_pipeline_that_runs_before_the_provider_withholds_the_client_user_id(
    rig: AuthGuardrailRig, policy: str, forwarded: bool
) -> None:
    gateway: Final = rig.owned.gateway
    with gateway.scenario() as scenario:
        key: Final = scenario.key()
        _drain(rig)
        body: Final = {
            **_route_body(ANTHROPIC_ROUTE, "Hello"),
            "metadata": {"tags": [policy], "user_id": "client-user-1"},
        }

        response: Final = _send(gateway, ANTHROPIC_ROUTE, body, key)

        assert response.status_code == 200, response.text
        assert [b"client-user-1" in request.body for request in rig.upstreams[ANTHROPIC_ROUTE].drain()] == [forwarded]

import uuid
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL: Final = "claude-sonnet-4-5"
_FIVE_MINUTES: Final = {"type": "ephemeral"}
_PONG: Final[tuple[dict[str, JsonValue], ...]] = ({"type": "text", "text": "PONG"},)
_SUBAGENT_BILLING: Final = (
    "x-anthropic-billing-header: cc_version=2.1.283.00; cc_entrypoint=sdk-cli; cc_is_subagent=true;"
)
_MAIN_AGENT_BILLING: Final = "x-anthropic-billing-header: cc_version=2.1.283.00; cc_entrypoint=sdk-cli;"


def _unmarked_claude_code_turn() -> dict[str, JsonValue]:
    return cc.without_cache_breakpoints({**cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"), "stream": False})


def _one_shot_turn(billing: str) -> dict[str, JsonValue]:
    return {
        "system": [{"type": "text", "text": billing}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": f"Summarize {uuid.uuid4().hex}"}]}],
        "max_tokens": 512,
        "stream": False,
    }


def _forward(
    gateway: Gateway,
    sent: Mapping[str, JsonValue],
    key_fields: Mapping[str, JsonValue],
    deployment: Mapping[str, JsonValue],
) -> tuple[cc.CacheForwarded, dict[str, JsonValue], dict[str, JsonValue]]:
    identity: Final = f"msg_{uuid.uuid4().hex}"

    def respond(request: Request) -> Reply:
        return Reply(body=cc.message_reply(identity, _MODEL, _PONG, {"input_tokens": 12, "output_tokens": 4}))

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(
            model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY, **deployment
        )
        key: Final = scenario.key(**key_fields)
        response: Final = gateway.request(
            "POST", "/v1/messages", {**sent, "model": model}, key=key, headers=cc.cli_headers(key, cc.CACHING_CLI_BETA)
        )
        assert response.status_code == 200, response.text
        upstream: Final = wire.drain()
        rows: Final = eventually(
            lambda: read_rows(
                "SELECT model_id, metadata->>'litellm_gateway_injected_cache' AS injected_for "
                'FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
    assert len(upstream) == 1, upstream
    return cc.cache_forwarded(sent, upstream[0]), cc.JSON_OBJECT.validate_json(upstream[0].body), rows[0]


def test_prompt_caching_key_adds_breakpoints_to_the_system_prompt_and_last_message(gateway: Gateway) -> None:
    sent: Final = _unmarked_claude_code_turn()
    forwarded, _, spend_row = _forward(gateway, sent, {"enable_prompt_caching": True}, {})
    assert cc.cache_breakpoints(sent) == {}
    assert forwarded.breakpoints == {"system[2]": _FIVE_MINUTES, "messages[0].content[7]": _FIVE_MINUTES}
    assert forwarded.other_changes == {}
    assert forwarded.caching_betas == ("extended-cache-ttl-2025-04-11", "prompt-caching-scope-2026-01-05")
    assert spend_row["injected_for"] == spend_row["model_id"], spend_row


def test_prompt_caching_key_turns_a_string_system_prompt_into_a_cached_block(gateway: Gateway) -> None:
    sent: Final = {**_unmarked_claude_code_turn(), "system": "Synthetic agent identity system prompt."}
    forwarded, _, _ = _forward(gateway, sent, {"enable_prompt_caching": True}, {})
    assert forwarded.breakpoints == {"system[0]": _FIVE_MINUTES, "messages[0].content[7]": _FIVE_MINUTES}
    assert forwarded.other_changes == {
        "system": {
            "expected": "Synthetic agent identity system prompt.",
            "upstream": [{"type": "text", "text": "Synthetic agent identity system prompt."}],
        }
    }


def test_key_without_prompt_caching_adds_no_breakpoints(gateway: Gateway) -> None:
    sent: Final = _unmarked_claude_code_turn()
    forwarded, _, spend_row = _forward(gateway, sent, {}, {})
    assert forwarded.breakpoints == {}
    assert forwarded.other_changes == {}
    assert spend_row["injected_for"] is None, spend_row


def test_prompt_caching_key_adds_nothing_when_the_client_marked_any_breakpoint(gateway: Gateway) -> None:
    sent: Final = cc.with_block_breakpoint(_unmarked_claude_code_turn(), "tools", 23, _FIVE_MINUTES)
    forwarded, _, spend_row = _forward(gateway, sent, {"enable_prompt_caching": True}, {})
    assert forwarded.breakpoints == {"tools[23]": _FIVE_MINUTES}
    assert forwarded.other_changes == {}
    assert spend_row["injected_for"] is None, spend_row


def test_prompt_caching_key_adds_nothing_when_extra_body_unmarks_the_client_tool_breakpoint(gateway: Gateway) -> None:
    marked: Final = cc.with_block_breakpoint(_unmarked_claude_code_turn(), "tools", 23, _FIVE_MINUTES)
    envelope: Final = {"tools": cc.without_cache_breakpoints(marked)["tools"]}
    sent: Final = {**marked, "extra_body": envelope}
    forwarded, _, spend_row = _forward(gateway, sent, {"enable_prompt_caching": True}, {})
    assert forwarded.breakpoints == {"tools[23]": _FIVE_MINUTES}
    assert forwarded.other_changes == {"extra_body": {"expected": envelope, "upstream": None}}
    assert spend_row["injected_for"] is None, spend_row


@pytest.mark.parametrize(
    ("billing", "received"),
    (
        pytest.param(_SUBAGENT_BILLING, {}, id="subagent-gets-no-breakpoints"),
        pytest.param(
            _MAIN_AGENT_BILLING,
            {"system[0]": _FIVE_MINUTES, "messages[0].content[0]": _FIVE_MINUTES},
            id="main-agent-gets-breakpoints",
        ),
    ),
)
def test_prompt_caching_key_skips_one_shot_claude_code_subagent_requests(
    gateway: Gateway, billing: str, received: dict[str, JsonValue]
) -> None:
    sent: Final = _one_shot_turn(billing)
    forwarded, _, _ = _forward(gateway, sent, {"enable_prompt_caching": True}, {})
    assert forwarded.breakpoints == received
    assert forwarded.other_changes == {}


def _three_client_breakpoints() -> dict[str, JsonValue]:
    system_marked: Final = cc.with_block_breakpoint(_unmarked_claude_code_turn(), "system", 1, _FIVE_MINUTES)
    return {**cc.with_block_breakpoint(system_marked, "tools", 23, _FIVE_MINUTES), "cache_control": dict(_FIVE_MINUTES)}


def _four_client_breakpoints() -> dict[str, JsonValue]:
    return cc.with_block_breakpoint(_three_client_breakpoints(), "system", 2, _FIVE_MINUTES)


@pytest.mark.parametrize(
    ("sent", "received"),
    (
        pytest.param(
            _three_client_breakpoints,
            {
                "cache_control": _FIVE_MINUTES,
                "system[1]": _FIVE_MINUTES,
                "tools[23]": _FIVE_MINUTES,
                "messages[0].content[7]": _FIVE_MINUTES,
            },
            id="three-client-breakpoints-plus-last-message",
        ),
        pytest.param(
            _four_client_breakpoints,
            {
                "cache_control": _FIVE_MINUTES,
                "system[1]": _FIVE_MINUTES,
                "system[2]": _FIVE_MINUTES,
                "tools[23]": _FIVE_MINUTES,
            },
            id="four-client-breakpoints-leave-no-room",
        ),
    ),
)
def test_configured_injection_points_stop_at_anthropics_four_breakpoint_limit(
    gateway: Gateway, sent: Callable[[], dict[str, JsonValue]], received: dict[str, JsonValue]
) -> None:
    client_body: Final = sent()
    forwarded, _, _ = _forward(
        gateway,
        client_body,
        {},
        {
            "cache_control_injection_points": [
                {"location": "message", "role": "system"},
                {"location": "message", "index": -1},
            ]
        },
    )
    assert forwarded.breakpoints == received
    assert forwarded.other_changes == {}


def test_fallback_deployment_gets_its_own_breakpoints_and_the_injection_credit(gateway: Gateway) -> None:
    identity: Final = f"msg_{uuid.uuid4().hex}"
    sent: Final = _unmarked_claude_code_turn()

    def fail(request: Request) -> Reply:
        return Reply(status=500, body=b'{"type":"error","error":{"type":"api_error","message":"primary down"}}')

    def respond(request: Request) -> Reply:
        return Reply(body=cc.message_reply(identity, _MODEL, _PONG, {"input_tokens": 12, "output_tokens": 4}))

    with wire_server(fail) as primary_wire, wire_server(respond) as fallback_wire, gateway.scenario() as scenario:
        primary: Final = scenario.model(
            model=f"anthropic/{_MODEL}", api_base=primary_wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        fallback: Final = scenario.model(
            model=f"anthropic/{_MODEL}", api_base=fallback_wire.url, api_key=cc.ANTHROPIC_API_KEY
        )
        key: Final = scenario.key(enable_prompt_caching=True)
        response: Final = gateway.request(
            "POST",
            "/v1/messages",
            {**sent, "model": primary, "fallbacks": [fallback]},
            key=key,
            headers=cc.cli_headers(key, cc.CACHING_CLI_BETA),
        )
        assert response.status_code == 200, response.text
        primary_legs: Final = primary_wire.drain()
        fallback_legs: Final = fallback_wire.drain()
        rows: Final = eventually(
            lambda: read_rows(
                "SELECT spend.model_id, deployment.model_name, "
                "spend.metadata->>'litellm_gateway_injected_cache' AS injected_for "
                'FROM "LiteLLM_SpendLogs" spend JOIN "LiteLLM_ProxyModelTable" deployment '
                "ON deployment.model_id = spend.model_id WHERE spend.request_id=%s",
                (identity,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
    injected: Final = {"system[2]": _FIVE_MINUTES, "messages[0].content[7]": _FIVE_MINUTES}
    legs: Final = [cc.cache_forwarded(sent, leg) for leg in (*primary_legs, *fallback_legs)]
    assert len(primary_legs) == 1, primary_legs
    assert len(fallback_legs) == 1, fallback_legs
    assert [(leg.breakpoints, leg.other_changes) for leg in legs] == [(injected, {}), (injected, {})], legs
    assert rows[0]["model_name"] == fallback, rows
    assert rows[0]["injected_for"] == rows[0]["model_id"], rows

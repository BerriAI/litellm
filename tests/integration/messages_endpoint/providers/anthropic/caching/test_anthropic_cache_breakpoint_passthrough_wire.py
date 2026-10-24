import uuid
from collections.abc import Callable
from typing import Final

import pytest
from integration._support import claude_code as cc
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL: Final = "claude-sonnet-4-5"
_FIVE_MINUTES: Final = {"type": "ephemeral"}
_ONE_HOUR: Final = {"type": "ephemeral", "ttl": "1h"}
_ONE_HOUR_GLOBAL: Final = {"type": "ephemeral", "ttl": "1h", "scope": "global"}
_CLAUDE_CODE_BREAKPOINTS: Final = {
    "system[1]": _FIVE_MINUTES,
    "system[2]": _FIVE_MINUTES,
    "messages[0].content[7]": _FIVE_MINUTES,
}


def _claude_code_default(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return body


def _one_hour_global_system_prompt(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cc.with_block_breakpoint(body, "system", 2, _ONE_HOUR_GLOBAL)


def _one_hour_tool_breakpoint(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return cc.with_block_breakpoint(body, "tools", 23, _ONE_HOUR)


def _top_level_automatic_caching(body: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {**body, "cache_control": dict(_FIVE_MINUTES)}


@pytest.mark.parametrize(
    ("mark", "received"),
    (
        pytest.param(_claude_code_default, _CLAUDE_CODE_BREAKPOINTS, id="claude-code-default"),
        pytest.param(
            _one_hour_global_system_prompt,
            {**_CLAUDE_CODE_BREAKPOINTS, "system[2]": _ONE_HOUR_GLOBAL},
            id="1h-global-scope-on-system-prompt",
        ),
        pytest.param(
            _one_hour_tool_breakpoint,
            {**_CLAUDE_CODE_BREAKPOINTS, "tools[23]": _ONE_HOUR},
            id="1h-on-last-tool",
        ),
        pytest.param(
            _top_level_automatic_caching,
            {"cache_control": _FIVE_MINUTES, **_CLAUDE_CODE_BREAKPOINTS},
            id="top-level-automatic-caching",
        ),
    ),
)
def test_client_cache_breakpoints_reach_anthropic_unchanged(
    gateway: Gateway, mark: Callable[[dict[str, JsonValue]], dict[str, JsonValue]], received: dict[str, JsonValue]
) -> None:
    sent: Final = mark({**cc.claude_code_request(f"cache-bust-{uuid.uuid4().hex}"), "stream": False})

    def respond(request: Request) -> Reply:
        return Reply(
            body=cc.message_reply(
                f"msg_{uuid.uuid4().hex}",
                _MODEL,
                ({"type": "text", "text": "PONG"},),
                {"input_tokens": 12, "output_tokens": 4},
            )
        )

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(model=f"anthropic/{_MODEL}", api_base=wire.url, api_key=cc.ANTHROPIC_API_KEY)
        response: Final = gateway.request(
            "POST", "/v1/messages", {**sent, "model": model}, headers=cc.cli_headers(gateway.key, cc.CACHING_CLI_BETA)
        )
        assert response.status_code == 200, response.text
        upstream: Final = wire.drain()
    assert len(upstream) == 1, upstream
    forwarded: Final = cc.cache_forwarded(sent, upstream[0])
    assert cc.cache_breakpoints(sent) == received
    assert forwarded.breakpoints == received
    assert forwarded.other_changes == {}
    assert forwarded.caching_betas == ("extended-cache-ttl-2025-04-11", "prompt-caching-scope-2026-01-05")

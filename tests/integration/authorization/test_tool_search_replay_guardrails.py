import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support import tool_search as ts
from integration._support.client import Gateway
from integration._support.database import write_rows
from integration._support.process import graceful_stop_seconds, owned_proxy
from integration._support.wire import Wire, wire_server
from pydantic import JsonValue

from tests.integration.caching.response_cache_case import await_entry, redis_store

pytestmark: Final = pytest.mark.timeout(2 * graceful_stop_seconds() + 120)

_POLICY: Final = "integration-tool-policy"
_BLOCK: Final = "tool-permission-block"
_REWRITE: Final = "tool-permission-rewrite"


@dataclass(frozen=True, slots=True)
class _Guarded:
    proxy: Gateway
    wire: Wire
    model: str
    shell: str
    blocked: str


def _permission(name: str, shell: str, action: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "tool_permission",
            "mode": "pre_call",
            "default_on": False,
            "default_action": "allow",
            "on_disallowed_action": action,
            "rules": [{"id": "deny_shell", "tool_name": f"^{shell}$", "decision": "deny"}],
        },
    }


def _config(directory: Path, model: str, upstream: str, shell: str) -> Path:
    path: Final = directory / "tool_search_guardrails.yaml"
    path.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {"model": pcb.MODEL, "api_base": f"{upstream}/v1", "api_key": "sk-upstream"},
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                },
                "router_settings": {"disable_cooldowns": True},
                "litellm_settings": {
                    "cache": True,
                    "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
                },
                "guardrails": [
                    {
                        "guardrail_name": _POLICY,
                        "litellm_params": {"guardrail": "tool_policy", "mode": "pre_call", "default_on": True},
                    },
                    _permission(_BLOCK, shell, "block"),
                    _permission(_REWRITE, shell, "rewrite"),
                ],
            }
        )
    )
    return path


@pytest.fixture(scope="module")
def guarded(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Guarded]:
    suffix: Final = uuid.uuid4().hex[:12]
    shell: Final = f"run_shell_{suffix}"
    blocked: Final = f"wipe_disk_{suffix}"
    model: Final = f"tool-search-guarded-{suffix}"
    policy: Final = session_gateway.request(
        "POST", "/v1/tool/policy", {"tool_name": blocked, "input_policy": "blocked"}
    )
    assert policy.status_code == 200, policy.text
    directory: Final = tmp_path_factory.mktemp("tool_search_guardrails")
    try:
        with (
            wire_server(ts.scripted) as wire,
            owned_proxy(session_gateway, directory, {}, config=_config(directory, model, wire.url, shell), workers=2)
            as proxy,
        ):
            yield _Guarded(proxy, wire, model, shell, blocked)
    finally:
        write_rows('DELETE FROM "LiteLLM_ToolTable" WHERE tool_name = %s', (blocked,))


def _call(marker: str, name: str) -> dict[str, JsonValue]:
    return {
        "id": ts.call_id(marker, name),
        "type": "function",
        "function": {"name": name, "arguments": '{"city": "Paris"}'},
        "provider_specific_fields": {"namespace": name},
    }


def _searched_turn(marker: str, name: str, label: str, loads: list[JsonValue]) -> list[JsonValue]:
    items: Final[list[JsonValue]] = [ts.search_call(marker, label), ts.search_output(marker, loads, label)]
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [_call(marker, name)],
            "provider_specific_fields": {"tool_search_items": items},
        },
        ts.tool_result(marker, name),
    ]


def _two_turn_body(guarded: _Guarded, marker: str, **fields: JsonValue) -> dict[str, JsonValue]:
    return {
        "model": guarded.model,
        "messages": [
            {"role": "user", "content": pcb.prompt(marker)},
            *_searched_turn(marker, ts.WEATHER, "a", [ts.loaded(ts.WEATHER)]),
            *_searched_turn(marker, ts.TIME, "b", [ts.loaded(ts.TIME), ts.loaded(guarded.shell)]),
        ],
        "tools": ts.chat_tools(ts.WEATHER, ts.TIME),
        **fields,
    }


def _posts(wire: Wire, marker: str) -> tuple[bytes, ...]:
    return tuple(request.body for request in pcb.with_marker(pcb.drained_posts(wire), marker))


def _assert_refused(response: httpx.Response, wire: Wire, marker: str, message: str, tool: str) -> None:
    assert response.status_code == 400, response.text
    assert message in response.text and tool in response.text, response.text
    assert _posts(wire, marker) == (), "a refused replay reached the upstream"


def _one_search_body(
    guarded: _Guarded, marker: str, loads: list[JsonValue], **fields: JsonValue
) -> dict[str, JsonValue]:
    return {
        "model": guarded.model,
        "messages": [{"role": "user", "content": pcb.prompt(marker)}, *_searched_turn(marker, ts.WEATHER, "a", loads)],
        "tools": ts.chat_tools(ts.WEATHER),
        **fields,
    }


def _blocked_load_body(guarded: _Guarded, marker: str) -> dict[str, JsonValue]:
    return _one_search_body(guarded, marker, [ts.loaded(ts.WEATHER), ts.loaded(guarded.blocked)])


def _cache(guarded: _Guarded, marker: str, body: dict[str, JsonValue], key: str | None = None) -> None:
    populated: Final = guarded.proxy.request("POST", "/v1/chat/completions", body, key=key)
    assert populated.status_code == 200, populated.text
    await_entry(redis_store(), str(rv.JSON_OBJECT.validate_json(populated.text)["id"]))
    hit: Final = guarded.proxy.request("POST", "/v1/chat/completions", body, key=key)
    assert hit.status_code == 200 and hit.headers.get("x-litellm-cache-key"), (hit.text, dict(hit.headers))
    assert ts.first_message(hit.text)["content"] == rv.answer(marker), hit.text
    assert len(_posts(guarded.wire, marker)) == 1


def _assert_policy_refuses_the_replay(guarded: _Guarded) -> None:
    marker: Final = uuid.uuid4().hex
    refused: Final = guarded.proxy.request("POST", "/v1/chat/completions", _blocked_load_body(guarded, marker))
    _assert_refused(refused, guarded.wire, marker, "Violated tool policy", guarded.blocked)
    detail: Final = rv.JSON_OBJECT.validate_python(rv.JSON_OBJECT.validate_json(refused.text)["error"])
    assert rv.JSON_OBJECT.validate_python(detail["provider_specific_fields"])["blocked_tools"] == [guarded.blocked]


def test_tool_policy_blocks_a_replayed_search_that_loaded_a_blocked_tool(guarded: _Guarded) -> None:
    control_marker: Final = uuid.uuid4().hex
    control: Final = guarded.proxy.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": guarded.model,
            "messages": [{"role": "user", "content": pcb.prompt(control_marker)}],
            "tools": [{"type": "function", "function": ts.function(guarded.blocked)}],
        },
    )
    _assert_refused(control, guarded.wire, control_marker, "Violated tool policy", guarded.blocked)
    _assert_policy_refuses_the_replay(guarded)
    _assert_policy_refuses_the_replay(guarded)


def test_tool_permission_block_mode_refuses_a_replayed_search_that_loaded_a_denied_tool(guarded: _Guarded) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = guarded.proxy.request(
        "POST", "/v1/chat/completions", _two_turn_body(guarded, marker, guardrails=[_BLOCK])
    )
    _assert_refused(response, guarded.wire, marker, "Violated guardrail policy", guarded.shell)


def test_tool_permission_rewrite_mode_drops_only_the_search_that_loaded_the_denied_tool(guarded: _Guarded) -> None:
    marker: Final = uuid.uuid4().hex
    response: Final = guarded.proxy.request(
        "POST", "/v1/chat/completions", _two_turn_body(guarded, marker, guardrails=[_REWRITE])
    )
    assert response.status_code == 200, response.text
    assert ts.first_message(response.text)["content"] == rv.answer(marker), response.text
    (posted,) = pcb.with_marker(pcb.drained_posts(guarded.wire), marker)
    assert ts.replayed(posted) == [
        ts.search_call(marker, "a"),
        ts.search_output(marker, [ts.loaded(ts.WEATHER)], "a"),
        ts.replayed_call(marker, ts.WEATHER),
        ts.replayed_call(marker, ts.TIME),
    ]
    assert guarded.shell not in posted.body.decode()


def test_guardrails_leave_an_allowed_replay_untouched(guarded: _Guarded) -> None:
    marker: Final = uuid.uuid4().hex
    body: Final = _one_search_body(guarded, marker, [ts.loaded(ts.WEATHER)], guardrails=[_BLOCK])
    response: Final = guarded.proxy.request("POST", "/v1/chat/completions", body)
    assert response.status_code == 200, response.text
    (posted,) = pcb.with_marker(pcb.drained_posts(guarded.wire), marker)
    assert ts.replayed(posted) == [
        ts.search_call(marker, "a"),
        ts.search_output(marker, [ts.loaded(ts.WEATHER)], "a"),
        ts.replayed_call(marker, ts.WEATHER),
    ]


def test_guardrails_serve_an_allowed_replay_from_the_cache(guarded: _Guarded) -> None:
    marker: Final = uuid.uuid4().hex
    _cache(guarded, marker, _one_search_body(guarded, marker, [ts.loaded(ts.WEATHER)], guardrails=[_BLOCK]))


def test_tool_policy_refuses_a_replay_whose_answer_an_exempt_key_cached(guarded: _Guarded) -> None:
    marker: Final = uuid.uuid4().hex
    body: Final = _blocked_load_body(guarded, marker)
    with guarded.proxy.scenario() as scenario:
        exempt: Final = scenario.key(models=[guarded.model], metadata={"disable_global_guardrails": True})
        _cache(guarded, marker, body, key=exempt)
        refused: Final = guarded.proxy.request("POST", "/v1/chat/completions", body)
        _assert_refused(refused, guarded.wire, marker, "Violated tool policy", guarded.blocked)


def test_tool_permission_block_mode_refuses_a_replay_whose_answer_is_cached(guarded: _Guarded) -> None:
    marker: Final = uuid.uuid4().hex
    _cache(guarded, marker, _two_turn_body(guarded, marker))
    response: Final = guarded.proxy.request(
        "POST", "/v1/chat/completions", _two_turn_body(guarded, marker, guardrails=[_BLOCK])
    )
    _assert_refused(response, guarded.wire, marker, "Violated guardrail policy", guarded.shell)


def test_tool_permission_rewrite_mode_is_not_served_the_cached_unrewritten_answer(guarded: _Guarded) -> None:
    marker: Final = uuid.uuid4().hex
    _cache(guarded, marker, _two_turn_body(guarded, marker))
    rewritten: Final = _two_turn_body(guarded, marker, guardrails=[_REWRITE])
    first: Final = guarded.proxy.request("POST", "/v1/chat/completions", rewritten)
    assert first.status_code == 200, first.text
    (posted,) = pcb.with_marker(pcb.drained_posts(guarded.wire), marker)
    assert guarded.shell not in posted.body.decode()
    await_entry(redis_store(), str(rv.JSON_OBJECT.validate_json(first.text)["id"]))
    hit: Final = guarded.proxy.request("POST", "/v1/chat/completions", rewritten)
    assert hit.status_code == 200 and hit.headers.get("x-litellm-cache-key"), (hit.text, dict(hit.headers))
    assert ts.first_message(hit.text)["content"] == rv.answer(marker), hit.text
    assert _posts(guarded.wire, marker) == ()

import uuid
from collections.abc import Sequence
from typing import Final

import httpx
import pytest
from integration._support import prompt_cache_breakpoint as pcb
from integration._support import responses_vendor as rv
from integration._support import tool_search as ts
from integration._support.client import Gateway, Scenario
from integration._support.openai_wire import answering_model_discovery, chat_reply
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

from tests.integration.caching.response_cache_case import await_entry, redis_store

pytestmark: Final = pytest.mark.timeout(120)


def _shell() -> str:
    return f"run_shell_{uuid.uuid4().hex[:12]}"


def _replay_body(model: str, marker: str, loads: Sequence[JsonValue], **fields: JsonValue) -> dict[str, JsonValue]:
    call: Final[dict[str, JsonValue]] = {
        "id": ts.call_id(marker),
        "type": "function",
        "function": {"name": ts.WEATHER, "arguments": '{"city": "Paris"}'},
        "provider_specific_fields": {"namespace": ts.WEATHER},
    }
    items: Final[list[JsonValue]] = [ts.search_call(marker), ts.search_output(marker, loads)]
    return {
        "model": model,
        "messages": [
            {"role": "user", "content": pcb.prompt(marker)},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [call],
                "provider_specific_fields": {"tool_search_items": items},
            },
            ts.tool_result(marker),
        ],
        "tools": ts.chat_tools(ts.WEATHER),
        **pcb.NO_CACHE,
        **fields,
    }


def _bridge(scenario: Scenario, wire: Wire, **parameters: JsonValue) -> str:
    return scenario.model(model=pcb.MODEL, api_base=f"{wire.url}/v1", **parameters)


def _posts(wire: Wire, marker: str) -> int:
    return len(pcb.with_marker(pcb.drained_posts(wire), marker))


def _assert_denied(response: httpx.Response, wire: Wire, marker: str, tool: str) -> None:
    assert response.status_code == 403, response.text
    error: Final = rv.JSON_OBJECT.validate_python(rv.JSON_OBJECT.validate_json(response.text)["error"])
    assert error["type"] == "tool_access_denied", response.text
    assert tool in str(error["message"]) and ts.WEATHER not in str(error["message"]), response.text
    assert _posts(wire, marker) == 0, "a denied replay reached the upstream"


def _assert_served(response: httpx.Response, wire: Wire, marker: str) -> None:
    assert response.status_code == 200, response.text
    assert ts.first_message(response.text)["content"] == rv.answer(marker), response.text
    assert _posts(wire, marker) == 1


def test_key_without_an_allowlist_still_replays_any_loaded_tool(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = _bridge(scenario, wire)
        key: Final = scenario.key(models=[model])
        body: Final = _replay_body(model, marker, [ts.loaded(ts.WEATHER), ts.loaded(_shell())])
        _assert_served(gateway.request("POST", "/v1/chat/completions", body, key=key), wire, marker)


@pytest.mark.parametrize("placement", ["tools", "extra_body", "deployment", "absent"])
def test_allowlisted_key_cannot_replay_a_search_that_loaded_another_tool(gateway: Gateway, placement: str) -> None:
    marker: Final = uuid.uuid4().hex
    shell: Final = _shell()
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = _bridge(scenario, wire, **({"tools": [ts.TOOL_SEARCH]} if placement == "deployment" else {}))
        key: Final = scenario.key(models=[model], metadata={"allowed_tools": [ts.WEATHER]})
        loads: Final[list[JsonValue]] = [ts.loaded(ts.WEATHER), ts.loaded(shell)]
        overrides: Final[dict[str, dict[str, JsonValue]]] = {
            "tools": {},
            "extra_body": {
                "tools": [ts.deferred(ts.WEATHER)],
                "extra_body": {"tools": [ts.TOOL_SEARCH, ts.loaded(ts.WEATHER)]},
            },
            "deployment": {"tools": [ts.deferred(ts.WEATHER)]},
            "absent": {"tools": [{"type": "function", "function": ts.function(ts.WEATHER)}]},
        }
        body: Final = {**_replay_body(model, marker, loads), **overrides[placement]}
        _assert_denied(gateway.request("POST", "/v1/chat/completions", body, key=key), wire, marker, shell)


def test_allowlisted_key_cannot_replay_a_denied_namespace_member(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    shell: Final = _shell()
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = _bridge(scenario, wire)
        key: Final = scenario.key(models=[model], metadata={"allowed_tools": [ts.WEATHER]})
        body: Final = _replay_body(model, marker, [ts.namespace("ops", ts.WEATHER, shell)])
        _assert_denied(gateway.request("POST", "/v1/chat/completions", body, key=key), wire, marker, shell)


def _chat_upstream(request: Request) -> Reply:
    marker: Final = rv.newest_marker(request.body.decode())
    return chat_reply(f"chatcmpl-{marker}", "gpt-4o-mini", rv.answer(marker), stream=False)


def test_plain_chat_deployment_also_counts_replayed_search_loads(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    shell: Final = _shell()
    with wire_server(answering_model_discovery(_chat_upstream)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], metadata={"allowed_tools": [ts.WEATHER]})
        body: Final = _replay_body(model, marker, [ts.loaded(shell)])
        _assert_denied(gateway.request("POST", "/v1/chat/completions", body, key=key), wire, marker, shell)


def test_team_allowlist_applies_and_a_key_allowlist_overrides_it(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    shell: Final = _shell()
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = _bridge(scenario, wire)
        team: Final = scenario.team(models=[model], metadata={"allowed_tools": [ts.WEATHER]})
        team_key: Final = scenario.key(team_id=team, models=[model])
        wider_key: Final = scenario.key(team_id=team, models=[model], metadata={"allowed_tools": [ts.WEATHER, shell]})
        body: Final = _replay_body(model, marker, [ts.loaded(ts.WEATHER), ts.loaded(shell)])
        _assert_denied(gateway.request("POST", "/v1/chat/completions", body, key=team_key), wire, marker, shell)
        _assert_served(gateway.request("POST", "/v1/chat/completions", body, key=wider_key), wire, marker)


def test_allowed_replay_is_served_and_its_cache_hit_still_checks_the_allowlist(gateway: Gateway) -> None:
    marker: Final = uuid.uuid4().hex
    shell: Final = _shell()
    with wire_server(ts.scripted) as wire, gateway.scenario() as scenario:
        model: Final = _bridge(scenario, wire)
        allowed: Final = scenario.key(models=[model], metadata={"allowed_tools": [ts.WEATHER]})
        cached: Final = {
            key: value
            for key, value in _replay_body(model, marker, [ts.loaded(ts.WEATHER)]).items()
            if key != "cache"
        }
        first: Final = gateway.request("POST", "/v1/chat/completions", cached, key=allowed)
        _assert_served(first, wire, marker)
        await_entry(redis_store(), str(rv.JSON_OBJECT.validate_json(first.text)["id"]))
        hit: Final = gateway.request("POST", "/v1/chat/completions", cached, key=allowed)
        assert hit.status_code == 200 and hit.headers.get("x-litellm-cache-key"), (hit.text, dict(hit.headers))
        assert ts.first_message(hit.text)["content"] == rv.answer(marker), hit.text
        shell_body: Final = {
            key: value
            for key, value in _replay_body(model, marker, [ts.loaded(shell)]).items()
            if key != "cache"
        }
        populated: Final = gateway.request("POST", "/v1/chat/completions", shell_body)
        assert populated.status_code == 200, populated.text
        await_entry(redis_store(), str(rv.JSON_OBJECT.validate_json(populated.text)["id"]))
        assert _posts(wire, marker) == 1
        _assert_denied(gateway.request("POST", "/v1/chat/completions", shell_body, key=allowed), wire, marker, shell)

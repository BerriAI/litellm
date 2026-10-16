import itertools
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    JsonRpc,
    McpCaller,
    Outcome,
    ScriptedTool,
    disconnecting_tool,
    echo_tool,
    listed_tools,
    mcp_peer,
    register_mcp,
    scripted_peer,
    slow_tool,
    text_result,
    tool_calls,
)
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply
from pydantic import BaseModel, TypeAdapter

_ECHO: Final = "catalog-echo:"
_PROBE: Final = "catalog-probe"
_DESCRIPTION: Final = "Look up one record"
_GUARDRAIL_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    texts = list(inputs.get("texts") or [])\n'
    '    function = inputs.get("tools", [{}])[0].get("function", {})\n'
    f'    if "{_PROBE}" in texts:\n'
    f'        return block("{_ECHO}" + json_stringify({{"description": function.get("description")}}))\n'
    "    return allow()\n"
)
_BURST: Final = 20
_OUTAGE: Final = 6
_SPEND_NONCES: Final = (
    "SELECT status, metadata->'mcp_tool_call_metadata'->'arguments'->>'nonce' AS nonce"
    ' FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s'
)
_OBJECTS: Final = TypeAdapter(Mapping[str, object])
_STRINGS: Final = TypeAdapter(Mapping[str, str])
_ECHOED: Final = TypeAdapter(Mapping[str, str | None])


class _Content(BaseModel):
    text: str


class _Result(BaseModel):
    content: tuple[_Content, ...]


class _RpcReply(BaseModel):
    id: int
    result: _Result


class _SessionsReport(BaseModel):
    worker_pid: int


@pytest.fixture(scope="module")
def echo_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    base: Final = _OBJECTS.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    guardrail: Final = {
        "guardrail_name": "catalog-echo-" + uuid.uuid4().hex[:8],
        "litellm_params": {
            "guardrail": "custom_code",
            "mode": "pre_mcp_call",
            "default_on": True,
            "custom_code": _GUARDRAIL_CODE,
        },
    }
    path: Final = tmp_path_factory.mktemp("failure-recovery") / "config.yaml"
    path.write_text(yaml.safe_dump({**base, "guardrails": [guardrail]}))
    return path


def _call(caller: McpCaller, name: str, arguments: dict[str, object], entry: EntryPoint, identity: str) -> Outcome:
    return caller.call(name, arguments, identity if entry == "rest" else None)


def _health(gateway: Gateway, key: str, identity: str) -> str:
    response: Final = gateway.client.get(
        "/v1/mcp/server/health", headers={"x-litellm-api-key": key}, params={"server_ids": [identity]}
    )
    assert response.status_code == 200, response.text
    statuses: Final = {entry["server_id"]: entry["status"] for entry in response.json()}
    assert identity in statuses, response.text
    return str(statuses[identity])


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_tool_error_surfaces_as_error_with_the_peer_message_and_never_as_success(
    gateway: Gateway, entry: EntryPoint
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "toolerr" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias)
        peer.drain()
        outcome: Final = _call(caller, f"{alias}-fail", {}, entry, identity)
        assert outcome.error is not None, f"failing tool reported success: {outcome.raw}"
        assert "Error executing tool fail" in str(outcome.raw), outcome.raw
        assert len(tool_calls(peer.drain())) == 1
        recovered: Final = _call(caller, f"{alias}-add", {"a": 2, "b": 3}, entry, identity)
        assert recovered.text == "5", recovered.raw


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_unreachable_peer_errors_while_a_healthy_sibling_keeps_serving(gateway: Gateway, entry: EntryPoint) -> None:
    with mcp_peer() as healthy, gateway.scenario() as scenario:
        good: Final = "good" + uuid.uuid4().hex[:8]
        bad: Final = "bad" + uuid.uuid4().hex[:8]
        good_id: Final = register_mcp(scenario, healthy, good)
        bad_id: Final = register_mcp(scenario, healthy, bad, url="http://127.0.0.1:9/mcp")
        key: Final = scenario.key(object_permission={"mcp_servers": [good_id, bad_id]})
        caller: Final = McpCaller(gateway, key, entry, good)
        listing: Final = caller.list_tools(good_id if entry == "rest" else None)
        assert listing.error is None, listing.raw
        assert {f"{good}-add", "add"} & set(listing.tools), listing.raw
        assert not {f"{bad}-add"} & set(listing.tools) or entry != "rest", listing.raw
        healthy.drain()
        served: Final = _call(caller, f"{good}-add", {"a": 2, "b": 3}, entry, good_id)
        assert served.text == "5", served.raw
        assert len(tool_calls(healthy.drain())) == 1
        if entry == "server_mcp":
            return
        failed: Final = _call(McpCaller(gateway, key, entry, bad), f"{bad}-add", {"a": 2, "b": 3}, entry, bad_id)
        assert failed.error is not None, f"call to unreachable peer succeeded: {failed.raw}"
        assert failed.text != "5"


def test_unreachable_peer_is_reported_unhealthy_and_healthy_peer_healthy(gateway: Gateway) -> None:
    with mcp_peer() as healthy, gateway.scenario() as scenario:
        good: Final = "hgood" + uuid.uuid4().hex[:8]
        bad: Final = "hbad" + uuid.uuid4().hex[:8]
        good_id: Final = register_mcp(scenario, healthy, good)
        bad_id: Final = register_mcp(scenario, healthy, bad, url="http://127.0.0.1:9/mcp")
        assert _health(gateway, gateway.key, good_id) == "healthy"
        assert _health(gateway, gateway.key, bad_id) == "unhealthy"


def test_slow_peer_beyond_configured_timeout_errors_and_does_not_hang_the_gateway(gateway: Gateway) -> None:
    with scripted_peer(slow_tool("nap", 4), echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = "slow" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, timeout=1)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        peer.drain()
        outcome: Final = caller.call(f"{alias}-nap", {})
        assert outcome.error is not None, f"call past the timeout succeeded: {outcome.raw}"
        assert outcome.text != "slept"
        quick: Final = caller.call(f"{alias}-echo", {"k": "v"})
        assert quick.text == '{"k": "v"}', quick.raw


def test_peer_disconnecting_mid_response_errors_and_the_next_call_succeeds(gateway: Gateway) -> None:
    with scripted_peer(disconnecting_tool("drop"), echo_tool("echo")) as peer, gateway.scenario() as scenario:
        alias: Final = "drop" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        for entry in ("mcp", "rest"):
            caller = McpCaller(gateway, key, entry, alias)
            dropped = caller.call(f"{alias}-drop", {}, identity if entry == "rest" else None)
            assert dropped.error is not None, f"half-written reply became success on {entry}: {dropped.raw}"
            recovered = caller.call(f"{alias}-echo", {"n": 1}, identity if entry == "rest" else None)
            assert recovered.text == '{"n": 1}', recovered.raw


def test_peer_restart_on_the_same_url_is_picked_up_without_gateway_restart(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        alias: Final = "restart" + uuid.uuid4().hex[:8]
        with mcp_peer() as first:
            identity: Final = register_mcp(scenario, first, alias)
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            assert set(listed_tools(gateway, key, identity)) == {"add", "multiply", "fail"}
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        down: Final = caller.call(f"{alias}-add", {"a": 1, "b": 1})
        assert down.error is not None, down.raw
        with scripted_peer(echo_tool("add")) as replacement:
            edited: Final = gateway.request(
                "PUT",
                "/v1/mcp/server",
                {"server_id": identity, "server_name": alias, "alias": alias, **replacement.registration()},
            )
            assert edited.status_code in (200, 202), edited.text
            back: Final = eventually(
                lambda: caller.call(f"{alias}-add", {"a": 1, "b": 1}), lambda outcome: outcome.error is None, seconds=40
            )
            assert back.text == '{"a": 1, "b": 1}', back.raw
            assert len(tool_calls(replacement.drain())) >= 1


def _rpc_reply(raw: str) -> _RpcReply:
    data: Final = tuple(line[5:].strip() for line in raw.splitlines() if line.startswith("data:"))
    return _RpcReply.model_validate_json(data[-1] if data else raw)


def _call_params(call: Mapping[str, object]) -> Mapping[str, object]:
    return _OBJECTS.validate_python(_OBJECTS.validate_python(call["body"])["params"])


def _call_nonce(call: Mapping[str, object]) -> str:
    return _STRINGS.validate_python(_call_params(call)["arguments"])["nonce"]


def _listed(caller: McpCaller, name: str) -> None:
    listing: Final = eventually(caller.list_tools, lambda outcome: name in outcome.tools, seconds=45)
    assert listing.error is None, (caller.gateway.client.base_url, listing.raw)


@dataclass(frozen=True, slots=True)
class _Worker:
    caller: McpCaller
    pid: int


def _worker(proxy: Gateway, key: str, alias: str) -> _Worker:
    sessions: Final = proxy.client.get("/v1/mcp/sessions", headers={"x-litellm-api-key": proxy.key})
    assert sessions.status_code == 200, sessions.text
    return _Worker(McpCaller(proxy, key, "mcp", alias), _SessionsReport.model_validate_json(sessions.text).worker_pid)


def _served(worker: _Worker, name: str, nonce: str) -> None:
    served: Final = worker.caller.call(name, {"nonce": nonce})
    assert served.text == "found", (worker.pid, served.raw)


def _probed_description(worker: _Worker, name: str) -> str | None:
    """The description the pre_mcp_call guardrail on that worker was handed, recovered from its block reason."""
    blocked: Final = worker.caller.call(name, {"nonce": _PROBE})
    assert blocked.error is not None, (worker.pid, blocked.raw)
    carrier: Final = next((item.text for item in _rpc_reply(blocked.raw).result.content if _ECHO in item.text), None)
    assert carrier is not None, (worker.pid, blocked.raw)
    return _ECHOED.validate_json(carrier.split(_ECHO, 1)[1])["description"]


def _catalog_is_cold(worker: _Worker, name: str) -> bool:
    return not _probed_description(worker, name)


@pytest.mark.timeout(600)
def test_worker_restart_cools_its_listed_catalog_while_the_sibling_worker_keeps_serving(
    gateway: Gateway, echo_config: Path, tmp_path: Path
) -> None:
    tool: Final = ScriptedTool("lookup", lambda _: text_result("found"), description=_DESCRIPTION)
    with scripted_peer(tool) as peer, gateway.scenario() as scenario:
        alias: Final = "cold" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        name: Final = f"{alias}-lookup"
        with owned_proxy_process(gateway, tmp_path / "sibling", {}, config=echo_config) as sibling_proxy:
            sibling: Final = _worker(sibling_proxy.gateway, key, alias)
            with owned_proxy_process(gateway, tmp_path / "first", {}, config=echo_config) as first_proxy:
                first: Final = _worker(first_proxy.gateway, key, alias)
                assert first.pid != sibling.pid
                _served(first, name, "first-unlisted")
                _served(sibling, name, "sibling-unlisted")
                assert _catalog_is_cold(first, name) and _catalog_is_cold(sibling, name)
                _listed(first.caller, name)
                assert _probed_description(first, name) == _DESCRIPTION
                assert _catalog_is_cold(sibling, name), "a listing on one worker warmed its sibling"
                _listed(sibling.caller, name)
                assert _probed_description(sibling, name) == _DESCRIPTION
            _served(sibling, name, "sibling-alone")
            with owned_proxy_process(gateway, tmp_path / "restarted", {}, config=echo_config) as restarted_proxy:
                restarted: Final = _worker(restarted_proxy.gateway, key, alias)
                assert restarted.pid not in (first.pid, sibling.pid)
                assert _catalog_is_cold(restarted, name), "a restarted worker kept the old process's catalog"
                assert _probed_description(sibling, name) == _DESCRIPTION
                _served(restarted, name, "restarted-unlisted")
                _listed(restarted.caller, name)
                assert _probed_description(restarted, name) == _DESCRIPTION
        calls: Final = tool_calls(peer.drain())
        assert [_call_nonce(call) for call in calls] == [
            "first-unlisted",
            "sibling-unlisted",
            "sibling-alone",
            "restarted-unlisted",
        ], calls
        assert all(set(_call_params(call)) - {"_meta"} == {"name", "arguments"} for call in calls), calls


def _outage_echo(name: str, failures: int) -> ScriptedTool:
    attempts: Final = itertools.count(1)

    def respond(params: JsonRpc) -> Reply | JsonRpc:
        if next(attempts) <= failures:
            return Reply(status=503, body=b'{"error": "scripted outage"}')
        return text_result(_STRINGS.validate_python(params["arguments"])["nonce"])

    return ScriptedTool(name, respond)


def _echo_call(caller: McpCaller, name: str, nonce: str) -> Outcome:
    return caller.call(name, {"nonce": nonce})


def _logged_nonces(key: str, count: int) -> tuple[tuple[str, str], ...]:
    rows: Final = eventually(
        lambda: read_rows(_SPEND_NONCES, (sha256(key.encode()).hexdigest(), "call_mcp_tool")),
        lambda found: len(found) >= count,
        seconds=70,
    )
    return tuple(sorted((str(row["status"]), str(row["nonce"])) for row in rows))


def test_peer_outage_during_a_bounded_burst_fails_exactly_the_outage_calls_and_lands_each_call_once(
    gateway: Gateway, peer: Gateway
) -> None:
    with scripted_peer(_outage_echo("echo", _OUTAGE)) as upstream, gateway.scenario() as scenario:
        alias: Final = "burst" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, upstream, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        name: Final = f"{alias}-echo"
        callers: Final = (McpCaller(gateway, key, "mcp", alias), McpCaller(peer, key, "mcp", alias))
        for caller in callers:
            _listed(caller, name)
        upstream.drain()
        nonces: Final = tuple(uuid.uuid4().hex for _ in range(_BURST))
        with ThreadPoolExecutor(max_workers=_BURST) as pool:
            outcomes: Final = tuple(pool.map(_echo_call, itertools.cycle(callers), itertools.repeat(name), nonces))
        raws: Final = [outcome.raw for outcome in outcomes]
        failed: Final = tuple(nonce for nonce, outcome in zip(nonces, outcomes) if outcome.error is not None)
        assert len(failed) == _OUTAGE, raws
        assert all(outcome.error is not None or outcome.text == nonce for nonce, outcome in zip(nonces, outcomes)), raws
        assert all(_rpc_reply(outcome.raw).id == 1 for outcome in outcomes), raws
        burst_calls: Final = tool_calls(upstream.drain())
        assert sorted(_call_nonce(call) for call in burst_calls) == sorted(nonces), burst_calls
        assert all(_call_params(call)["arguments"] == {"nonce": _call_nonce(call)} for call in burst_calls), burst_calls
        assert all(set(_call_params(call)) - {"_meta"} == {"name", "arguments"} for call in burst_calls), burst_calls
        recovered: Final = tuple(
            _echo_call(caller, name, nonce) for caller, nonce in zip(itertools.cycle(callers), failed)
        )
        assert [outcome.text for outcome in recovered] == list(failed), [outcome.raw for outcome in recovered]
        assert sorted(_call_nonce(call) for call in tool_calls(upstream.drain())) == sorted(failed)
        assert _logged_nonces(key, _BURST + len(failed)) == tuple(
            sorted([("success", nonce) for nonce in nonces] + [("failure", nonce) for nonce in failed])
        )

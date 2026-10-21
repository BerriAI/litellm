"""pre_mcp_call guardrails are handed the tool entry ``tools/list`` served to the caller.

One owned proxy carries a default-on ``custom_code`` pre_mcp_call guardrail. At listing time it masks
``SECRET`` out of every scanned text. At call time, when an argument carries the probe marker, it
blocks and echoes the description and parameters it was handed, which is the only way to observe from outside
what metadata the gateway attached to the hook
"""

import json
import threading
import uuid
from collections.abc import Callable, Generator, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.mcp import (
    EntryPoint,
    JsonRpc,
    McpCaller,
    ScriptedTool,
    listed_tools,
    openapi_peer,
    register_mcp,
    scripted_peer,
    text_result,
    tool_calls,
)
from integration._support.process import owned_proxy
from pydantic import TypeAdapter

_ECHO: Final = "catalog-echo:"
_PROBE: Final = "catalog-probe"
_CALLERS_PER_SERVER: Final = 256
_PIN_SECONDS: Final = 120
_COLD: Final[tuple[str, JsonRpc]] = ("", {"type": "object", "properties": {}, "additionalProperties": False})
_PID: Final = TypeAdapter(int)
_HEADERS: Final = TypeAdapter(dict[bytes, bytes])
_SCHEMA: Final = TypeAdapter(dict[str, object])
_LOOKUP_SCHEMA: Final = {
    "type": "object",
    "properties": {"probe": {"type": "string", "description": "a probe marker"}},
    "additionalProperties": False,
}
_GUARDRAIL_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    '    texts = list(inputs.get("texts") or [])\n'
    '    function = inputs.get("tools", [{}])[0].get("function", {})\n'
    f'    if "{_PROBE}" in texts:\n'
    f'        return block("{_ECHO}" + json_stringify('
    '{"description": function.get("description"), "parameters": function.get("parameters")}))\n'
    '    masked = [text.replace("SECRET", "[MASKED]") for text in texts]\n'
    "    if masked != texts:\n"
    "        return modify(texts=masked)\n"
    "    return allow()\n"
)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("listed-tool-metadata")
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": "catalog-echo-" + uuid.uuid4().hex[:8],
            "litellm_params": {
                "guardrail": "custom_code",
                "mode": "pre_mcp_call",
                "default_on": True,
                "custom_code": _GUARDRAIL_CODE,
            },
        }
    ]
    config["general_settings"] = {**config["general_settings"], "proxy_config_reload_interval_seconds": 1}
    path: Final = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        gateway_from_environment() as gateway,
        owned_proxy(gateway, directory, {"KEEPALIVE_TIMEOUT": "120"}, config=path, workers=2) as candidate,
        ExitStack() as stack,
    ):
        _two_workers(stack, candidate)
        yield candidate


def _strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
        return
    children: Final = value.values() if isinstance(value, Mapping) else value if isinstance(value, list) else ()
    for child in children:
        yield from _strings(child)


def _decoded(raw: str) -> object:
    data: Final = tuple(line[5:].strip() for line in raw.splitlines() if line.startswith("data:"))
    return json.loads(data[-1] if data else raw)


def _echoed(raw: str) -> tuple[str | None, Mapping[str, object] | None]:
    """The (description, parameters) the guardrail was handed, recovered from its block reason."""
    carrier: Final = next((text for text in _strings(_decoded(raw)) if _ECHO in text), None)
    assert carrier is not None, raw
    echoed, _ = json.JSONDecoder().raw_decode(carrier.split(_ECHO, 1)[1])
    assert isinstance(echoed, dict), carrier
    return echoed.get("description"), echoed.get("parameters")


def _probe(caller: McpCaller, name: str, server_id: str) -> tuple[str | None, Mapping[str, object] | None]:
    outcome: Final = caller.call(name, {"probe": _PROBE}, server_id=server_id)
    assert outcome.error is not None, outcome.raw
    return _echoed(outcome.raw)


def _worker(gateway: Gateway) -> int:
    response: Final = gateway.request("GET", "/debug/memory/summary")
    assert response.status_code == 200, response.text
    return _PID.validate_python(response.json()["worker_pid"])


@contextmanager
def _pinned(rig: Gateway) -> Generator[Gateway, None, None]:
    """A single keep-alive connection, so every request on it is served by the worker that accepted it."""
    limits: Final = httpx.Limits(max_connections=1, max_keepalive_connections=1)
    with httpx.Client(base_url=rig.client.base_url, timeout=15, trust_env=False, limits=limits) as client:
        yield Gateway(client, rig.key, rig.upstream_url)


def _connection(stack: ExitStack, rig: Gateway, wanted: Callable[[int], bool]) -> tuple[Gateway, int]:
    """A pinned connection to a worker ``wanted`` accepts; a worker still starting up accepts nothing yet."""

    def attempt() -> tuple[Gateway, int] | None:
        with ExitStack() as candidate:
            gateway: Final = candidate.enter_context(_pinned(rig))
            pid: Final = _worker(gateway)
            if not wanted(pid):
                return None
            stack.enter_context(candidate.pop_all())
            return gateway, pid

    found: Final = eventually(attempt, lambda pair: pair is not None, seconds=_PIN_SECONDS)
    assert found is not None
    return found


def _two_workers(stack: ExitStack, rig: Gateway) -> tuple[tuple[Gateway, int], tuple[Gateway, int]]:
    """Connects while the first worker is busy answering, so the idle worker wins the accept race."""
    first: Final = _connection(stack, rig, lambda _: True)
    stop: Final = threading.Event()

    def keep_busy() -> None:
        while not stop.is_set():
            _worker(first[0])

    with ThreadPoolExecutor(max_workers=1) as pool:
        busy: Final = pool.submit(keep_busy)
        try:
            other: Final = _connection(stack, rig, lambda pid: pid != first[1])
        finally:
            stop.set()
        busy.result()
    return first, other


def _served_name(gateway: Gateway, key: str, identity: str, tool: str) -> str:
    """The prefixed name this worker lists for ``tool`` once its registry reload carries the server."""
    listing: Final = eventually(
        lambda: McpCaller(gateway, key, "rest").list_tools(server_id=identity),
        lambda value: value.ok and any(full.endswith(tool) for full in value.tools),
    )
    return next(full for full in listing.tools if full.endswith(tool))


def _settled_probe(
    gateway: Gateway, key: str, name: str, identity: str
) -> tuple[str | None, Mapping[str, object] | None]:
    """The hook echo for a direct call, once this worker's registry reload carries the server."""
    outcome: Final = eventually(
        lambda: McpCaller(gateway, key, "rest").call(name, {"probe": _PROBE}, server_id=identity),
        lambda value: value.error is not None and _ECHO in value.raw,
    )
    return _echoed(outcome.raw)


def _forwarded_tenants(observed: tuple[dict[str, object], ...]) -> frozenset[bytes]:
    return frozenset(_HEADERS.validate_python(item["headers"]).get(b"x-tenant", b"") for item in observed) - {b""}


def _lookup_tool(description: str | Callable[[Mapping[str, str]], str] = "Look up one record") -> ScriptedTool:
    return ScriptedTool("lookup", lambda _: text_result("found"), description=description, input_schema=_LOOKUP_SCHEMA)


@pytest.mark.parametrize("entry", ["rest", "mcp"])
def test_pre_call_hook_receives_the_description_and_input_schema_the_caller_was_listed(
    rig: Gateway, entry: EntryPoint
) -> None:
    schema: Final = {"type": "object", "properties": {"probe": {"type": "string", "description": "a probe marker"}}}
    tool: Final = ScriptedTool(
        "lookup", lambda _: text_result("found"), description="Look up one record", input_schema=schema
    )
    with scripted_peer(tool) as peer, rig.scenario() as scenario:
        alias: Final = "meta" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(rig, key, entry, headers={"x-mcp-servers": alias})
        assert caller.initialize().ok
        listed: Final = caller.list_tools(server_id=identity)
        assert listed.ok, listed.raw
        name: Final = next(full for full in listed.tools if full.endswith("lookup"))
        description, parameters = _probe(caller, name, identity)
        assert description == "Look up one record", (description, parameters)
        assert parameters is not None and parameters.get("properties") == schema["properties"], parameters


def test_each_caller_is_evaluated_against_the_catalog_its_own_forwarded_headers_produced(rig: Gateway) -> None:
    tool: Final = ScriptedTool(
        "report",
        lambda _: text_result("ok"),
        description=lambda headers: f"Report for tenant {headers.get('x-tenant', 'nobody')}",
    )
    with scripted_peer(tool) as peer, rig.scenario() as scenario:
        alias: Final = "tenant" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, extra_headers=["x-tenant"])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        acme: Final = McpCaller(rig, key, "server_mcp", alias, headers={"x-tenant": "acme"})
        globex: Final = McpCaller(rig, key, "server_mcp", alias, headers={"x-tenant": "globex"})
        acme_listing: Final = acme.list_tools()
        globex_listing: Final = globex.list_tools()
        assert acme_listing.ok and globex_listing.ok, (acme_listing.raw, globex_listing.raw)
        name: Final = next(full for full in acme_listing.tools if full.endswith("report"))
        acme_seen, _ = _probe(acme, name, identity)
        globex_seen, _ = _probe(globex, name, identity)
        assert (acme_seen, globex_seen) == ("Report for tenant acme", "Report for tenant globex"), (
            "each caller's tools/call must be evaluated against the catalog its own headers listed"
        )


def test_call_is_evaluated_against_the_masked_description_the_listing_served(rig: Gateway) -> None:
    tool: Final = ScriptedTool("read_note", lambda _: text_result("note"), description="Read a note")
    with scripted_peer(tool) as peer, rig.scenario() as scenario:
        alias: Final = "note" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario, peer, alias, tool_name_to_description={"read_note": "Read a SECRET note"}
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        served: Final = listed_tools(rig, key, identity)
        name: Final = next(full for full in served if full.endswith("read_note"))
        assert served[name]["description"] == "Read a [MASKED] note", served[name]
        seen, _ = _probe(McpCaller(rig, key, "rest"), name, identity)
        assert seen == "Read a [MASKED] note", "the admin override must not restore wording the listing masked"


def test_openapi_call_is_evaluated_against_the_masked_override_the_listing_served(rig: Gateway) -> None:
    with openapi_peer() as peer, rig.scenario() as scenario:
        alias: Final = "pets" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario, peer, alias, tool_name_to_description={"getpet": "Fetch one SECRET pet"}
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        served: Final = listed_tools(rig, key, identity)
        name: Final = next(full for full in served if full.endswith("getpet"))
        assert served[name]["description"] == "Fetch one [MASKED] pet", served[name]
        seen, parameters = _probe(McpCaller(rig, key, "rest"), name, identity)
        assert seen == "Fetch one [MASKED] pet", "the OpenAPI call path must hand hooks the entry the listing served"
        assert parameters is not None and "petId" in parameters.get("properties", {}), parameters
        assert not [call for call in peer.drain() if call["path"].startswith("/pets")], "blocked before upstream"


def test_openapi_call_is_evaluated_against_the_entry_this_key_was_listed_not_the_last_listing(
    rig: Gateway,
) -> None:
    with openapi_peer() as peer, rig.scenario() as scenario:
        alias: Final = "pets" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario, peer, alias, tool_name_to_description={"getpet": "Fetch one SECRET pet"}
        )
        guarded: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        opted_out: Final = scenario.key(
            object_permission={"mcp_servers": [identity]}, metadata={"disable_global_guardrails": True}
        )
        guarded_served: Final = listed_tools(rig, guarded, identity)
        opted_out_served: Final = listed_tools(rig, opted_out, identity)
        name: Final = next(full for full in guarded_served if full.endswith("getpet"))
        assert (guarded_served[name]["description"], opted_out_served[name]["description"]) == (
            "Fetch one [MASKED] pet",
            "Fetch one SECRET pet",
        ), (guarded_served[name], opted_out_served[name])
        seen, _ = _probe(McpCaller(rig, guarded, "rest"), name, identity)
        assert seen == "Fetch one [MASKED] pet", (
            "the guarded key must be evaluated against its own listing, not the opted-out key's later one"
        )


def test_direct_call_without_a_listing_hands_the_hook_no_metadata_on_either_worker(rig: Gateway) -> None:
    with scripted_peer(_lookup_tool()) as peer, ExitStack() as stack:
        (first, first_pid), (other, other_pid) = _two_workers(stack, rig)
        with first.scenario() as scenario:
            identity: Final = register_mcp(scenario, peer, "cold" + uuid.uuid4().hex[:8])
            observer: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            name: Final = _served_name(first, observer, identity, "lookup")
            seen: Final = tuple(_settled_probe(gateway, key, name, identity) for gateway in (first, other))
            assert seen == (_COLD, _COLD), (seen, first_pid, other_pid)
            assert (_worker(first), _worker(other)) == (first_pid, other_pid)
            assert tool_calls(peer.drain()) == (), "the probe is blocked at the hook, before the upstream"


def test_warm_metadata_is_local_to_the_worker_that_served_the_listing(rig: Gateway) -> None:
    with scripted_peer(_lookup_tool()) as peer, ExitStack() as stack:
        (first, first_pid), (other, other_pid) = _two_workers(stack, rig)
        with first.scenario() as scenario:
            identity: Final = register_mcp(scenario, peer, "local" + uuid.uuid4().hex[:8])
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            name: Final = _served_name(first, key, identity, "lookup")
            warm: Final = ("Look up one record", _LOOKUP_SCHEMA)
            assert _probe(McpCaller(first, key, "rest"), name, identity) == warm, first_pid
            assert _settled_probe(other, key, name, identity) == _COLD, (
                "a worker that never served this caller a listing has no catalog for it",
                other_pid,
            )
            assert _served_name(other, key, identity, "lookup") == name
            assert _probe(McpCaller(other, key, "rest"), name, identity) == warm, other_pid
            assert (_worker(first), _worker(other)) == (first_pid, other_pid)
            assert tool_calls(peer.drain()) == ()


def test_listed_tools_without_a_description_still_hand_the_hook_the_schema_the_listing_served(rig: Gateway) -> None:
    open_schema: Final[JsonRpc] = {"type": "object", "properties": {}, "additionalProperties": True}
    undescribed: Final = ScriptedTool("undescribed", lambda _: text_result("ok"), input_schema=open_schema)
    blank: Final = ScriptedTool("blank", lambda _: text_result("ok"), description="", input_schema=open_schema)
    with scripted_peer(undescribed, blank) as peer, _pinned(rig) as worker, worker.scenario() as scenario:
        pid: Final = _worker(worker)
        identity: Final = register_mcp(scenario, peer, "bare" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        served: Final = listed_tools(worker, key, identity)
        names: Final = tuple(next(full for full in served if full.endswith(tool)) for tool in ("undescribed", "blank"))
        assert tuple((served[name].get("description") or "", served[name]["inputSchema"]) for name in names) == (
            ("", open_schema),
            ("", open_schema),
        ), served
        seen: Final = tuple(_probe(McpCaller(worker, key, "rest"), name, identity) for name in names)
        assert seen == (("", open_schema), ("", open_schema)), (seen, _COLD)
        assert _worker(worker) == pid
        assert tool_calls(peer.drain()) == ()


def test_hook_receives_the_nested_schema_with_the_leaves_the_listing_masked(rig: Gateway) -> None:
    schema: Final = {
        "type": "object",
        "required": ["filter"],
        "additionalProperties": False,
        "properties": {
            "filter": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "SECRET path"},
                    "tags": {"type": "array", "items": {"type": "string", "description": "one SECRET tag"}},
                },
            }
        },
    }
    masked: Final = _SCHEMA.validate_python(json.loads(json.dumps(schema).replace("SECRET", "[MASKED]")))
    tool: Final = ScriptedTool(
        "search", lambda _: text_result("hit"), description="Search SECRET records", input_schema=schema
    )
    with scripted_peer(tool) as peer, _pinned(rig) as worker, worker.scenario() as scenario:
        pid: Final = _worker(worker)
        identity: Final = register_mcp(scenario, peer, "nested" + uuid.uuid4().hex[:8])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        served: Final = listed_tools(worker, key, identity)
        name: Final = next(full for full in served if full.endswith("search"))
        assert (served[name]["description"], served[name]["inputSchema"]) == ("Search [MASKED] records", masked), served
        assert _probe(McpCaller(worker, key, "rest"), name, identity) == ("Search [MASKED] records", masked), (
            "the hook must be handed the nested schema exactly as the listing served it"
        )
        assert _worker(worker) == pid
        assert tool_calls(peer.drain()) == ()


def test_a_server_definition_update_drops_the_catalog_on_every_worker_until_the_caller_lists_again(
    rig: Gateway,
) -> None:
    with scripted_peer(_lookup_tool()) as peer, ExitStack() as stack:
        (first, first_pid), (other, other_pid) = _two_workers(stack, rig)
        with first.scenario() as scenario:
            identity: Final = register_mcp(scenario, peer, "upd" + uuid.uuid4().hex[:8])
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            name: Final = _served_name(first, key, identity, "lookup")
            assert _served_name(other, key, identity, "lookup") == name
            before: Final = tuple(_probe(McpCaller(gateway, key, "rest"), name, identity) for gateway in (first, other))
            assert before == (("Look up one record", _LOOKUP_SCHEMA),) * 2, before
            updated: Final = first.request(
                "PUT",
                "/v1/mcp/server",
                {"server_id": identity, "tool_name_to_description": {"lookup": "Audited lookup"}},
            )
            assert updated.status_code == 202, updated.text
            assert _probe(McpCaller(first, key, "rest"), name, identity) == _COLD, (
                "the worker that applied the update must drop its catalog at once",
                first_pid,
            )
            assert (
                eventually(lambda: _probe(McpCaller(other, key, "rest"), name, identity), lambda seen: seen == _COLD)
                == _COLD
            ), other_pid
            relisted: Final = tuple(
                listed_tools(gateway, key, identity)[name]["description"] for gateway in (first, other)
            )
            assert relisted == ("Audited lookup", "Audited lookup"), relisted
            after: Final = tuple(_probe(McpCaller(gateway, key, "rest"), name, identity) for gateway in (first, other))
            assert after == (("Audited lookup", _LOOKUP_SCHEMA),) * 2, after
            assert (_worker(first), _worker(other)) == (first_pid, other_pid)
            assert tool_calls(peer.drain()) == ()


def test_a_listing_in_flight_across_a_server_update_does_not_resurrect_the_old_catalog(rig: Gateway) -> None:
    started: Final = threading.Event()
    release: Final = threading.Event()

    def describe(_: Mapping[str, str]) -> str:
        started.set()
        assert release.wait(20), "the listing was never released"
        return "Look up one record"

    with (
        scripted_peer(_lookup_tool(describe)) as peer,
        ExitStack() as stack,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        worker, pid = _connection(stack, rig, lambda _: True)
        sibling, _ = _connection(stack, rig, lambda candidate: candidate == pid)
        with worker.scenario() as scenario:
            identity: Final = register_mcp(scenario, peer, "stale" + uuid.uuid4().hex[:8])
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            release.set()
            name: Final = _served_name(worker, key, identity, "lookup")
            assert _probe(McpCaller(worker, key, "rest"), name, identity) == ("Look up one record", _LOOKUP_SCHEMA)
            started.clear()
            release.clear()
            pending: Final = pool.submit(McpCaller(worker, key, "rest").list_tools, identity)
            assert started.wait(10), "the upstream never saw the in-flight listing"
            updated: Final = sibling.request(
                "PUT",
                "/v1/mcp/server",
                {"server_id": identity, "tool_name_to_description": {"lookup": "Audited lookup"}},
            )
            assert updated.status_code == 202, updated.text
            release.set()
            stale: Final = pending.result(timeout=20)
            assert stale.ok, stale.raw
            assert _probe(McpCaller(worker, key, "rest"), name, identity) == _COLD, (
                "a listing fetched before the update must not be recorded after it"
            )
            assert listed_tools(worker, key, identity)[name]["description"] == "Audited lookup"
            assert _probe(McpCaller(worker, key, "rest"), name, identity) == ("Audited lookup", _LOOKUP_SCHEMA)
            assert (_worker(worker), _worker(sibling)) == (pid, pid)
            assert tool_calls(peer.drain()) == ()


def test_a_server_keeps_the_newest_256_caller_catalogs_and_evicts_the_oldest(rig: Gateway) -> None:
    tool: Final = _lookup_tool(lambda headers: f"Lookup for tenant {headers.get('x-tenant', 'nobody')}")
    with scripted_peer(tool) as peer, _pinned(rig) as worker, worker.scenario() as scenario:
        pid: Final = _worker(worker)
        alias: Final = "cap" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, extra_headers=["x-tenant"])
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        tenants: Final = tuple(f"t{index}" for index in range(_CALLERS_PER_SERVER + 1))
        callers: Final = {
            tenant: McpCaller(worker, key, "server_mcp", alias, headers={"x-tenant": tenant}) for tenant in tenants
        }
        listings: Final = tuple(callers[tenant].list_tools() for tenant in tenants)
        assert all(listing.ok for listing in listings), [listing.raw for listing in listings if not listing.ok]
        name: Final = next(full for full in listings[0].tools if full.endswith("lookup"))

        def seen(tenant: str) -> str | None:
            return _probe(callers[tenant], name, identity)[0]

        assert (seen("t0"), seen("t1"), seen(tenants[-1])) == (
            "",
            "Lookup for tenant t1",
            f"Lookup for tenant {tenants[-1]}",
        ), "the oldest of 257 callers is evicted, the newest 256 keep their own catalog"
        assert callers["t0"].list_tools().ok
        assert (seen("t0"), seen("t1")) == ("Lookup for tenant t0", ""), "relisting makes t0 newest and evicts t1"
        assert _worker(worker) == pid
        observed: Final = peer.drain()
        assert _forwarded_tenants(observed) == frozenset(tenant.encode() for tenant in tenants), len(observed)
        assert tool_calls(observed) == ()


def test_an_admin_include_disabled_tools_listing_does_not_warm_the_runtime_catalog(rig: Gateway) -> None:
    """``include_disabled_tools=true`` is the admin-only configuration view, not a listing the caller
    runs against: recording it would warm tools/call metadata no runtime listing ever served."""
    tool: Final = _lookup_tool()
    with scripted_peer(tool) as peer, rig.scenario() as scenario:
        alias: Final = "adminview" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        admin: Final = rig.key
        caller: Final = McpCaller(rig, admin, "rest")
        name: Final = eventually(
            lambda: caller.call(f"{alias}-lookup", {"probe": _PROBE}, server_id=identity),
            lambda value: value.error is not None and _ECHO in value.raw,
        )
        assert _echoed(name.raw) == _COLD, "before any listing the call is cold"

        view: Final = eventually(
            lambda: rig.client.get(
                "/mcp-rest/tools/list",
                params={"server_id": identity, "include_disabled_tools": "true"},
                headers={"x-litellm-api-key": admin},
            ),
            lambda response: response.status_code == 200
            and any(entry["name"].endswith("lookup") for entry in response.json()["tools"]),
        )
        assert view.status_code == 200, view.text

        after_view: Final = _probe(caller, f"{alias}-lookup", identity)
        assert after_view == _COLD, (
            "the admin-only include_disabled_tools view must not record the caller's listed-tools slot"
        )

        runtime: Final = caller.list_tools(server_id=identity)
        assert runtime.ok, runtime.raw
        assert _probe(caller, f"{alias}-lookup", identity)[0] == "Look up one record", (
            "a genuine runtime listing still warms the slot"
        )

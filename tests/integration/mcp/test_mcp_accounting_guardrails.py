import json
import uuid
from collections.abc import Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
)
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    McpCaller,
    McpPeer,
    Outcome,
    ScriptedTool,
    mcp_peer,
    register_mcp,
    scripted_peer,
    text_result,
    tool_calls,
)
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

DEFAULT_COST: Final = 0.25
ADD_COST: Final = 0.5
FORBIDDEN: Final = "forbidden-integration-word"
SPEND_ROWS: Final = (
    'SELECT call_type, model, spend, status, metadata FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime"'
)


def _digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _rows(key: str, count: int) -> list[dict[str, JsonValue]]:
    return eventually(lambda: read_rows(SPEND_ROWS, (_digest(key),)), lambda rows: len(rows) >= count, seconds=70)


def _priced_server(scenario: Scenario, peer: McpPeer, alias: str) -> str:
    return register_mcp(
        scenario,
        peer,
        alias,
        mcp_info={
            "server_name": alias,
            "mcp_server_cost_info": {
                "default_cost_per_query": DEFAULT_COST,
                "tool_name_to_cost_per_query": {"add": ADD_COST},
            },
        },
    )


def _tool_metadata(row: dict[str, JsonValue]) -> dict[str, JsonValue]:
    metadata: Final = row["metadata"]
    assert isinstance(metadata, dict), row
    tool: Final = metadata.get("mcp_tool_call_metadata")
    assert isinstance(tool, dict), metadata
    return tool


def _call(caller: McpCaller, name: str, arguments: dict[str, object], entry: EntryPoint, identity: str) -> Outcome:
    return caller.call(name, arguments, identity if entry == "rest" else None)


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_each_tool_call_writes_one_spend_row_with_server_tool_and_configured_cost(
    gateway: Gateway, entry: EntryPoint
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "acct" + uuid.uuid4().hex[:8]
        identity: Final = _priced_server(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias)
        peer.drain()
        assert _call(caller, f"{alias}-add", {"a": 2, "b": 3}, entry, identity).text == "5"
        assert _call(caller, f"{alias}-multiply", {"a": 2, "b": 3}, entry, identity).text == "6"
        assert len(tool_calls(peer.drain())) == 2
        rows: Final = [row for row in _rows(key, 2) if row["call_type"] == "call_mcp_tool"]
        assert len(rows) == 2, rows
        by_tool: Final = {_tool_metadata(row)["name"]: row for row in rows}
        assert set(by_tool) == {"add", "multiply"}, rows
        assert float(str(by_tool["add"]["spend"])) == pytest.approx(ADD_COST)
        assert float(str(by_tool["multiply"]["spend"])) == pytest.approx(DEFAULT_COST)
        for row in rows:
            assert _tool_metadata(row)["mcp_server_name"] == alias, row
            assert row["model"] == f"MCP: {alias}-{_tool_metadata(row)['name']}", row
        later: Final = read_rows(SPEND_ROWS, (_digest(key),))
        assert len([row for row in later if row["call_type"] == "call_mcp_tool"]) == 2, later


def test_key_spend_and_key_max_budget_count_mcp_tool_calls(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "budget" + uuid.uuid4().hex[:8]
        identity: Final = _priced_server(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]}, max_budget=ADD_COST / 2)
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        assert caller.call(f"{alias}-add", {"a": 2, "b": 3}).text == "5"
        info: Final = eventually(
            lambda: gateway.client.get("/key/info", params={"key": key}, headers={"x-litellm-api-key": gateway.key}),
            lambda response: response.status_code == 200 and float(response.json()["info"]["spend"]) > 0,
            seconds=70,
        )
        assert float(info.json()["info"]["spend"]) == pytest.approx(ADD_COST)
        eventually(
            lambda: caller.call(f"{alias}-add", {"a": 2, "b": 3}),
            lambda outcome: outcome.error is not None,
            seconds=70,
        )
        peer.drain()
        denied: Final = caller.call(f"{alias}-add", {"a": 2, "b": 3})
        assert denied.error is not None and "budget" in str(denied.raw).lower(), denied.raw
        assert tool_calls(peer.drain()) == (), "over-budget call reached the peer"


@contextmanager
def _content_filter(gateway: Gateway, mode: str) -> Iterator[str]:
    name: Final = "filter" + uuid.uuid4().hex[:8]
    created: Final = gateway.client.post(
        "/guardrails",
        headers={"x-litellm-api-key": gateway.key},
        json={
            "guardrail": {
                "guardrail_name": name,
                "litellm_params": {
                    "guardrail": "litellm_content_filter",
                    "mode": mode,
                    "default_on": True,
                    "blocked_words": [{"keyword": FORBIDDEN, "action": "BLOCK"}],
                },
            }
        },
    )
    assert created.status_code == 200, created.text
    identity: Final = created.json()["guardrail_id"]
    try:
        yield name
    finally:
        deleted: Final = gateway.client.delete(f"/guardrails/{identity}", headers={"x-litellm-api-key": gateway.key})
        assert deleted.status_code == 200, deleted.text


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_pre_mcp_call_guardrail_blocks_before_the_peer_and_still_logs_spend(
    gateway: Gateway, entry: EntryPoint
) -> None:
    with _content_filter(gateway, "pre_mcp_call"), mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "guard" + uuid.uuid4().hex[:8]
        identity: Final = _priced_server(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, entry, alias)
        peer.drain()
        clean: Final = _call(caller, f"{alias}-add", {"a": 2, "b": 3}, entry, identity)
        assert clean.text == "5", clean.raw
        blocked: Final = _call(caller, f"{alias}-add", {"a": 1, "b": FORBIDDEN}, entry, identity)
        assert blocked.error is not None, f"guardrail-blocked call succeeded: {blocked.raw}"
        assert FORBIDDEN in str(blocked.raw) or "blocked" in str(blocked.raw).lower(), blocked.raw
        assert len(tool_calls(peer.drain())) == 1, "blocked call reached the peer"
        rows: Final = [row for row in _rows(key, 2) if row["call_type"] == "call_mcp_tool"]
        assert len(rows) == 2, rows
        failures: Final = [row for row in rows if row["status"] == "failure"]
        assert len(failures) == 1, rows
        assert failures[0]["model"] == f"MCP: {alias}-add", failures[0]
        assert _tool_metadata(failures[0])["mcp_server_name"] == alias, failures[0]


def test_guardrail_blocked_call_never_reaches_peer_through_the_official_client(gateway: Gateway) -> None:
    with _content_filter(gateway, "pre_mcp_call"), mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "guardsdk" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "server_mcp", alias)
        peer.drain()
        blocked: Final = caller.call(f"{alias}-add", {"a": 1, "b": FORBIDDEN})
        assert blocked.error is not None, blocked.raw
        assert tool_calls(peer.drain()) == ()
        allowed: Final = caller.call(f"{alias}-add", {"a": 4, "b": 5})
        assert allowed.text == "9", allowed.raw
        assert len(tool_calls(peer.drain())) == 1


def test_guardrail_removal_stops_blocking_without_restart(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "guardoff" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(gateway, key, "mcp", alias)
        with _content_filter(gateway, "pre_mcp_call"):
            assert caller.call(f"{alias}-add", {"a": 1, "b": FORBIDDEN}).error is not None
        peer.drain()
        eventually(
            lambda: (caller.call(f"{alias}-add", {"a": 1, "b": FORBIDDEN}), tool_calls(peer.drain()))[1],
            lambda calls: len(calls) >= 1,
            seconds=40,
        )


MASK_ME: Final = "mask-integration-secret"
MASKED: Final = "[MASKED]"
COUNT_MISMATCH: Final = "count-mismatch-marker"
LOOKUP_DESCRIPTION: Final = "Look up one record"
LOOKUP_SCHEMA: Final = {
    "type": "object",
    "properties": {"record": {"type": "string", "description": "record identifier"}},
}
SPEND_ROW: Final = 'SELECT status, metadata FROM "LiteLLM_SpendLogs" WHERE request_id = %s'
_RECORDER_CODE: Final = """\
import os

import httpx
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.integrations.custom_logger import CustomLogger

SINK = "{sink}/native"


def _record(stage, data, call_type):
    logging_obj = data.get("litellm_logging_obj")
    return {{
        "stage": stage,
        "pid": os.getpid(),
        "call_type": call_type,
        "litellm_call_id": None if logging_obj is None else logging_obj.litellm_call_id,
        "messages": data.get("messages"),
        "mcp_tool_name": data.get("mcp_tool_name"),
        "mcp_arguments": data.get("mcp_arguments"),
        "mcp_tool_description": data.get("mcp_tool_description"),
        "mcp_input_schema": data.get("mcp_input_schema"),
    }}


async def _post(record):
    async with httpx.AsyncClient(timeout=5) as client:
        await client.post(SINK, json=record)


class HookRecorder(CustomLogger):
    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        await _post(_record("pre", data, call_type))

    async def async_moderation_hook(self, data, user_api_key_dict, call_type):
        await _post(_record("during", data, call_type))

    async def async_post_mcp_tool_call_hook(self, kwargs, response_obj, start_time, end_time):
        await _post(
            {{
                "stage": "post",
                "pid": os.getpid(),
                "litellm_call_id": kwargs.get("litellm_call_id"),
                "tool": kwargs.get("mcp_tool_call_metadata"),
                "content": [item.model_dump() for item in response_obj.mcp_tool_call_response],
            }}
        )


class SinkGuardrail(CustomGuardrail):
    def __init__(self, api_base, **kwargs):
        super().__init__(**kwargs)
        self.api_base = api_base

    async def apply_guardrail(self, inputs, request_data, input_type, logging_obj=None):
        payload = {{
            "pid": os.getpid(),
            "input_type": input_type,
            "litellm_call_id": None if logging_obj is None else logging_obj.litellm_call_id,
            "texts": inputs.get("texts"),
            "tools": inputs.get("tools"),
            "structured_messages": inputs.get("structured_messages"),
            "mcp_tool_name": request_data.get("mcp_tool_name"),
        }}
        async with httpx.AsyncClient(timeout=5) as client:
            verdict = (await client.post(self.api_base, json=payload)).json()
        return {{**inputs, "texts": verdict["texts"]}}


recorder = HookRecorder()
"""


@dataclass(frozen=True, slots=True)
class Sunk:
    target: str
    body: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class HooksRig:
    gateway: Gateway
    sibling: Gateway
    sink: Wire
    guardrail: str

    def sunk(self) -> tuple[Sunk, ...]:
        return tuple(Sunk(request.target, JSON_OBJECT.validate_json(request.body)) for request in self.sink.drain())


def _guardrail_sink(request: Request) -> Reply:
    if not request.target.startswith("/guardrail"):
        return Reply()
    texts: Final = JSON_OBJECT.validate_json(request.body).get("texts")
    assert isinstance(texts, list), texts
    masked: Final = [str(text).replace(MASK_ME, MASKED) for text in texts]
    extra: Final = ["extra"] if any(COUNT_MISMATCH in text for text in masked) else []
    return Reply(body=json.dumps({"texts": [*masked, *extra]}).encode())


@pytest.fixture(scope="module")
def hooks_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[HooksRig]:
    directory: Final = tmp_path_factory.mktemp("guardrail-payloads")
    guardrail: Final = "sink" + uuid.uuid4().hex[:8]
    with wire_server(_guardrail_sink) as sink:
        (directory / "hook_recorder.py").write_text(_RECORDER_CODE.format(sink=sink.url))
        config: Final = JSON_OBJECT.validate_python(
            yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        )
        config["guardrails"] = [
            {
                "guardrail_name": guardrail,
                "litellm_params": {
                    "guardrail": "hook_recorder.SinkGuardrail",
                    "mode": ["pre_mcp_call", "post_mcp_call"],
                    "default_on": True,
                    "api_base": f"{sink.url}/guardrail",
                },
            }
        ]
        config["litellm_settings"] = {
            **object_value(config["litellm_settings"]),
            "callbacks": ["hook_recorder.recorder"],
        }
        path: Final = directory / "config.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            gateway_from_environment() as gateway,
            owned_proxy(gateway, directory, {"KEEPALIVE_TIMEOUT": "120"}, config=path, workers=2) as candidate,
            owned_proxy(gateway, directory, {}, config=path) as sibling,
        ):
            yield HooksRig(candidate, sibling, sink, guardrail)


def _worker(gateway: Gateway) -> int:
    response: Final = gateway.client.get("/debug/memory/summary", headers={"x-litellm-api-key": gateway.key})
    assert response.status_code == 200, response.text
    worker: Final = JSON_OBJECT.validate_json(response.content)["worker_pid"]
    assert isinstance(worker, int), response.text
    return worker


@contextmanager
def _pinned(gateway: Gateway) -> Generator[tuple[Gateway, int], None, None]:
    limits: Final = httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=120)
    with httpx.Client(base_url=gateway.client.base_url, timeout=15, trust_env=False, limits=limits) as client:
        pinned: Final = Gateway(client, gateway.key, gateway.upstream_url)
        yield pinned, _worker(pinned)


def _lookup_tool(result: str = "found") -> ScriptedTool:
    return ScriptedTool(
        "lookup", lambda _: text_result(result), description=LOOKUP_DESCRIPTION, input_schema=LOOKUP_SCHEMA
    )


def _generic(sunk: tuple[Sunk, ...], input_type: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(item.body for item in sunk if item.target == "/guardrail" and item.body["input_type"] == input_type)


def _native(sunk: tuple[Sunk, ...], stage: str) -> tuple[dict[str, JsonValue], ...]:
    return tuple(item.body for item in sunk if item.target == "/native" and item.body["stage"] == stage)


def _only(records: tuple[dict[str, JsonValue], ...]) -> dict[str, JsonValue]:
    assert len(records) == 1, records
    return records[0]


def _scan(sunk: tuple[Sunk, ...], call_id: JsonValue) -> dict[str, JsonValue]:
    return _only(tuple(record for record in _generic(sunk, "request") if record["litellm_call_id"] == call_id))


def _texts(content: JsonValue) -> list[JsonValue]:
    assert isinstance(content, list), content
    return [object_value(item)["text"] for item in content]


def _has_lookup(listing: Outcome) -> bool:
    return any(tool.endswith("lookup") for tool in listing.tools)


def _spend_row(call_id: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(call_id, str), call_id
    rows: Final = eventually(lambda: read_rows(SPEND_ROW, (call_id,)), lambda found: len(found) == 1, seconds=70)
    return rows[0]


def _synthetic_message(name: str, arguments: Mapping[str, str]) -> list[dict[str, str]]:
    return [{"role": "user", "content": f"Tool: {name}\nArguments: {dict(arguments)}"}]


def test_generic_sink_and_native_hooks_receive_listed_metadata_on_typed_keys_with_the_message_bytes_unchanged(
    hooks_rig: HooksRig,
) -> None:
    with (
        scripted_peer(_lookup_tool()) as peer,
        _pinned(hooks_rig.gateway) as (pinned, worker),
        pinned.scenario() as scenario,
    ):
        alias: Final = "payload" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(pinned, key, "mcp", headers={"x-mcp-servers": alias})
        assert eventually(caller.initialize, lambda outcome: outcome.ok, seconds=30).ok
        hooks_rig.sunk()
        listed: Final = eventually(caller.list_tools, _has_lookup, seconds=30)
        name: Final = next(tool for tool in listed.tools if tool.endswith("lookup"))
        scans: Final = _generic(hooks_rig.sunk(), "request")
        assert scans and all(scan["texts"] == [LOOKUP_DESCRIPTION, "record identifier"] for scan in scans), scans
        arguments: Final = {"record": "r-1"}
        outcome: Final = caller.call(name, arguments)
        assert outcome.text == "found", outcome.raw
        assert _worker(pinned) == worker
        sunk: Final = hooks_rig.sunk()
        pre: Final = _only(_native(sunk, "pre"))
        call_id: Final = pre["litellm_call_id"]
        generic: Final = _scan(sunk, call_id)
        assert generic["texts"] == [LOOKUP_DESCRIPTION, "record identifier", "r-1"], generic
        assert generic["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": LOOKUP_DESCRIPTION,
                    "parameters": {**LOOKUP_SCHEMA, "additionalProperties": False},
                    "strict": False,
                },
            }
        ], generic
        response: Final = _only(_generic(sunk, "response"))
        assert (response["texts"], response["pid"]) == (["found"], worker), response
        during: Final = _only(_native(sunk, "during"))
        post: Final = _only(_native(sunk, "post"))
        assert all(record["pid"] == worker for record in (pre, during, post)), sunk
        assert all(record["litellm_call_id"] == call_id for record in (pre, during, post)), sunk
        assert pre["call_type"] == "call_mcp_tool" and during["call_type"] == "call_mcp_tool", sunk
        assert pre["messages"] == _synthetic_message("lookup", arguments), pre
        assert during["messages"] == _synthetic_message("lookup", arguments), during
        assert (pre["mcp_tool_name"], pre["mcp_arguments"]) == ("lookup", arguments), pre
        assert pre["mcp_tool_description"] == LOOKUP_DESCRIPTION, pre
        assert pre["mcp_input_schema"] == LOOKUP_SCHEMA, pre
        assert (during["mcp_tool_description"], during["mcp_input_schema"]) == (None, None), during
        assert _texts(post["content"]) == ["found"], post
        row: Final = _spend_row(call_id)
        assert row["status"] == "success", row
        assert _tool_metadata(row)["name"] == "lookup", row
        metadata: Final = row["metadata"]
        assert isinstance(metadata, dict) and metadata["applied_guardrails"] == [hooks_rig.guardrail], metadata


def test_pre_call_mask_reaches_the_peer_and_post_call_mask_reaches_the_caller_on_one_call_id(
    hooks_rig: HooksRig,
) -> None:
    with (
        scripted_peer(_lookup_tool(f"found {MASK_ME}")) as peer,
        _pinned(hooks_rig.gateway) as (pinned, worker),
        pinned.scenario() as scenario,
    ):
        alias: Final = "mask" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(pinned, key, "mcp", headers={"x-mcp-servers": alias})
        assert eventually(caller.initialize, lambda outcome: outcome.ok, seconds=30).ok
        name: Final = next(
            tool for tool in eventually(caller.list_tools, _has_lookup, seconds=30).tools if "lookup" in tool
        )
        peer.drain()
        hooks_rig.sunk()
        outcome: Final = caller.call(name, {"record": MASK_ME})
        assert outcome.text == f"found {MASKED}", outcome.raw
        assert _worker(pinned) == worker
        reached: Final = tool_calls(peer.drain())
        assert len(reached) == 1, reached
        params: Final = object_value(JSON_OBJECT.validate_python(reached[0]["body"])["params"])
        assert params["arguments"] == {"record": MASKED}, params
        sunk: Final = hooks_rig.sunk()
        generic: Final = _scan(sunk, _only(_native(sunk, "pre"))["litellm_call_id"])
        scanned: Final = generic["texts"]
        assert isinstance(scanned, list) and scanned[-1] == MASK_ME and MASKED not in scanned, generic
        assert _only(_generic(sunk, "response"))["texts"] == [f"found {MASK_ME}"], sunk
        during: Final = _only(_native(sunk, "during"))
        assert during["mcp_arguments"] == {"record": MASKED}, during
        assert during["messages"] == _synthetic_message("lookup", {"record": MASKED}), during
        assert _only(_native(sunk, "post"))["litellm_call_id"] == generic["litellm_call_id"], sunk
        assert _spend_row(generic["litellm_call_id"])["status"] == "success"


def test_call_time_description_and_schema_come_from_the_catalog_of_the_worker_that_served_the_listing(
    hooks_rig: HooksRig,
) -> None:
    with (
        scripted_peer(_lookup_tool()) as peer,
        hooks_rig.gateway.scenario() as scenario,
        _pinned(hooks_rig.sibling) as (second, second_worker),
    ):
        alias: Final = "local" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        other: Final = McpCaller(second, key, "mcp", headers={"x-mcp-servers": alias})
        assert eventually(other.initialize, lambda outcome: outcome.ok, seconds=60).ok
        with _pinned(hooks_rig.gateway) as (first, first_worker):
            assert first_worker != second_worker
            lister: Final = McpCaller(first, key, "mcp", headers={"x-mcp-servers": alias})
            assert eventually(lister.initialize, lambda outcome: outcome.ok, seconds=30).ok
            name: Final = next(
                tool for tool in eventually(lister.list_tools, _has_lookup, seconds=30).tools if "lookup" in tool
            )
            hooks_rig.sunk()
            assert other.call(name, {"record": "r-2"}).text == "found"
            assert _worker(second) == second_worker
            elsewhere: Final = hooks_rig.sunk()
            unlisted: Final = _only(_native(elsewhere, "pre"))
            assert unlisted["pid"] == second_worker, unlisted
            assert (unlisted["mcp_tool_description"], unlisted["mcp_input_schema"]) == (None, None), unlisted
            assert _scan(elsewhere, unlisted["litellm_call_id"])["texts"] == ["r-2"], elsewhere
            assert lister.call(name, {"record": "r-3"}).text == "found"
            assert _worker(first) == first_worker
            at_lister: Final = hooks_rig.sunk()
            listed: Final = _only(_native(at_lister, "pre"))
            assert listed["pid"] == first_worker, listed
            assert (listed["mcp_tool_description"], listed["mcp_input_schema"]) == (LOOKUP_DESCRIPTION, LOOKUP_SCHEMA)
            assert _scan(at_lister, listed["litellm_call_id"])["texts"] == [
                LOOKUP_DESCRIPTION,
                "record identifier",
                "r-3",
            ]
            assert _has_lookup(other.list_tools())
            hooks_rig.sunk()
            assert other.call(name, {"record": "r-4"}).text == "found"
            assert _worker(second) == second_worker
            populated: Final = _only(_native(hooks_rig.sunk(), "pre"))
            assert populated["pid"] == second_worker, populated
            assert populated["mcp_tool_description"] == LOOKUP_DESCRIPTION, populated


@pytest.mark.parametrize("entry", ("mcp", "rest"))
@pytest.mark.parametrize("listed", (False, True))
@pytest.mark.parametrize("record", ("r-1", MASK_ME))
def test_long_descriptions_do_not_refuse_small_tpm_calls_or_change_masked_message_bytes(
    hooks_rig: HooksRig, entry: EntryPoint, listed: bool, record: str
) -> None:
    description: Final = "Gateway tool metadata. " * 300
    tool: Final = ScriptedTool(
        "lookup", lambda _: text_result("found"), description=description, input_schema=LOOKUP_SCHEMA
    )
    with (
        scripted_peer(tool) as peer,
        _pinned(hooks_rig.gateway) as (pinned, worker),
        pinned.scenario() as scenario,
    ):
        alias: Final = "quota" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]}, tpm_limit=64)
        caller: Final = McpCaller(pinned, key, entry, headers={"x-mcp-servers": alias})
        assert eventually(caller.initialize, lambda outcome: outcome.ok, seconds=30).ok
        if listed:
            listing: Final = eventually(lambda: caller.list_tools(identity), _has_lookup, seconds=30)
            assert _has_lookup(listing), listing.raw
        hooks_rig.sunk()
        peer.drain()
        arguments: Final = {"record": record}
        outcome: Final = caller.call(f"{alias}-lookup", arguments, identity)
        assert outcome.ok and outcome.text == "found", outcome.raw
        assert _worker(pinned) == worker
        reached: Final = tool_calls(peer.drain())
        assert len(reached) == 1, reached
        params: Final = object_value(JSON_OBJECT.validate_python(reached[0]["body"])["params"])
        masked_arguments: Final = {"record": record.replace(MASK_ME, MASKED)}
        assert params["arguments"] == masked_arguments, params
        sunk: Final = hooks_rig.sunk()
        pre: Final = _only(tuple(hook for hook in _native(sunk, "pre") if hook["mcp_tool_name"] == "lookup"))
        during: Final = _only(tuple(hook for hook in _native(sunk, "during") if hook["mcp_tool_name"] == "lookup"))
        assert pre["messages"] == _synthetic_message("lookup", arguments), pre
        assert during["messages"] == _synthetic_message("lookup", masked_arguments), during
        assert _spend_row(pre["litellm_call_id"])["status"] == "success"


def _nested_schema(levels: int) -> dict[str, object]:
    if levels == 0:
        return {"type": "string", "description": "deepest leaf"}
    return {"type": "object", "properties": {"a": _nested_schema(levels - 1)}}


def test_a_schema_past_the_scan_depth_is_not_published_while_one_at_the_limit_is_scanned_on_the_call(
    hooks_rig: HooksRig,
) -> None:
    shallow: Final = ScriptedTool("shallow", lambda _: text_result("found"), input_schema=_nested_schema(49))
    deep: Final = ScriptedTool("deep", lambda _: text_result("found"), input_schema=_nested_schema(50))
    with (
        scripted_peer(shallow, deep) as peer,
        _pinned(hooks_rig.gateway) as (pinned, worker),
        pinned.scenario() as scenario,
    ):
        alias: Final = "depth" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(pinned, key, "mcp", headers={"x-mcp-servers": alias})
        assert eventually(caller.initialize, lambda outcome: outcome.ok, seconds=30).ok
        listing: Final = eventually(caller.list_tools, lambda outcome: len(outcome.tools) > 0, seconds=30)
        assert listing.tools == (f"{alias}-shallow",), listing.raw
        assert _worker(pinned) == worker
        hooks_rig.sunk()
        assert caller.call(f"{alias}-shallow", {"record": "r-1"}).text == "found"
        scanned: Final = _only(_generic(hooks_rig.sunk(), "request"))
        assert scanned["texts"] == ["deepest leaf", "r-1"], scanned
        unpublished: Final = caller.call(f"{alias}-deep", {"record": "r-2"})
        assert _worker(pinned) == worker
        assert unpublished.error is not None and "Tool 'deep' not found" in unpublished.raw, unpublished.raw
        reached: Final = tool_calls(peer.drain())
        assert [object_value(JSON_OBJECT.validate_python(call["body"])["params"])["name"] for call in reached] == [
            "shallow"
        ]
        relisted: Final = _generic(hooks_rig.sunk(), "request")
        assert all((scan["mcp_tool_name"], scan["litellm_call_id"]) == ("shallow", None) for scan in relisted), relisted
        rows: Final = _rows(key, 3)
        assert [(row["call_type"], row["status"]) for row in rows] == [
            ("list_mcp_tools", "success"),
            ("call_mcp_tool", "success"),
            ("call_mcp_tool", "failure"),
        ], rows
        assert [_tool_metadata(row)["name"] for row in rows[1:]] == ["shallow", "deep"], rows
        failed: Final = object_value(JSON_OBJECT.validate_python(rows[2]["metadata"])["error_information"])
        assert failed["error_message"] == "404: Tool 'deep' not found", failed


def test_an_adapter_returning_the_wrong_number_of_texts_fails_closed_before_the_peer(hooks_rig: HooksRig) -> None:
    with (
        scripted_peer(_lookup_tool()) as peer,
        _pinned(hooks_rig.gateway) as (pinned, worker),
        pinned.scenario() as scenario,
    ):
        alias: Final = "count" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        caller: Final = McpCaller(pinned, key, "mcp", headers={"x-mcp-servers": alias})
        assert eventually(caller.initialize, lambda outcome: outcome.ok, seconds=30).ok
        name: Final = next(
            tool for tool in eventually(caller.list_tools, _has_lookup, seconds=30).tools if "lookup" in tool
        )
        assert _worker(pinned) == worker
        hooks_rig.sunk()
        blocked: Final = caller.call(name, {"record": COUNT_MISMATCH})
        assert _worker(pinned) == worker
        assert blocked.error is not None, blocked.raw
        assert (
            "guardrail returned 4 texts for 3 MCP tool strings, so the redaction cannot be mapped back" in blocked.raw
        )
        assert tool_calls(peer.drain()) == (), "the blocked call reached the peer"
        sunk: Final = hooks_rig.sunk()
        assert _only(_generic(sunk, "request"))["texts"] == [LOOKUP_DESCRIPTION, "record identifier", COUNT_MISMATCH]
        assert _native(sunk, "post") == (), sunk
        rows: Final = _rows(key, 2)
        assert [(row["call_type"], row["status"]) for row in rows] == [
            ("list_mcp_tools", "success"),
            ("call_mcp_tool", "failure"),
        ], rows
        assert _tool_metadata(rows[1])["arguments"] == {"record": COUNT_MISMATCH}, rows[1]

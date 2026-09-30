"""pre_mcp_call guardrails are handed the tool entry ``tools/list`` served to the caller.

One owned proxy carries a default-on ``custom_code`` pre_mcp_call guardrail. At listing time it masks
``SECRET`` out of every scanned text. At call time, when an argument carries the probe marker, it
blocks and echoes the description and parameters it was handed, which is the only way to observe from outside
what metadata the gateway attached to the hook
"""

import json
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, gateway_from_environment
from integration._support.mcp import (
    EntryPoint,
    McpCaller,
    ScriptedTool,
    listed_tools,
    openapi_peer,
    register_mcp,
    scripted_peer,
    text_result,
)
from integration._support.process import owned_proxy

_ECHO: Final = "catalog-echo:"
_PROBE: Final = "catalog-probe"
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
    path: Final = directory / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with gateway_from_environment() as gateway, owned_proxy(gateway, directory, {}, config=path) as candidate:
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

import re
import textwrap
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import pytest
from integration._support.client import Gateway, gateway_from_environment
from integration._support.mcp import McpCaller, McpPeer, Outcome, echo_tool, register_mcp, scripted_peer, tool_calls
from integration._support.process import owned_proxy

_PRE_ECHO: Final = "pre-saw"
_PRE_PROBE: Final = "pre-probe"
_HIDDEN_TOOL: Final = "list_markets"
_HIDDEN_ECHO: Final = "catalog-hidden"
_DURING_ECHO: Final = "during-saw"
_LLM_ECHO: Final = "llm-saw"
_LLM_PROBE: Final = "llm-probe"
_DURING_PROBE: Final = "during-probe"
_BLOCKED_ALIAS: Final = "srvb" + uuid.uuid4().hex[:8]
_PRE_GUARDRAIL_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    f'    if request_data.get("mcp_server_name") != "{_BLOCKED_ALIAS}":\n'
    "        return allow()\n"
    '    url = request_data.get("mcp_server_url")\n'
    '    tool = inputs.get("tools", [{}])[0].get("function", {}).get("name")\n'
    f'    if tool == "{_HIDDEN_TOOL}":\n'
    '        origin_only = url is not None and "@" not in url and "?" not in url and url.count("/") == 2\n'
    f'        return block("{_HIDDEN_ECHO}") if origin_only else allow()\n'
    f'    if "{_PRE_PROBE}" not in list(inputs.get("texts") or []):\n'
    "        return allow()\n"
    f'    return block("{_PRE_ECHO}[" + str(request_data.get("mcp_server_name")) + "|" + str(url) + "]")\n'
)
_DURING_GUARDRAIL_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    f'    if "{_DURING_PROBE}" not in list(inputs.get("texts") or []):\n'
    "        return allow()\n"
    f'    return block("{_DURING_ECHO}[" + str(request_data.get("mcp_server_name")) + "|"'
    ' + str(request_data.get("mcp_server_url")) + "]")\n'
)
_LLM_GUARDRAIL_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    f'    if "{_LLM_PROBE}" not in list(inputs.get("texts") or []):\n'
    "        return allow()\n"
    f'    return block("{_LLM_ECHO}[" + str(request_data.get("mcp_server_name")) + "|"'
    ' + str(request_data.get("mcp_server_url")) + "]")\n'
)
_GUARDRAILS_YAML: Final = (
    "guardrails:\n"
    "  - guardrail_name: server-identity-pre\n"
    "    litellm_params:\n"
    "      guardrail: custom_code\n"
    "      mode: pre_mcp_call\n"
    "      default_on: true\n"
    "      custom_code: |\n"
    + textwrap.indent(_PRE_GUARDRAIL_CODE, 8 * " ")
    + "  - guardrail_name: server-identity-during\n"
    "    litellm_params:\n"
    "      guardrail: custom_code\n"
    "      mode: during_mcp_call\n"
    "      default_on: true\n"
    "      custom_code: |\n"
    + textwrap.indent(_DURING_GUARDRAIL_CODE, 8 * " ")
    + "  - guardrail_name: server-identity-llm\n"
    "    litellm_params:\n"
    "      guardrail: custom_code\n"
    "      mode: pre_call\n"
    "      default_on: true\n"
    "      custom_code: |\n" + textwrap.indent(_LLM_GUARDRAIL_CODE, 8 * " ")
)


@pytest.fixture(scope="module")
def identity_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("server-identity")
    path: Final = directory / "server_identity.yaml"
    path.write_text((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text() + _GUARDRAILS_YAML)
    with gateway_from_environment() as gateway, owned_proxy(gateway, directory, {}, config=path, workers=2) as rig:
        yield rig


@dataclass(frozen=True, slots=True)
class TwinServers:
    gateway: Gateway
    peer: McpPeer
    allowed_alias: str
    blocked_alias: str
    origin: str
    key: str


def _dirty_url(peer: McpPeer) -> str:
    parts: Final = urlsplit(peer.url)
    return f"{parts.scheme}://admin:hunter2@{parts.netloc}{parts.path}?token=sk-upstream-secret"


def _echoed(outcome: Outcome, marker: str) -> str:
    found: Final = re.search(rf"{marker}\[(.*?)\]", outcome.raw)
    assert found is not None, outcome.raw
    return found.group(1)


@pytest.fixture
def twins(identity_rig: Gateway) -> Iterator[TwinServers]:
    with scripted_peer(echo_tool("get_ticker"), echo_tool(_HIDDEN_TOOL)) as peer, identity_rig.scenario() as scenario:
        allowed_alias: Final = "srva" + uuid.uuid4().hex[:8]
        allowed: Final = register_mcp(scenario, peer, allowed_alias)
        blocked: Final = register_mcp(
            scenario,
            peer,
            _BLOCKED_ALIAS,
            url=_dirty_url(peer),
            auth_type="api_key",
            credentials={"auth_value": "synthetic-upstream-key"},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [allowed, blocked]})
        parts: Final = urlsplit(peer.url)
        yield TwinServers(identity_rig, peer, allowed_alias, _BLOCKED_ALIAS, f"{parts.scheme}://{parts.netloc}", key)


@pytest.mark.parametrize("entry", ("mcp", "server_mcp"))
def test_pre_mcp_call_custom_code_sees_the_server_the_tool_routes_to(twins: TwinServers, entry: str) -> None:
    allowed_caller: Final = McpCaller(twins.gateway, twins.key, entry, twins.allowed_alias)
    blocked_caller: Final = McpCaller(twins.gateway, twins.key, entry, twins.blocked_alias)
    twins.peer.drain()
    allowed: Final = allowed_caller.call(f"{twins.allowed_alias}-get_ticker", {"symbol": _PRE_PROBE})
    assert allowed.ok and allowed.text == f'{{"symbol": "{_PRE_PROBE}"}}', allowed.raw
    assert [call["body"]["params"]["name"] for call in tool_calls(twins.peer.drain())] == ["get_ticker"]
    blocked: Final = blocked_caller.call(f"{twins.blocked_alias}-get_ticker", {"symbol": _PRE_PROBE})
    assert blocked.error is not None, f"the same tool through {twins.blocked_alias} was not blocked: {blocked.raw}"
    assert _echoed(blocked, _PRE_ECHO) == f"{twins.blocked_alias}|{twins.origin}", blocked.raw
    assert tool_calls(twins.peer.drain()) == (), "a blocked call reached the peer"


def test_tools_list_catalog_scan_sees_the_server_name_and_origin_only_url(twins: TwinServers) -> None:
    listed: Final = McpCaller(twins.gateway, twins.key, "mcp").list_tools()
    assert listed.ok, listed.raw
    assert {f"{twins.allowed_alias}-get_ticker", f"{twins.allowed_alias}-{_HIDDEN_TOOL}"} <= set(listed.tools), (
        listed.tools
    )
    assert f"{twins.blocked_alias}-get_ticker" in listed.tools, listed.tools
    assert f"{twins.blocked_alias}-{_HIDDEN_TOOL}" not in listed.tools, (
        f"the catalog scan did not see {twins.blocked_alias} with its origin-only URL: {listed.tools}"
    )


def test_during_mcp_call_custom_code_sees_the_same_server_name_and_origin(twins: TwinServers) -> None:
    caller: Final = McpCaller(twins.gateway, twins.key, "mcp", twins.allowed_alias)
    probed: Final = caller.call(f"{twins.allowed_alias}-get_ticker", {"symbol": _DURING_PROBE})
    assert probed.error is not None, f"the during probe was not blocked: {probed.raw}"
    assert _echoed(probed, _DURING_ECHO) == f"{twins.allowed_alias}|{twins.origin}", probed.raw


def test_llm_request_body_cannot_spoof_the_mcp_server_identity(identity_rig: Gateway) -> None:
    response: Final = identity_rig.request(
        "POST",
        "/v1/chat/completions",
        {
            "model": "openai/gpt-5-mini",
            "messages": [{"role": "user", "content": _LLM_PROBE}],
            "mcp_server_name": "spoofed",
            "mcp_server_url": "https://attacker.example",
            "mcp_server_identity": {"name": "spoofed", "url": "https://attacker.example"},
        },
    )
    assert _echoed(Outcome(response.status_code, None, raw=response.text), _LLM_ECHO) == "None|None", response.text

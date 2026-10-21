"""Akto guardrail in `mode: logging_only` on MCP tool calls through a real proxy with two workers.

The scripted MCP server and the Akto `/api/http-proxy` service are the only doubles.
"""

import json
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.mcp import EntryPoint, McpCaller, McpPeer, echo_tool, register_mcp, scripted_peer, tool_calls
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server

AKTO_KEY: Final = "synthetic-akto-mcp-log-key"
BLOCK_MARK: Final = "SYNTHETIC-AKTO-BLOCK"
RESPONSE_CHECK: Final = {"akto_connector": "litellm", "response_guardrails": "true", "ingest_data": "true"}
MCP_SPEND_ROWS: Final = 'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s AND call_type = %s'


@dataclass(frozen=True, slots=True)
class AktoCall:
    path: str
    flags: dict[str, str]
    authorization: str
    payload: dict[str, object]


def _akto_call(request: Request) -> AktoCall:
    target: Final = urlsplit(request.target)
    return AktoCall(
        path=target.path,
        flags={name: values[0] for name, values in parse_qs(target.query).items()},
        authorization=request.headers.get("authorization", ""),
        payload=json.loads(request.body),
    )


def _akto_verdict(request: Request) -> Reply:
    verdict: Final = (
        {"Allowed": False, "Behaviour": "block", "Reason": "Synthetic Akto policy block"}
        if BLOCK_MARK.encode() in request.body
        else {"Allowed": True}
    )
    return Reply(body=json.dumps({"data": {"guardrailsResult": verdict}}).encode())


def _config(akto_url: str, root: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": "akto-mcp-log",
            "litellm_params": {
                "guardrail": "akto",
                "mode": "logging_only",
                "default_on": True,
                "akto_base_url": akto_url,
                "akto_api_key": AKTO_KEY,
            },
        }
    ]
    path: Final = root / "akto-mcp-logging-only.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    akto: Wire
    peer: McpPeer
    scenario: Scenario
    alias: str
    server_id: str

    def key(self) -> str:
        return self.scenario.key(object_permission={"mcp_servers": [self.server_id]})

    def settled_akto_calls(self, key: str, marker: str) -> tuple[AktoCall, ...]:
        eventually(
            lambda: read_rows(MCP_SPEND_ROWS, (sha256(key.encode()).hexdigest(), "call_mcp_tool")),
            lambda rows: len(rows) >= 1,
            seconds=70,
        )
        eventually(lambda: self.akto.received.qsize(), lambda count: count >= 1, seconds=30)
        calls: Final = tuple(_akto_call(request) for request in self.akto.drain() if marker.encode() in request.body)
        return tuple(call for call in calls if call.authorization == AKTO_KEY)


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    root: Final = tmp_path_factory.mktemp("akto-mcp-logging-only")
    alias: Final = "aktolog" + uuid.uuid4().hex[:8]
    with (
        gateway_from_environment() as gateway,
        wire_server(_akto_verdict) as akto,
        scripted_peer(echo_tool("echo")) as peer,
        owned_proxy_process(gateway, root, {}, config=_config(akto.url, root), workers=2) as owned,
        owned.gateway.scenario() as scenario,
    ):
        identity: Final = register_mcp(scenario, peer, alias)
        peer.drain()
        akto.drain()
        yield Rig(owned.gateway, akto, peer, scenario, alias, identity)


@pytest.mark.parametrize("entry", ("mcp", "rest"))
def test_logging_only_akto_block_verdict_never_blocks_an_mcp_tool_call_and_still_checks_it(
    rig: Rig, entry: EntryPoint
) -> None:
    marker: Final = "mark-" + uuid.uuid4().hex
    arguments: Final = {"note": f"{BLOCK_MARK} {marker}"}
    key: Final = rig.key()

    outcome: Final = McpCaller(rig.proxy, key, entry, rig.alias).call(
        f"{rig.alias}-echo", arguments, server_id=rig.server_id
    )

    assert (outcome.error, outcome.text) == (None, json.dumps(arguments, sort_keys=True)), outcome.raw
    upstream: Final = [call["body"]["params"]["arguments"] for call in tool_calls(rig.peer.drain())]
    assert upstream == [arguments], upstream
    calls: Final = rig.settled_akto_calls(key, marker)
    assert all(call.path == "/api/http-proxy" for call in calls), calls
    checked: Final = [call for call in calls if call.flags == RESPONSE_CHECK]
    assert [marker in str(call.payload.get("responsePayload")) for call in checked] == [True], f"{entry}: {calls}"

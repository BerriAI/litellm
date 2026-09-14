import json
import queue
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Final

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.types import Message, Receive, Scope, Send
from starlette.requests import Request

from mcp_tests.mcp_e2e_upstream_server import add, multiply
from integration._support.asgi import asgi_server
from integration._support.client import Gateway, Scenario
from integration._support.database import read_rows


@dataclass(frozen=True, slots=True)
class McpPeer:
    url: str
    calls: queue.Queue[dict[str, object]]

    def drain(self) -> tuple[dict[str, object], ...]:
        return tuple(self.calls.get_nowait() for _ in range(self.calls.qsize()))


@contextmanager
def mcp_peer() -> Iterator[McpPeer]:
    service: Final = FastMCP("integration-math", stateless_http=True, json_response=True, transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False))
    service.add_tool(add)
    service.add_tool(multiply)

    @service.tool()
    def fail() -> str:
        raise ValueError("synthetic tool failure")

    app: Final = service.streamable_http_app()
    observed: Final[queue.Queue[dict[str, object]]] = queue.Queue()

    async def capture(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        body: Final = await Request(scope, receive).body()
        assert len(body) <= 65536
        if body:
            observed.put({"body": json.loads(body), "headers": dict(scope["headers"])})
        pending = True
        async def replay() -> Message:
            nonlocal pending
            if pending:
                pending = False
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()
        await app(scope, replay, send)

    with asgi_server(capture) as url:
        yield McpPeer(url + "/mcp", observed)


def register_mcp(scenario: Scenario, peer: McpPeer, alias: str, **fields: object) -> str:
    response: Final = scenario.gateway.request("POST", "/v1/mcp/server", {"server_name": alias, "alias": alias, "url": peer.url, "transport": "http", **fields})
    identity: Final = response.json()["server_id"]
    scenario.cleanups.callback(delete_mcp, scenario.gateway, identity)
    assert response.status_code == 201, response.text
    return identity


def delete_mcp(gateway: Gateway, identity: str) -> None:
    response: Final = gateway.request("DELETE", f"/v1/mcp/server/{identity}")
    assert response.status_code == 202, response.text
    assert read_rows('SELECT server_id FROM "LiteLLM_MCPServerTable" WHERE server_id = %s', (identity,)) == []


def tool_names(gateway: Gateway, key: str, identity: str) -> dict[str, str]:
    response: Final = gateway.client.get("/mcp-rest/tools/list", headers={"x-litellm-api-key": key})
    assert response.status_code == 200, response.text
    return {name: tool["name"] for tool in response.json()["tools"] if tool.get("mcp_info", {}).get("server_id") == identity for name in ("add", "multiply", "fail") if tool["name"].endswith(name)}


def call_tool(gateway: Gateway, key: str, identity: str, name: str, arguments: dict[str, object]):
    return gateway.client.post("/mcp-rest/tools/call", headers={"x-litellm-api-key": key}, json={"server_id": identity, "name": name, "arguments": arguments})

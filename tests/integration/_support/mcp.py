import asyncio
import json
import os
import queue
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import httpx
from integration._support.asgi import asgi_server
from integration._support.client import Gateway, Scenario
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import SamplingMessage, TextContent
from mcp_tests.mcp_e2e_upstream_server import add, multiply
from pydantic import BaseModel
from sse_starlette.sse import AppStatus
from starlette.requests import Request as StarletteRequest
from starlette.types import Message, Receive, Scope, Send

Transport = Literal["http", "sse", "stdio"]
STDIO_PEER: Final = Path(__file__).with_name("mcp_stdio_peer.py")


@dataclass(frozen=True, slots=True)
class McpPeer:
    url: str
    calls: queue.Queue[dict[str, object]]
    transport: Transport = "http"
    command: str | None = None
    args: tuple[str, ...] = ()
    record: Path | None = None
    consumed: list[int] = field(default_factory=lambda: [0])

    def drain(self) -> tuple[dict[str, object], ...]:
        if self.record is not None:
            lines: Final = self.record.read_text().splitlines() if self.record.exists() else []
            fresh: Final = tuple(json.loads(line) for line in lines[self.consumed[0] :])
            self.consumed[0] = len(lines)
            return fresh
        return tuple(self.calls.get_nowait() for _ in range(self.calls.qsize()))

    def registration(self) -> dict[str, object]:
        if self.transport == "stdio":
            return {"transport": "stdio", "command": self.command, "args": list(self.args)}
        return {"transport": self.transport, "url": self.url}


class Confirmation(BaseModel):
    confirmed: bool


def math_service(name: str = "integration-math", *, rich: bool = False) -> MCPServer:
    service: Final = MCPServer(name)
    service.add_tool(add)
    service.add_tool(multiply)

    @service.tool()
    def fail() -> str:
        raise ValueError("synthetic tool failure")

    if not rich:
        return service

    @service.tool()
    async def slow(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "slept"

    @service.tool()
    async def progress(steps: int, ctx: Context) -> str:
        for step in range(steps):
            await ctx.report_progress(step + 1, steps, f"step {step + 1}")
        return f"{steps} steps"

    @service.tool()
    async def sample(prompt: str, ctx: Context) -> str:
        result: Final = await ctx.session.create_message(
            messages=[SamplingMessage(role="user", content=TextContent(type="text", text=prompt))],
            max_tokens=32,
        )
        return "sampled:" + (result.content.text if isinstance(result.content, TextContent) else "")

    @service.tool()
    async def elicit(question: str, ctx: Context) -> str:
        result: Final = await ctx.elicit(message=question, schema=Confirmation)
        return f"elicited:{result.action}"

    @service.prompt()
    def greeting(name: str) -> str:
        return f"Hello, {name}"

    @service.resource("status://ready")
    def status() -> str:
        return "ready"

    @service.resource("greeting://{name}")
    def greeting_resource(name: str) -> str:
        return f"Hello, {name}"

    return service


def _capturing(app: Callable[[Scope, Receive, Send], object], observed: queue.Queue[dict[str, object]]):
    async def capture(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await app(scope, receive, send)
            return
        body: Final = await StarletteRequest(scope, receive).body()
        assert len(body) <= 65536
        if body:
            observed.put({"body": json.loads(body), "headers": dict(scope["headers"]), "path": scope["path"]})
        message: Final = {"type": "http.request", "body": body, "more_body": False}
        pending: Final = iter((message,))

        async def replay() -> Message:
            buffered: Final = next(pending, None)
            if buffered is not None:
                return buffered
            return await receive()

        await app(scope, replay, send)

    return capture


def _draining_sse_watcher(app: Callable[[Scope, Receive, Send], object]):
    """sse_starlette parks a per-loop watcher that only stops once AppStatus.should_exit flips."""

    async def lifespan(scope: Scope, receive: Receive, send: Send) -> None:
        while True:
            message: Final = await receive()
            if message["type"] == "lifespan.startup":
                AppStatus.should_exit = False
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                AppStatus.should_exit = True
                watchers: Final = tuple(
                    task for task in asyncio.all_tasks() if "_shutdown_watcher" in repr(task.get_coro())
                )
                await asyncio.gather(*watchers)
                await send({"type": "lifespan.shutdown.complete"})
                return

    async def wrapped(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await lifespan(scope, receive, send)
            return
        await app(scope, receive, send)

    return wrapped


@contextmanager
def mcp_peer(transport: Literal["http", "sse"] = "http", *, rich: bool = False) -> Iterator[McpPeer]:
    service: Final = math_service(rich=rich)
    security: Final = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app: Final = (
        _draining_sse_watcher(service.sse_app(transport_security=security))
        if transport == "sse"
        else service.streamable_http_app(stateless_http=True, json_response=True, transport_security=security)
    )
    observed: Final[queue.Queue[dict[str, object]]] = queue.Queue()
    with asgi_server(_capturing(app, observed)) as url:
        yield McpPeer(url + ("/sse" if transport == "sse" else "/mcp"), observed, transport)


@contextmanager
def stdio_peer(directory: Path, *, rich: bool = False) -> Iterator[McpPeer]:
    record: Final = directory / f"stdio-{os.getpid()}-{time.monotonic_ns()}.jsonl"
    yield McpPeer(
        "",
        queue.Queue(),
        "stdio",
        sys.executable,
        (str(STDIO_PEER), str(record), "rich" if rich else "plain"),
        record,
    )


JsonRpc = Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ScriptedTool:
    name: str
    respond: Callable[[JsonRpc], Reply | JsonRpc]


def jsonrpc_reply(identity: object, result: JsonRpc) -> Reply:
    return Reply(body=json.dumps({"jsonrpc": "2.0", "id": identity, "result": result}).encode())


def jsonrpc_error(identity: object, code: int, message: str) -> Reply:
    return Reply(
        body=json.dumps({"jsonrpc": "2.0", "id": identity, "error": {"code": code, "message": message}}).encode()
    )


@contextmanager
def scripted_peer(*tools: ScriptedTool) -> Iterator[McpPeer]:
    """Raw JSON-RPC peer for shapes the SDK server cannot produce: half-written bodies, stalls, wire errors."""
    observed: Final[queue.Queue[dict[str, object]]] = queue.Queue()
    by_name: Final = {tool.name: tool for tool in tools}

    def provider(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=405)
        body: Final = json.loads(request.body)
        observed.put({"body": body, "headers": dict(request.headers), "path": request.target})
        if "id" not in body:
            return Reply(status=202)
        identity: Final = body["id"]
        method: Final = body["method"]
        if method == "initialize":
            return jsonrpc_reply(
                identity,
                {
                    "protocolVersion": body["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "integration-scripted-peer", "version": "1"},
                },
            )
        if method == "tools/list":
            return jsonrpc_reply(
                identity, {"tools": [{"name": name, "inputSchema": {"type": "object"}} for name in by_name]}
            )
        if method != "tools/call":
            return jsonrpc_error(identity, -32601, f"unsupported method {method}")
        tool: Final = by_name.get(body["params"]["name"])
        if tool is None:
            return jsonrpc_error(identity, -32602, "unknown tool")
        produced: Final = tool.respond(body["params"])
        return produced if isinstance(produced, Reply) else jsonrpc_reply(identity, produced)

    with wire_server(provider) as wire:
        yield McpPeer(wire.url + "/mcp", observed)


def text_result(text: str) -> JsonRpc:
    return {"content": [{"type": "text", "text": text}], "isError": False}


def slow_tool(name: str, seconds: float) -> ScriptedTool:
    def respond(params: JsonRpc) -> JsonRpc:
        time.sleep(seconds)
        return text_result("slept")

    return ScriptedTool(name, respond)


def disconnecting_tool(name: str) -> ScriptedTool:
    return ScriptedTool(name, lambda params: Reply(chunks=(b'{"jsonrpc":"2.0",', b'"id":1}'), abort_after=1))


def echo_tool(name: str) -> ScriptedTool:
    return ScriptedTool(name, lambda params: text_result(json.dumps(params.get("arguments", {}), sort_keys=True)))


@contextmanager
def openapi_peer() -> Iterator[tuple[McpPeer, Path]]:
    """OpenAPI-described HTTP service plus the spec file the proxy turns into MCP tools."""
    observed: Final[queue.Queue[dict[str, object]]] = queue.Queue()

    def provider(request: Request) -> Reply:
        observed.put(
            {
                "body": json.loads(request.body) if request.body else None,
                "headers": dict(request.headers),
                "path": request.target,
                "method": request.method,
            }
        )
        if request.target.startswith("/pets/") and request.method == "GET":
            return Reply(body=json.dumps({"id": request.target.rsplit("/", 1)[1], "name": "integration-pet"}).encode())
        if request.target == "/pets" and request.method == "POST":
            return Reply(status=201, body=json.dumps({"created": json.loads(request.body)}).encode())
        return Reply(status=404, body=b'{"error":"synthetic not found"}')

    with wire_server(provider) as wire:
        spec: Final = {
            "openapi": "3.0.0",
            "info": {"title": "integration pets", "version": "1"},
            "servers": [{"url": wire.url}],
            "paths": {
                "/pets/{petId}": {
                    "get": {
                        "operationId": "getPet",
                        "summary": "Fetch one pet",
                        "parameters": [{"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}}],
                        "responses": {"200": {"description": "pet"}},
                    }
                },
                "/pets": {
                    "post": {
                        "operationId": "createPet",
                        "summary": "Create a pet",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {"name": {"type": "string"}},
                                        "required": ["name"],
                                    }
                                }
                            },
                        },
                        "responses": {"201": {"description": "created"}},
                    }
                },
            },
        }
        yield McpPeer(wire.url, observed), _spec_file(spec)


def _spec_file(spec: JsonRpc) -> Path:
    path: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", "/tmp")) / f"openapi-{time.monotonic_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec))
    return path


def register_mcp(scenario: Scenario, peer: McpPeer, alias: str, **fields: object) -> str:
    response: Final = scenario.gateway.request(
        "POST", "/v1/mcp/server", {"server_name": alias, "alias": alias, **peer.registration(), **fields}
    )
    identity: Final = response.json()["server_id"]
    scenario.cleanups.callback(delete_mcp, scenario.gateway, identity)
    assert response.status_code == 201, response.text
    return identity


def delete_mcp(gateway: Gateway, identity: str) -> None:
    response: Final = gateway.request("DELETE", f"/v1/mcp/server/{identity}")
    assert response.status_code == 202, response.text
    assert read_rows('SELECT server_id FROM "LiteLLM_MCPServerTable" WHERE server_id = %s', (identity,)) == []


def listed_tools(gateway: Gateway, key: str, identity: str | None = None) -> dict[str, dict[str, object]]:
    response: Final = gateway.client.get(
        "/mcp-rest/tools/list",
        headers={"x-litellm-api-key": key},
        params={"server_id": identity} if identity else None,
    )
    assert response.status_code == 200, response.text
    return {
        tool["name"]: tool
        for tool in response.json()["tools"]
        if identity is None or tool.get("mcp_info", {}).get("server_id") == identity
    }


def tool_names(gateway: Gateway, key: str, identity: str) -> dict[str, str]:
    return {
        name: full
        for full in listed_tools(gateway, key, identity)
        for name in ("add", "multiply", "fail", "slow", "progress", "sample", "elicit")
        if full.endswith(name)
    }


def call_tool(gateway: Gateway, key: str, identity: str, name: str, arguments: dict[str, object]) -> httpx.Response:
    return gateway.client.post(
        "/mcp-rest/tools/call",
        headers={"x-litellm-api-key": key},
        json={"server_id": identity, "name": name, "arguments": arguments},
    )


def tool_calls(observed: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        item for item in observed if isinstance(item.get("body"), dict) and item["body"].get("method") == "tools/call"
    )

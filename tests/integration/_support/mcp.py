import asyncio
import json
import os
import queue
import sys
import time
from collections.abc import Callable, Generator, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import httpx
from integration._support.asgi import asgi_server
from integration._support.client import Gateway, Scenario
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server
from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import SamplingMessage, TextContent
from mcp_tests.mcp_e2e_upstream_server import add, multiply
from pydantic import BaseModel
from sse_starlette.sse import AppStatus
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response
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
    spec_path: Path | None = None
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
        if self.spec_path is not None:
            return {"transport": "http", "url": self.url, "spec_path": str(self.spec_path)}
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
        if scope["method"] == "GET" and scope["path"].endswith("/mcp"):
            await Response(status_code=405, headers={"Allow": "POST, DELETE"})(scope, receive, send)
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


def _drain_sse_streams() -> None:
    AppStatus.should_exit = True


def _draining_sse_watcher(app: Callable[[Scope, Receive, Send], object]):
    """sse_starlette parks a per-loop watcher that only stops once AppStatus.should_exit flips."""

    async def lifespan(scope: Scope, receive: Receive, send: Send) -> None:
        while True:
            message: Final = await receive()
            if message["type"] == "lifespan.startup":
                AppStatus.should_exit = False
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                _drain_sse_streams()
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
        starts: Final = [0]

        async def send_once(message: Message) -> None:
            if message["type"] == "http.response.start":
                starts[0] += 1
                if starts[0] == 2:
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
            if starts[0] > 1:
                return
            await send(message)

        await app(scope, receive, send_once)

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
    with asgi_server(_capturing(app, observed), before_stop=_drain_sse_streams if transport == "sse" else None) as url:
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
def openapi_peer() -> Iterator[McpPeer]:
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
        yield McpPeer(wire.url, observed, spec_path=_spec_file(spec))


def scratch_directory() -> Path:
    path: Final = Path(os.environ.get("INTEGRATION_RESULTS_DIR", "/tmp")) / "mcp-peers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _spec_file(spec: JsonRpc) -> Path:
    path: Final = scratch_directory() / f"openapi-{time.monotonic_ns()}.json"
    path.write_text(json.dumps(spec))
    return path


PeerKind = Literal["http", "sse", "stdio", "openapi"]
PEER_KINDS: Final[tuple[PeerKind, ...]] = ("http", "sse", "stdio", "openapi")


@contextmanager
def peer_of(kind: PeerKind, *, rich: bool = False) -> Iterator[McpPeer]:
    if kind == "openapi":
        with openapi_peer() as candidate:
            yield candidate
    elif kind == "stdio":
        with stdio_peer(scratch_directory(), rich=rich) as candidate:
            yield candidate
    else:
        with mcp_peer(kind, rich=rich) as candidate:
            yield candidate


@contextmanager
def stateful_mcp_peer() -> Generator[McpPeer]:
    service: Final = MCPServer("integration-stateful")
    selected: Final[dict[str, str]] = {}  # mutable-ok: per-session state the upstream keeps across tool calls

    def upstream_session(ctx: Context) -> str:
        assert ctx.headers is not None, "stateful upstream requires HTTP request headers"
        return ctx.headers["mcp-session-id"]

    @service.tool()
    def select_project(name: str, ctx: Context) -> str:
        selected[upstream_session(ctx)] = name
        return f"selected {name}"

    @service.tool()
    def create_feature(title: str, ctx: Context) -> str:
        project: Final = selected.get(upstream_session(ctx))
        if project is None:
            raise ValueError("no project selected in this session")
        return f"{project}/{title}"

    app: Final = service.streamable_http_app(
        stateless_http=False,
        json_response=True,
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )
    observed: Final[queue.Queue[dict[str, object]]] = queue.Queue()

    async def capture(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] == "GET":
            await Response(status_code=405)(scope, receive, send)
            return
        if scope["type"] != "http" or scope["method"] != "POST":
            await app(scope, receive, send)
            return
        body: Final = await StarletteRequest(scope, receive).body()
        observed.put({"body": json.loads(body) if body else None, "headers": dict(scope["headers"])})
        message: Final[Message] = {"type": "http.request", "body": body, "more_body": False}
        pending: Final = iter((message,))

        async def replay() -> Message:
            return next(pending, {"type": "http.disconnect"})

        await app(scope, replay, send)

    with asgi_server(capture) as url:
        yield McpPeer(url + "/mcp", observed)


def register_mcp(scenario: Scenario, peer: McpPeer, alias: str, **fields: object) -> str:
    response: Final = scenario.gateway.request(
        "POST", "/v1/mcp/server", {"server_name": alias, "alias": alias, **peer.registration(), **fields}
    )
    identity: Final = response.json()["server_id"]
    scenario.cleanups.callback(forget_mcp, scenario.gateway, identity)
    assert response.status_code == 201, response.text
    return identity


def forget_mcp(gateway: Gateway, identity: str) -> None:
    response: Final = gateway.request("DELETE", f"/v1/mcp/server/{identity}")
    assert response.status_code in (202, 404), response.text


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


EntryPoint = Literal["mcp", "server_mcp", "root", "sse", "rest"]
ENTRY_POINTS: Final[tuple[EntryPoint, ...]] = ("mcp", "server_mcp", "root", "sse", "rest")
INITIALIZE: Final = {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "integration", "version": "1"},
}


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a caller saw from one MCP operation, normalised across entry points."""

    status: int
    error: str | None
    tools: tuple[str, ...] = ()
    text: str | None = None
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None


def _parse_rpc_body(response: httpx.Response) -> Mapping[str, object] | None:
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        data: Final = tuple(line[5:].strip() for line in response.text.splitlines() if line.startswith("data:"))
        return json.loads(data[-1]) if data else None
    try:
        return json.loads(response.text)
    except ValueError:
        return None


def _outcome_from_rpc(response: httpx.Response) -> Outcome:
    body: Final = _parse_rpc_body(response)
    if response.status_code != 200 or body is None:
        return Outcome(response.status_code, response.text or f"HTTP {response.status_code}", raw=response.text)
    if "error" in body:
        return Outcome(response.status_code, json.dumps(body["error"]), raw=response.text)
    result: Final = body.get("result", {})
    assert isinstance(result, dict)
    if "tools" in result:
        return Outcome(200, None, tuple(tool["name"] for tool in result["tools"]), raw=response.text)
    content: Final = result.get("content", [])
    text: Final = content[0].get("text") if content else None
    if result.get("isError"):
        return Outcome(200, text or "isError", text=text, raw=response.text)
    return Outcome(200, None, text=text, raw=response.text)


def _outcome_from_rest(response: httpx.Response) -> Outcome:
    if response.status_code != 200:
        return Outcome(response.status_code, response.text, raw=response.text)
    body: Final = response.json()
    if "tools" in body:
        return Outcome(200, None, tuple(tool["name"] for tool in body["tools"]), raw=response.text)
    content: Final = body.get("content", [])
    text: Final = content[0].get("text") if content else None
    if body.get("isError"):
        return Outcome(200, text or "isError", text=text, raw=response.text)
    return Outcome(200, None, text=text, raw=response.text)


@dataclass(frozen=True, slots=True)
class McpCaller:
    """One caller's view of the gateway through a specific entry point."""

    gateway: Gateway
    key: str | None
    entry: EntryPoint
    alias: str | None = None
    headers: Mapping[str, str] = field(default_factory=dict)

    def _path(self) -> str:
        if self.entry == "server_mcp":
            assert self.alias is not None
            return f"/{self.alias}/mcp"
        return {"mcp": "/mcp", "root": "/mcp/", "sse": "/mcp/sse", "rest": "/mcp-rest"}[self.entry]

    def _headers(self) -> dict[str, str]:
        return {
            **({"x-litellm-api-key": self.key} if self.key is not None else {}),
            "Accept": "application/json, text/event-stream",
            **self.headers,
        }

    def rpc(self, method: str, params: JsonRpc | None = None) -> httpx.Response:
        if self.entry == "sse":
            return _legacy_sse_rpc(self.gateway, self._headers(), method, params)
        return self.gateway.client.post(
            self._path(),
            json={"jsonrpc": "2.0", "id": 1, "method": method, "params": dict(params or {})},
            headers=self._headers(),
        )

    def initialize(self) -> Outcome:
        if self.entry == "rest":
            return Outcome(200, None)
        return _outcome_from_rpc(self.rpc("initialize", INITIALIZE))

    def list_tools(self, server_id: str | None = None) -> Outcome:
        if self.entry == "rest":
            return _outcome_from_rest(
                self.gateway.client.get(
                    "/mcp-rest/tools/list",
                    headers=self._headers(),
                    params={"server_id": server_id} if server_id else None,
                )
            )
        return _outcome_from_rpc(self.rpc("tools/list"))

    def call(self, name: str, arguments: JsonRpc, server_id: str | None = None) -> Outcome:
        if self.entry == "rest":
            return _outcome_from_rest(
                self.gateway.client.post(
                    "/mcp-rest/tools/call",
                    headers=self._headers(),
                    json={
                        "name": name,
                        "arguments": dict(arguments),
                        **({"server_id": server_id} if server_id else {}),
                    },
                )
            )
        return _outcome_from_rpc(self.rpc("tools/call", {"name": name, "arguments": dict(arguments)}))


def _legacy_sse_rpc(
    gateway: Gateway, headers: Mapping[str, str], method: str, params: JsonRpc | None
) -> httpx.Response:
    """Drive the legacy GET /mcp/sse + POST /mcp/sse/messages pair for one request and synthesise a JSON response."""
    with gateway.client.stream("GET", "/mcp/sse", headers=headers, timeout=15) as stream:
        if stream.status_code != 200:
            stream.read()
            return httpx.Response(stream.status_code, text=stream.text)
        lines: Final = stream.iter_lines()
        endpoint: Final = next(line[5:].strip() for line in lines if line.startswith("data:"))
        init: Final = gateway.client.post(
            endpoint,
            json={"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": INITIALIZE},
            headers=headers,
        )
        assert init.status_code in (200, 202), init.text
        gateway.client.post(endpoint, json={"jsonrpc": "2.0", "method": "notifications/initialized"}, headers=headers)
        posted: Final = gateway.client.post(
            endpoint, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": dict(params or {})}, headers=headers
        )
        if posted.status_code not in (200, 202):
            return httpx.Response(posted.status_code, text=posted.text)
        for line in lines:
            if line.startswith("data:") and '"id": 1' in line.replace('"id":1', '"id": 1'):
                return httpx.Response(200, text=line[5:].strip(), headers={"content-type": "application/json"})
    return httpx.Response(599, text="legacy SSE stream ended without a reply")


def official_client_outcomes(
    gateway: Gateway, key: str, path: str, name: str, arguments: JsonRpc, *, legacy_sse: bool = False
) -> tuple[Outcome, Outcome]:
    """List then call through the official MCP client session, returning both outcomes."""
    url: Final = str(gateway.client.base_url).rstrip("/") + path
    headers: Final = {"x-litellm-api-key": key}

    async def run() -> tuple[Outcome, Outcome]:
        transport: Final = (
            sse_client(url, headers=headers)
            if legacy_sse
            else streamable_http_client(url, http_client=httpx.AsyncClient(headers=headers, timeout=30))
        )
        async with transport as streams, ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            listed: Final = await session.list_tools()
            result: Final = await session.call_tool(name, dict(arguments))
            content: Final = result.content[0] if result.content else None
            text: Final = content.text if isinstance(content, TextContent) else None
            return (
                Outcome(200, None, tuple(tool.name for tool in listed.tools)),
                Outcome(200, (text or "isError") if result.is_error else None, text=text),
            )

    return asyncio.run(run())


def tool_calls(observed: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    return tuple(
        item for item in observed if isinstance(item.get("body"), dict) and item["body"].get("method") == "tools/call"
    )

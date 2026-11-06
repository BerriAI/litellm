import json
import queue
from collections.abc import Callable, Mapping
from typing import Final

import httpx2

COUNT_OUTPUT_SCHEMA: Final = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


def count_tool_upstream(
    structured: Mapping[str, object],
    methods_seen: "queue.SimpleQueue[str]",
    output_schema: Mapping[str, object] = COUNT_OUTPUT_SCHEMA,
) -> Callable[[httpx2.Request], httpx2.Response]:
    """An MCP upstream with one tool, `count`, whose declared output schema defaults to wanting an integer `n`."""

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST":
            return httpx2.Response(200)
        payload: Final = json.loads(request.content)
        methods_seen.put(payload["method"])
        if "id" not in payload:
            return httpx2.Response(202)
        results: Final[Mapping[str, Mapping[str, object]]] = {
            "initialize": {
                "protocolVersion": payload.get("params", {}).get("protocolVersion"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "count", "version": "1"},
            },
            "tools/list": {
                "tools": [{"name": "count", "inputSchema": {"type": "object"}, "outputSchema": output_schema}]
            },
            "tools/call": {"content": [{"type": "text", "text": "ok"}], "structuredContent": structured},
        }
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": payload["id"], "result": results.get(payload["method"], {})}
        )

    return respond


def methods_received(methods_seen: "queue.SimpleQueue[str]") -> tuple[str, ...]:
    return tuple(methods_seen.get_nowait() for _ in range(methods_seen.qsize()))

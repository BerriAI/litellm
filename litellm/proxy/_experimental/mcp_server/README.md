# MCP server: long-running tool calls

How the gateway streams progress for tools that take a while to finish, and what
the path forward looks like once the MCP Tasks extension lands.

## Today: progress notifications over the stateful transport

A host that wants progress updates includes a `progressToken` in the `_meta` of
its `tools/call` request. The gateway captures that token from the host request
context and registers a `host_progress_callback` (`forward_progress` in
`server.py`). As the upstream MCP server emits progress notifications during the
call, the callback relays each one back to the host session with
`send_progress_notification`, so the host sees progress in real time while the
call is still running.

This relies on the stateful StreamableHTTP transport. Request routing picks the
stateful session manager whenever the request carries an `mcp-session-id` header
or is an `initialize` (see the routing block in `server.py`); stateless callers
(curl, Inspector) fall through to the stateless manager and do not get a
streamed progress channel. The stateful path was restored in #26857 (LIT-2196).

The practical bound is the connection and its TTL window. Progress streams for as
long as the host keeps the connection open and the server honors it; if the
connection drops, the stream stops with it. For work that needs to outlive a
single connection, progress notifications are not the right tool, which is what
the Tasks extension is meant to solve.

## Not yet: the MCP Tasks extension

The MCP Tasks lifecycle (`tasks/get`, `tasks/update`, `tasks/cancel`, with a
`resultType: "task"` handle returned from `tools/call`) is the stateless
successor for long-running work. The gateway does not implement it yet. Tasks
was experimental in the `2025-11-25` spec and was reshaped into an extension
(SEP-2663) in the `2026-07-28` release candidate, which also removes
protocol-level sessions. Support is gated on the `mcp` Python SDK v2 line
(LiteLLM currently pins `mcp>=1.26.0,<2.0`) and on the spec finalizing. The
migration is tracked in LIT-3784.

## Code pointers

- `server.py`: `forward_progress` / `host_progress_callback` capture and relay
- `server.py`: session-id routing that selects the stateful vs stateless manager
- `call_mcp_tool` in this package threads `host_progress_callback` down to the
  upstream client call

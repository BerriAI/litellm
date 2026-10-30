# MCP code mode

Code mode lets an MCP client run JavaScript that combines tool calls and returns only the output it needs. It is off by default. Install the optional runtime only in deployments that will enable it; regular proxy installations do not include it.

```sh
pip install "litellm[proxy,mcp-code-mode]"
```

Keep your existing LiteLLM version pin when adding the extra to a deployed proxy. For containers, include the extra in the image build. Then enable it in the gateway configuration and restart. No separate sandbox service or client integration is required, and the gateway never installs packages at runtime.

```yaml
litellm_settings:
  mcp_code_mode: true
```

Reconnect the MCP client or refresh its tool list. It should discover `litellm_execute_code` alongside its existing tools. Turning it off also rejects calls from clients that cached the code tool.

The extra installs QuickJS-NG. Platforms without a compatible wheel need a source build and its compiler toolchain; source builds on these platforms have not been verified. A missing runtime produces an actionable startup error. The original `quickjs` package shares files with `quickjs-ng`; use a separate environment without that package before installing the extra. Regular proxy installations and disabled gateways do not require either runtime.

The code tool accepts a `code` string. Discover a tool and its argument schema normally, then copy the code-mode `tool_id` from its description or `search_tools` into `tools.call_tool`. Copy that reference unchanged: clients can rename the displayed tool. Exact gateway catalog names also work. Calls return the full MCP result, including `content`, `structuredContent` when provided, and `isError`. Use `text(value)` to return a JSON value to the model. For example, after discovering a tool named `example-add` that accepts `a` and `b`:

```javascript
const result = await tools.call_tool({
  tool_id: "example-add",
  arguments: { a: 2, b: 3 }
});
if (result.isError) {
  text({ error: result.content });
} else {
  text(result.structuredContent ?? result.content);
}
```

Use `await` for every call, including calls grouped with `Promise.all`. Scripts run in fresh local worker processes. They have no persistent state, general network, filesystem, package imports, or shell API. The parent gateway performs tool calls using the original caller's permissions, upstream authentication, guardrails, and accounting. A worker process isolates normal runtime failures; it is not an operating-system sandbox against an exploit of the embedded JavaScript engine.

REST clients discover the gateway code tool through `/mcp-rest/tools/list` without a `server_id` filter, and call it through `/mcp-rest/tools/call` without `server_id`. An explicit `server_id` keeps `litellm_execute_code` routed to that upstream. Per-server REST listings contain upstream tools only. Caller permissions and an optional `toolset_name` still restrict gateway code calls.

The script itself and each nested tool call count toward request-rate limits. Server request limits also count tool lookups, as they do on `/mcp/proxy`. The script holds its caller's concurrency slot until it finishes; nested calls reuse overlapping concurrency slots and retain their other limits. The wrapper has no tool charge, and each upstream call retains its configured cost.

## Tool compatibility

When code mode is enabled, tool listings include `_meta["litellm.ai/code_mode"]`; search and schema results include `code_mode`. `available` means the tool can be attempted through the JSON call bridge, not that the gateway has tested every behavior of that upstream. `unsupported` includes a reason. Descriptions also expose the call reference or the unsupported reason, for clients that omit metadata. Existing tools remain callable directly.

Task-only tools and nested orchestration are unsupported. Interactive continuations cannot be resumed inside a script. An unexpected interaction requirement returns an error instead of bypassing approval or replaying the script. Approving the outer code tool in an MCP client is not per-call approval of every nested action.

An administrator can exclude tools known to need direct interaction. Use the exact tool name from the gateway catalog:

```yaml
litellm_settings:
  mcp_code_mode:
    enabled: true
    excluded_tools:
      example-approve: "Requires interactive approval. Call this tool directly."
```

Keep large identifiers as strings. Values crossing the JavaScript bridge must be JSON-compatible, and integer values must fit JavaScript's safe integer range. Tool errors and unsupported data are not silently treated as successful results.

## Failures and cancellation

Each script has a 25-second deadline, a 64 MiB JavaScript heap limit, and a maximum of 100 tool calls with eight in flight. Each proxy process admits four scripts at once; additional requests receive a busy error. Code is limited to 128 KiB, individual bridge messages to 2 MiB, and returned output to 256 KiB. The deadline leaves room for a structured error before the existing MCP transport's 30-second response limit. Authentication or hooks can still consume that transport budget. Use direct calls for tools whose results or duration exceed the code-mode limits.

A script that exceeds a limit fails explicitly. The result includes `output`, `error`, and `calls_started`; the count records attempted tool calls, not confirmed upstream actions. When cancellation reaches the running request, it stops further dispatch and terminates the worker, but cannot undo a request already sent upstream. A client timeout or disconnect does not guarantee server-side cancellation; work can continue until the script deadline. A failed script can therefore have partial effects. Check the upstream state before retrying a write; the gateway does not automatically retry whole scripts.

# Gateway catalog pagination

Tools, prompts, resources, and resource templates retain upstream page boundaries. Follow the gateway's opaque `nextCursor` until it is absent. Each request checks the caller's current access and tool permissions. A cursor can be replayed and can be sent to another replica serving the same configuration

Set the same nonempty `LITELLM_SALT_KEY` on every replica. Pagination state uses authenticated encryption with purpose-separated HKDF keys derived only from this value. The master key is never a fallback. Complete single-page lists and direct tool calls work without a salt key; a listing that needs continuation returns an actionable configuration error instead of truncated results. Some SDKs automatically list tools when validating a tool-call response, so those SDK calls also require a salt when that listing is paginated

Cursors expire ten minutes after the first page. Continuations do not extend that deadline. Rotating the salt invalidates all outstanding cursors; clients must start a new listing. During a rolling key change, replicas with different keys cannot accept each other's cursors. Coordinate the change across the fleet

A changed registry, caller scope, or available upstream revision requires a fresh listing. When an upstream exposes a string or integer `_meta.revision`, subsequent pages must retain it. Otherwise consistency follows that upstream's own cursor guarantees. Repeated upstream cursors and the existing upstream page limit stop traversal with an explicit error

Listing failures retain per-server outcome metadata. An incomplete upstream catalog cannot establish a bare tool-name route. Complete initial pages retain legacy bare-name routing. Use the server-prefixed names returned by the gateway for paginated catalogs

For a scoped rollout, keep the previous source build serving the control pool and send only selected clients to a separate candidate pool. All candidate replicas must share configuration and salt. Verify page one on one candidate replica and continuation on another, plus a fresh listing and tool call on the control pool. Do not mirror tool calls between pools

For rollback, stop sending new requests to the candidate pool, drain its in-flight operations, and return selected clients to the control pool. Clients must discard candidate cursors and start a fresh listing when crossing versions; older gateways do not validate these cursors. Keep the registry-revision migration installed when rolling back pagination. Verify a fresh listing and a tool call after switching pools

## Elicitation

Elicitation is disabled by default. Enable it per upstream in `config.yaml`; `allow_elicitation` is currently YAML-only and is not editable through the Admin UI or database-backed server API

```yaml
mcp_servers:
  interactive:
    url: https://mcp.example.com/mcp
    transport: http
    allow_elicitation: true
    timeout: 60
```

The downstream MCP client must advertise the requested form or URL capability during initialization. Use the gateway's legacy SSE endpoint (`/mcp/sse`) for the verified interactive form and URL relay path. The current Streamable HTTP endpoint (`/mcp`) can lose initialization state before a tool call, so even a client that advertised support receives an explicit elicitation error instead of an input request. Successful interactive relay over Streamable HTTP is not currently verified. Stateless calls and LLM tool bridges with no downstream MCP client also receive explicit errors

The relay wait uses the upstream server's existing `timeout` setting, or `LITELLM_MCP_CLIENT_TIMEOUT` (60 seconds by default). The enclosing tool call also retains its existing timeout. Unsupported modes, disconnects and relay failures return errors, never a fabricated user decline. Actual user accept, decline and cancel responses are preserved; cancellation stops the relay

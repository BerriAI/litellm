# written_claude.md

Work log of AI-assisted contributions to the MCP gateway v2 rewrite. This is a
record of what was built and why; it is not authoritative guidance. The
authoritative folder map and conventions live in CLAUDE.md, and when the two
disagree CLAUDE.md wins

Branch: litellm_mcp_v2_rewrite

## Scaffold (committed: ad45a90b42)

Flat-concern skeleton under litellm/proxy/gateway/mcp/: 15 empty concern
packages (foundation, transport, sessions, authn, oauth_server, authz, catalog,
dispatch, hooks, connections, credentials, servers, openapi, management,
observability), each with a documented __init__.py, plus app.py (composition
root stub), CLAUDE.md (folder map + conventions), and .importlinter (the
downward-only layer contract). Direction (inbound -> pipeline -> outbound ->
leaves) is enforced by the import-linter contract, not by folder nesting

## S0: chassis / foundation

Goal: a thin orchestration shell over the official MCP SDK. Ships almost no
behavior; stands up an ASGI app from injected fakes, proves the seam and the
toolchain, then stops. The initialize + list_tools handshake returns an empty
tool list. Real logic begins in S1

### Toolchain added to .venv

- expression 5.6.0: the functional spine (Result, tagged_union, Block)
- basedpyright 1.39.7: the strict type gate
- import-linter 1.12.1: the layer contract runner. Pinned to the 1.x line on
  purpose; import-linter 2.x requires rich>=14.2, which conflicts with litellm's
  rich>=13.9.4,<14.0 pin. The 1.x line predates the rich dependency and still
  supports the "|" independent-sibling layer syntax
- mcp 1.26.0 and starlette 1.1.0 were already present

### Files

| File | What it contains |
|---|---|
| foundation/result.py | Re-exports Expression Ok/Error/Result; defines GatewayResult[T] pinning the error channel to GatewayError. TypeAlias + TypeVar for py3.10 compat |
| foundation/errors.py | GatewayError tagged union (db_unavailable, unauthorized, invalid_input, not_implemented) + pure transport-agnostic reason(e) -> str via exhaustive match on .tag with assert_never. Leaf; does not import result.py. No error-to-status/JSON-RPC mapping here on purpose (that is an edge concern) |
| foundation/naming.py | Pure namespace_tool / split_namespaced + SEP-986 is_valid_name |
| foundation/types.py | The single vocabulary import point: FP spine, SDK wire types (Tool, CallToolResult, ListToolsResult, TextContent), error/result vocab, and a frozen Subject seed model |
| foundation/deps.py | Frozen GatewayDeps (leaf adapters) + narrow Clock/Cache/HttpxFactory Protocols + build_test_deps() returning typed in-memory fakes (immutable FakeClock, FakeCache, no-network httpx factory) |
| foundation/__init__.py | Public surface re-export |
| app.py | build_server(deps) constructs the SDK low-level Server with skeleton list_tools -> [] and call_tool -> typed not-implemented result; build_gateway(deps) wraps it in StreamableHTTPSessionManager(stateless=True) and a Starlette app whose lifespan enters manager.run(). The endpoint is mounted with Mount("/mcp", app=manager.handle_request) because handle_request is a raw ASGI app, not a request/response handler. Zero module state |
| pyrightconfig.json | Strict, scoped to this folder; reportMatchNotExhaustive / reportAny / reportExplicitAny set to error |
| tests/test_litellm/proxy/gateway/mcp/ | test_build_gateway.py (Starlette type, independent instances, /mcp mounted, the DoD initialize+list_tools==[] handshake via the SDK in-memory client) + foundation/test_errors.py + foundation/test_naming.py |

### Key decisions

- Expression Result matching: Ok(x) and Error(e) are factory functions that both
  return one Result class, so `case Ok(v)` is not possible. Exhaustive matching
  is done on the .tag literal with assert_never in the default, which passes
  strict and makes pyright flag any newly added variant
- Server instructions are passed through the Server(instructions=...)
  constructor, which is the supported path. No monkeypatch of
  create_initialization_options
- stateless=True is the default ("stateless protocol, stateful application").
  Stateful mode (Redis session/event store) is deferred to S1/S17
- Expression's tag() and case() are typed as returning Any, which trips reportAny
  under strict. Each is wrapped in cast(...) so the declared field types stay
  honest with zero Any leakage
- prisma/redis/settings on GatewayDeps are typed as object for now, not Any. They
  are unused in S0 and will be narrowed to real Protocols when a section first
  needs them
- pyrightconfig.json is scoped to this folder so strict mode does not touch the
  rest of litellm
- Error-to-wire translation is an edge concern, not vocabulary. foundation only
  defines the semantic GatewayError and a transport-agnostic reason(e) -> str.
  The error -> JSON-RPC code mapping (for /mcp) and error -> HTTP status mapping
  (for the REST surface) are deferred to their transports (S1, S13), each a total
  match with assert_never. This keeps the leaf channel-neutral and avoids shipping
  speculative, mislayered code in S0. reason() is what keeps the @tagged_union +
  exhaustive match strict-mode proof live in S0, and app.py consumes it

### Verification (all green)

- basedpyright strict: 0 errors, 0 warnings, 0 notes
- ruff (PLR0915, PLR0912, C901, PLR0911, PLC0415, BLE001): passed
- import-linter: 1 kept, 0 broken
- pytest (tests/test_litellm/proxy/gateway/mcp): 11 passed

Commands (run from repo root):

```
.venv/bin/basedpyright --project litellm/proxy/gateway/mcp/pyrightconfig.json
.venv/bin/ruff check --select PLR0915,PLR0912,C901,PLR0911,PLC0415,BLE001 litellm/proxy/gateway/mcp
.venv/bin/lint-imports --config litellm/proxy/gateway/mcp/.importlinter
.venv/bin/python -m pytest tests/test_litellm/proxy/gateway/mcp -q
```

The one genuine risk for the FP approach is retired here: Expression
@tagged_union plus exhaustive match passes basedpyright strict with no Any, no
type: ignore, and no weakened config

Live socket check: running build_gateway(build_test_deps()) under uvicorn and
connecting with the SDK streamable-http client over a real socket returns
serverInfo litellm-mcp-gateway 2.0.0, the configured instructions, and an empty
tools list. This caught a bug the in-memory handshake test could not see: the
endpoint was first mounted with Route("/mcp", manager.handle_request), but
handle_request is a raw ASGI app (scope, receive, send), not a request/response
endpoint, so Starlette called it with one argument and every request 500ed. The
in-memory test talks to the inner Server directly and never exercises Starlette
routing, so it passed regardless. Fixed by mounting with Mount, and added
test_initialize_handshake_over_real_asgi_transport which drives an initialize
through the real Starlette -> Mount -> manager path (it fails on the Route
mistake and passes on Mount). Mount serves the endpoint at /mcp/ and 307s /mcp
to it; the SDK client follows the redirect. Exact-path routing for /mcp and the
/{tenant}/{server}/mcp patterns is S1's TransportGateway concern

### Known issues / follow-ups

1. (resolved) naming.py previously filed validation failures under
   not_implemented (which means "code path not wired yet"). Added an
   invalid_input arm to GatewayError and pointed naming.py at it; tests assert
   the invalid_input arm specifically
2. prisma/redis/settings are typed as object on GatewayDeps; narrow them to
   Protocols when first used
3. Error-to-wire mappings are deferred to their transports, each a total match
   on GatewayError with assert_never: to_jsonrpc_error at the /mcp edge (S1) and
   to_http_status at the REST edge (S13). foundation stays channel-neutral
4. CI wiring is not done yet; the four gates run locally only. The semgrep
   mcp_v2 rule set called for in the S0 scope (ban module-level service
   construction outside app.py, ban raise across a seam, ban SDK monkeypatch,
   etc.) and the LOC/method-cap script are not yet authored
5. The scaffold commit (ad45a90b42) carried a Co-Authored-By trailer that the
   repo CLAUDE.md forbids; pending an amend to strip it

### Research grounding

- Local design docs: /Users/tin/Documents/mcpv2_S0_scope.md, mcpv2_sections.md,
  MCP_GATEWAY_V2.md
- Notion: "Building V2", "v2 - Goals & Requirements", "Phase A - Requirements"
- Competitor patterns reviewed: agentgateway (virtual MCP, per-server tool
  namespacing, fail-closed federation), IBM ContextForge (virtual servers,
  team/RBAC multi-tenancy, Redis-backed federation), Cloudflare (stateless
  createMcpHandler vs stateful McpAgent Durable Object, OAuth-issued tokens)

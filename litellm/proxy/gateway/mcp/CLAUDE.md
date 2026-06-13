# MCP Gateway v2 — agent & contributor guide

A from-scratch rebuild of the MCP gateway. **Tests are the spec; v1 parity is
not a goal.** Build in dependency order: S0 chassis → aggregator (S1–S4) →
auth/RBAC (S5–S6) → credentials/OAuth (S8) → the rest.

Goals & requirements: see the Notion **v2 — Goals & Requirements** page.

---

## How this is organized

**Flat concern layout.** One folder per gateway concern — no `inbound/`/`outbound/`
nesting. The request **direction** (inbound → pipeline → outbound → leaves) is
enforced by the layer contract in `.importlinter`, not by directory nesting.

A request flows top-to-bottom through the layers below; **imports only ever
point downward**. A PR that imports upward fails CI.

```
transport | oauth_server | management   ← edges (nothing internal imports these)
sessions
catalog | dispatch                      ← the pipeline (read path ⊥ execute path)
hooks | authz
connections
credentials
authn | servers
foundation | observability              ← leaves (no internal deps)
```

Modules joined by `|` are **independent siblings** and may not import each other.

---

## What goes in each folder

| Folder | Section | Responsibility |
|---|---|---|
| `foundation/` | S0 | Frozen types, `Result`/errors, `GatewayDeps`, `naming`. No I/O, no logic. Leaf. |
| `transport/` | S1 | Terminate client connections (`/mcp`, `/sse`), parse scope → `(tenant, server, bearer)`, dispatch. Routes stay thin. |
| `sessions/` | S1/S17 | Session lifecycle: one `StatefulSessionRegistry` (one lock, one invariant), owner-binding, idle-TTL reaper. |
| `authn/` | S6 | WHO is the caller. Resolve vk/JWT/`x-mcp-auth`/bearer/SSO → `Subject`; the single `auth_context_var` accessor. |
| `oauth_server/` | S6 | Gateway-as-Authorization-Server surface: `/.well-known/*`, `/authorize`, `/token`, `/callback`, `/register` (DCR), the consent card page. |
| `authz/` | S5 | What the caller MAY do. The single pure `enforce(auth, action) -> Result` predicate; frozen `Role`/`Toolset`/`Grant`; `org→team→key` resolution. |
| `catalog/` | S3 | Aggregation & namespacing (read path / fan-in). Fan out `*/list`, merge, namespace `{alias}__{tool}`, origin-tag, `list_visible(auth)`. |
| `dispatch/` | S4 | The ONE `tools/call` path (execute path). `ToolCaller`; param gating. No second dispatcher. |
| `hooks/` | S9 | Pre/post pipeline: arg validation, PII/secret redaction, response transforms, guardrails. |
| `connections/` | S2/S11 | Upstream connection pool. Long-lived `ClientSession` per `(tenant, server, auth-key)`; `UpstreamClient.call -> UpstreamCall`. |
| `credentials/` | S8 | Outbound auth material (~90% of bugs). Mode-as-judge resolver, single-flight rotation-safe refresh, token store, OBO/M2M/token-exchange/BYOK, outbound discovery. |
| `servers/` | S2/S14 | What upstreams exist. Frozen `MCPServerSpec` (config+DB merged once), `ServerRegistry` atomic-swap + the shared resolver, toolsets/virtual servers. |
| `openapi/` | S10 | OpenAPI → MCP tools; routed through the same `ToolCaller`. |
| `management/` | — | Admin CRUD surface. |
| `observability/` | S12 | Per-call audit record, cost tracking, structured logging. Leaf. |
| `app.py` | S0 | `build_gateway(deps)` — the composition root. The single construction site. |

---

## Conventions

- **One concern per file**, ~400-line soft cap; past 600, split, don't append.
- **One verb per surface file:** each inbound surface exposes `register(app)`
  (`transport/streamable_http.py`, `oauth_server/*`, `management/api.py`);
  `app.py` calls each once.
- **Public surface only in `__init__.py`.** Cross-folder imports go through it;
  internals stay private.
- **Tests mirror this tree** under `tests/test_litellm/proxy/gateway/mcp/<folder>/`.
- **Functional core / imperative shell:** decisions are pure functions over
  frozen values; I/O lives only in edge modules (`transport`, `oauth_server`,
  `connections`, `credentials`, `servers/store`, `sessions/registry`).

### Hard rules (each is a CI gate — see project coding-practices)

- Composition over inheritance.
- Never-nester: early returns, small functions.
- Don't throw; model failures as values (`Result[T,E]` / tagged unions). No
  business-logic `raise` outside edge modules.
- No mutation: models are `frozen`; no in-place mutation of shared state.
- Dependency injection throughout; no global singletons, no module-level
  mutable state (outside a registry/cache that owns its own lock).
- Fully typed; no `Any`. **Passes `basedpyright --strict` with zero errors.**
- Tagged unions + exhaustive `match` over `isinstance` ladders.
- Standard over hand-rolled: use the official MCP SDK; follow the spec.
  Per-request upstream auth goes on the `httpx.AsyncClient`, NOT
  `streamablehttp_client(headers=…/auth=…)` (silently ignored in SDK 1.26).
- Never auto-retry `tools/call`; reactive-401 is the only retry (auth replay,
  once). Credential refresh is single-flight + bounded; no nested locks.
- Atomic-swap on reload; never mutate live state.

# Intent vs reality: gap register

Quadrant: reference. Verified on `main` at `13118948` (2026-10-15)

Every place this series found where the code's apparent intent and its actual behavior part ways, in one table. Each explanation page has the same rows in context. "Inv" links to the matching invariant in [litellm-invariants-handoff.md](/mnt/project-files/invariants/litellm-invariants-handoff.md) where one exists

Impact is a judgment call. **High** means users can see wrong behavior or a security boundary is weaker than it looks. **Medium** means it misleads contributors or makes a planned change much harder. **Low** means drift or dead code

## Layering and structure

| ID | Intent | Reality | Evidence | Impact | Inv |
|---|---|---|---|---|---|
| G-L1 | The SDK never imports the proxy | `import litellm` loads zero proxy modules now, but 11 module-level, 29 type-only and 170 function-local imports remain | [02](../explanation/02-layers-and-dependency-direction.md) | Medium | L1 |
| G-L2 | Provider code never imports proxy, main or router | 2 module-level, 20 type-only, 30 function-local | same | Low | L2 |
| G-L3 | A CI check stops layer regressions | None on main. The circular-import job can't fail | `tests/documentation_tests/test_circular_imports.py` | Medium | L7 |
| G-L4 | Core utilities sit below integrations and providers | Core imports integrations 94 times (88 from the logging core and its registry) and providers 14 times | scan | Medium | L4 |
| G-L5 | Providers sit below `utils.py` | 116 provider modules import `litellm.utils` | scan | Medium | L4 |
| G-L6 | Settings flow down explicitly | 419 package-root names, set by `setattr` from YAML, read through bare `import litellm` in 395 modules | `__init__.py`, `proxy_server.py:6864` | Medium | |
| G-L7 | Proxy parts are separable | 1,029 function-local imports of `proxy_server` globals from 230 files | `rg "from litellm.proxy.proxy_server import"` | Medium | |
| G-L8 | No monster files | `proxy_server.py` 20,830, `router.py` 15,478, `llm_http_handler.py` 13,947, `utils.py` 10,627, `main.py` 9,016 lines | `wc -l` | Medium | T12 |
| G-L9 | `litellm/repositories` is SDK-level and proxy-free (`ARCHITECTURE.md:244`) | Function-local proxy imports remain in 5 repository files and `models/model.py` | `rg litellm.proxy litellm/repositories` | Low | L1 |

## Data-plane pipeline

| ID | Intent | Reality | Evidence | Impact | Inv |
|---|---|---|---|---|---|
| G-P1 | Every provider uses `BaseLLMHTTPHandler` | OpenAI uses the legacy SDK handler unless an env flag is set. Anthropic chat and Azure use their own. 90 `handler.py` files remain | `main.py:2577`, `:2811`, `:1220` | Medium | P3 |
| G-P2 | Async SDK path is async | `acompletion` runs sync `completion` in a thread executor | `main.py:680-686` | Medium | |
| G-P3 | Provider dispatch is data-driven | If/elif chain over about 60 providers | `main.py:5864-6072` | Low | |
| G-P4 | One routing decision | `route_request` has its own 8-step ladder before the Router, plus a pre-call fallback loop | `route_llm_request.py:599-826` | Medium | |
| G-P5 | One logging side channel | `Logging` is created in four places, and nested wrappers claim the success log to avoid double counting | `utils.py:1398`, `pass_through_endpoints.py:1234` | Medium | |
| G-P6 | Spend recording can't be skipped | Spend is a callback, so any path that skips `async_success_handler` skips spend | `proxy_server.py:2958` | High | D7 |
| G-P7 | Responses sessions live in the Responses layer | The chat bridge rebuilds history from `LiteLLM_SpendLogs` with raw SQL, so bridged sessions need the DB and force spend-log flushes | `session_handler.py:275`, `db_spend_update_writer.py:1533` | Medium | D8 |
| G-P8 | Messages for OpenAI goes through chat (implied by the adapter docs) | It goes Messages -> Responses by default | `messages/handler.py:53,93` | Low | |
| G-P9 | `ARCHITECTURE.md` describes the pipeline | Misses the Rust dispatch layer, lazy routes, budget reservation, legacy handlers. Wrong file for `get_llm_provider`, wrong cost method name, says spend queues in Redis, stale `experimental_pass_through` paths, stale job intervals | `ARCHITECTURE.md:24-260` | Medium | |

## Control plane and state

| ID | Intent | Reality | Evidence | Impact | Inv |
|---|---|---|---|---|---|
| G-C1 | Config is validated | Unknown `litellm_settings` keys silently become module attributes | `proxy_server.py:6864` | Medium | |
| G-C2 | DB config applies at boot | It applies on the first `add_deployment` tick, since `load_config` runs before Prisma connects | `proxy_server.py:1401-1455` | Medium | |
| G-C3 | UI-created guardrails apply on every pod | Only with `store_model_in_db`, otherwise only on the pod that served the POST | `proxy_server.py:8456,8531` | High | |
| G-C4 | `/model/new` works out of the box | 500 unless `store_model_in_db` | `model_management_endpoints.py:2648` | Low | |
| G-C5 | Repositories own data access | Endpoints use repositories only as raw `.table` handles. Domain methods have no callers outside the package, and `ModelRepository._encrypt_litellm_params` is unused | `proxy/utils.py:5558-5592`, `repositories/model_repository.py:41` | Medium | |
| G-C6 | Auth cache is shared across pods | Per worker unless `enable_redis_auth_cache`. Revocation relies on the invalidation pub/sub reaching every worker | `proxy_server.py:5157`, `auth_checks.py:3315` | High | |
| G-C7 | One key lookup path | `IdentityStore._resolve_key` and `get_key_object` duplicate each other | `auth/resolvers/store.py:108`, `auth_checks.py:4185` | Medium | |
| G-C8 | Per-instance state | `PrismaClient.spend_log_transactions` is class-level | `proxy/utils.py:4664` | Low | |
| G-C9 | Every route is authenticated unless on a public allowlist | 107 of 969 routes have no `user_api_key_auth` dependency. Most are public on purpose, but nothing fails when a new route forgets it | [permission-checks.md](/mnt/project-files/invariants/permission-checks.md) | High | D4 |
| G-C10 | Admin and premium checks go through one helper | 220 raw `PROXY_ADMIN` comparisons, 9 duplicate helpers, 68 premium checks in 4 spellings | handoff D5, D6 | Medium | D5, D6 |

## Extension points

| ID | Intent | Reality | Evidence | Impact | Inv |
|---|---|---|---|---|---|
| G-X1 | A registry builds callbacks by name | Construction is a long if/elif. The registry only does reverse lookups | `litellm_logging.py:4580`, `custom_logger_registry.py:67` | Low | |
| G-X2 | `CustomLogger` is a focused interface | About 60 hooks across five concerns, a third undocumented | `integrations/custom_logger.py` | Medium | |
| G-X3 | One way to write a guardrail | Unified `apply_guardrail`, native hooks and the v1 initializer all coexist | `proxy/utils.py:1905-1915` | Medium | |
| G-X4 | Callbacks never install packages | Legacy sentry path runs `pip install` at runtime | `litellm_logging.py:4482` | Medium | |

## Deployment

| ID | Intent | Reality | Evidence | Impact | Inv |
|---|---|---|---|---|---|
| G-D1 | Gateway and backend pods each serve only their plane | Lazy routes register after the startup trim, so both pods serve any lazy route | `_lazy_features.py:498`, `gateway/main.py:61-73` | High | |
| G-D2 | Every route belongs to one plane | About 30 lazy prefixes in neither allowlist, 8 paths in both | the two `allowlist.py` files | Medium | |
| G-D3 | Ingress matches the allowlists | Helm sends several provider prefixes and `/v1/live/sessions` to the backend, where the websocket 404s | `helm/litellm/templates/ingress.yaml:90-108` | High | |
| G-D4 | MCP tool calls are data plane | `/mcp` routes are backend-only | allowlists | Medium | |
| G-D5 | The split is documented and published | Not in `ARCHITECTURE.md`, images only built in an image-scan workflow | `image-scan.yml:197,257` | Low | |

## Python and Rust

| ID | Intent | Reality | Evidence | Impact | Inv |
|---|---|---|---|---|---|
| G-R1 | The Rust gateway replaces the Python proxy | Master-key auth only, no DB, no accounting, trivial router, management crate not mounted | `crates/gateway*`, `crates/router`, `crates/db` | Medium | |
| G-R2 | Chat, embeddings and responses move to Rust | Bridges exist but the catalog pins all three to Python, and the Rust embeddings route declines | `rust_bridge/catalog.py:70-75` | Low | |
| G-R3 | Rust crate dependencies only point down | `llms` depends on `host` and `python-compat`, `router` on `inference` | crate `Cargo.toml`s | Low | L9 |
| G-R4 | Hard decisions get ADRs | Two ADRs. The biggest decisions are only in `AGENTS.md` prose | `litellm-rust/docs/adrs/` | Medium | |
| G-R5 | Python is fully typed | ADR 001 says not to type requests in Python | ADR 001 vs root `AGENTS.md` | Low | |
| G-R6 | Python and Rust stay in parity | The parity harness is local only | `tests/rust-python-harness/AGENTS.md` | Medium | |
| G-R7 | The native extension is optional | OCR and Bedrock transcription have no Python fallback, so every wheel compiles Rust | `pyproject.toml:307-317` | Low | |
| G-R8 | Dispatch bans stop bypasses | `TID251` covers completion, responses and messages but not embedding, OCR or transcription, and sits inside a ratchet | `ruff-strict.toml:64-75` | Low | L8 |

## Dead or orphaned

| ID | Item | Evidence |
|---|---|---|
| G-Z1 | `mcp_servers.json`, `router_plugins.json` referenced nowhere | repo-wide search |
| G-Z2 | ROI calculator routers, config pass-through settings router, enterprise vector store router never included | [surface-map.md](surface-map.md) |
| G-Z3 | `crates/gateway-management` not mounted, `crates/litellm` placeholder | crate manifests |
| G-Z4 | Four CI checks that pass without checking anything, about 10 unwired scripts in `tests/code_coverage_tests/` | handoff Q6, P1, T9, L7 |

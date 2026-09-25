# LiteLLM feature sitemap

A machine-oriented map of every customer-facing feature in LiteLLM, written for agents verifying code changes. When you change code, find the domain in the table below, open only that file, locate the feature leaf, and use its `verify` recipe as your live proof. Each leaf lists the docs page a customer reads, the code entrypoints, the test dirs that cover it, the coverage-registry ids it feeds, and one concrete check you can run against a live proxy on `localhost:4000` (master key `sk-1234`)

## How to use it

Do not read this file end to end. Pick the domain whose feature your change touches, open that one file, find the leaf, and run the `verify` line. If a leaf's code pointers match the files you edited, the registry ids tell you which e2e behaviors already exist, and the tests column tells you where unit coverage lives

Leaf ids are `<domain>.<feature>[.<sub>]`, stable slugs at feature granularity. They describe what a customer can do, never a single behavior. Behavior-level coverage (one id per asserted behavior like `mgmt.key.update.preserves_unrelated_fields`) lives in `tests/e2e/coverage_registry/*.yaml`; a sitemap leaf maps to a registry id prefix, so `auth.virtual_keys` covers `mgmt.key.*`

`surfaces` is the customer-facing surface: `sdk` (Python SDK), `api` (proxy HTTP route), `config` (config.yaml or env), `ui` (Admin UI page), `cli` (command line). `flags`: `ent` (enterprise-licensed), `beta`, `dep` (deprecated), `db` (needs Postgres), `redis` (needs Redis)

## Domains

| file | scope |
|---|---|
| [01-llm-endpoints.md](01-llm-endpoints.md) | every inference endpoint plus pass-through and cross-cutting request behavior |
| [02-models-providers.md](02-models-providers.md) | model_list, model management API, credentials, provider registry, cost map, model UI |
| [03-routing-reliability.md](03-routing-reliability.md) | routing strategies, fallbacks, retries, cooldowns, auto router, scheduler, rust gateway |
| [04-auth-identity.md](04-auth-identity.md) | master key, virtual keys, users, teams, orgs, projects, customers, RBAC, SSO, SAML, JWT, SCIM |
| [05-budgets-ratelimits-spend.md](05-budgets-ratelimits-spend.md) | budgets at every level, TPM/RPM/parallel limits, spend logs and reporting, billing exports |
| [06-guardrails-policies.md](06-guardrails-policies.md) | guardrail framework and providers, scoping, policy engine, tool policies, MCP security |
| [07-caching-cost-optimization.md](07-caching-cost-optimization.md) | response caches, semantic cache, cache controls, prompt compression, savings pages |
| [08-observability.md](08-observability.md) | callbacks and integrations, otel/prometheus, alerting, email, logs UI, redaction, audit, health |
| [09-agent-mcp-gateway.md](09-agent-mcp-gateway.md) | MCP servers and protocol endpoint, auth, toolsets, A2A, workflows, memory, skills, Claude Code gateway |
| [10-prompt-management.md](10-prompt-management.md) | native prompts, prompt management providers, prompts UI |
| [11-deployment-config-platform.md](11-deployment-config-platform.md) | config.yaml, CLIs, docker/helm, postgres/redis, secret managers, plugins, tuning, license |
| [12-sdk-utilities.md](12-sdk-utilities.md) | SDK-only features: Router, token/cost helpers, exceptions, BudgetManager, adapters, callbacks |
| [13-client-integrations.md](13-client-integrations.md) | pointing Claude Code, Codex, Cursor, Copilot, agent SDKs, and chat UIs at the gateway |
| [14-security-compliance.md](14-security-compliance.md) | redaction, request validation, compliance endpoints, hardened image, audit trail, SSRF, encryption |

Appendices: [appendix-providers.md](appendix-providers.md) is the provider x endpoint capability matrix generated from `provider_endpoints_support.json`; [appendix-integrations.md](appendix-integrations.md) lists logging callbacks, guardrail providers, secret managers, and pass-through providers with their code modules

## Maintenance

Add a leaf when you ship a customer-visible feature, matching the id convention. Keep every path real: code and test paths must exist in the repo, docs links must resolve to a file under `litellm-docs/docs/`, and verify recipes must name real routes, UI page keys, or SDK calls. When a feature moves, update the leaf in place rather than appending a second entry

## Known gaps

Coverage caveats for this pass:

- `docs: none found` appears on `models.deprecation_dates`, `models.api_playground`, `guardrails.monitor`, `policies.block_code_execution`, `costopt.page`, `agents.workflows`, `sdk.budget_manager`, `clients.vscode_extension`: no matching docs page exists in litellm-docs today
- `tests: none found` or thin test pointers on `obs.profiling`, `security.hardened_image`, `clients.vscode_extension`
- `registry: none` on most sub-feature leaves: the coverage registry is behavior-level and only covers a subset of features; leaves that cite `none` have no dedicated e2e cells yet
- `clients.vscode_extension` lists no surfaces in the skeleton and has no docs or registry coverage
- Code entrypoints were taken from route decorators scanned in `litellm/proxy/` and `enterprise/` plus known SDK files; `litellm/proxy/openapi.json` in the repo is a partial hand-written spec, not the generated surface, so a handful of routes registered via `include_router` or `add_api_route` may be under-cited

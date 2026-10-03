# Copilot instructions for LiteLLM

This repo has detailed contributor docs already; read them rather than duplicating:

- `CLAUDE.md` / `AGENTS.md` (identical content) - coding conventions, lint/type-suppression rules, PR/commit conventions. **Read this in full before writing code.**
- `ARCHITECTURE.md` - request flow diagrams for the SDK and the proxy (AI Gateway), the translation-layer pattern, and the data-access layer (`litellm/models/` + `litellm/repositories/`).
- `CONTRIBUTING.md` - setup, testing, linting walkthrough.
- `tests/test_litellm/readme.md` - unit test directory conventions.
- `tests/e2e/CLAUDE.md` and `tests/e2e/CONTRIBUTING.md` - e2e harness rules (shared transport only, no raw `requests`, suite-per-folder layout).

## Big picture

LiteLLM is two things sharing one codebase:

1. **SDK** (`litellm/`): `main.py` (`completion()`/`acompletion()`) resolves provider via `utils.py:get_llm_provider()`, then `llms/custom_httpx/llm_http_handler.py` (`BaseLLMHTTPHandler`) calls a provider's `Config.transform_request()` / `transform_response()` (in `llms/{provider}/chat/transformation.py`, subclassing `BaseConfig` from `llms/base_llm/chat/transformation.py`). You almost never modify the handler itself, just the provider's `transformation.py`.
2. **AI Gateway / Proxy** (`litellm/proxy/`): wraps the SDK with auth (`proxy/auth/user_api_key_auth.py`), rate limiting/budgets (`proxy/hooks/`), and routing (`router.py`), then calls into the SDK's `main.py` for the actual LLM call. Cost is calculated post-call in `litellm_logging.py` -> `cost_calculator.py` and queued to Postgres via `proxy/db/db_spend_update_writer.py`.

Persistence/caching: Redis (`caching/redis_cache.py`, `caching/dual_cache.py`) for rate limits/API-key cache/cooldowns; Postgres (`proxy/schema.prisma`) for keys/teams/users/spend logs, accessed through `litellm/repositories/` (`BaseRepository[T]` + entity repos) with entity Pydantic models in `litellm/models/` (re-exported from `proxy/_types.py` for backwards compat).

To add/modify a provider or feature, find the relevant `transformation.py` from the table in `ARCHITECTURE.md` section 3, and add unit tests in `tests/llm_translation/test_{provider}.py` that call `transform_request`/`transform_response` directly (no live API calls needed).

## Build / install

- Package manager is `uv`, not pip/poetry directly.
- `make install-dev` - base dev deps. `make install-proxy-dev` - adds proxy extras. `make install-test-deps` - full env incl. Postgres/Prisma, needed for proxy or DB-backed tests.
- `make bootstrap` - full fresh-clone/worktree setup (deps, Prisma client, UI npm install).

## Tests

- Unit tests: `tests/test_litellm/` mirrors `litellm/`'s structure 1:1 (e.g. `litellm/utils.py` -> `tests/test_litellm/test_utils.py`); **mocked only, no real API calls**.
- Run one file: `uv run pytest tests/test_litellm/test_your_file.py -v`. Run one test: append `::test_name`.
- Run everything: `make test-unit` (parallelized, `tests/test_litellm`). Other `make test-unit-*` targets split proxy/llms/integration tests into matrix groups mirroring CI (see `make help`).
- Integration/live-API tests live under `tests/llm_translation/`, `tests/e2e/` (real proxy + real providers, see e2e CLAUDE.md/CONTRIBUTING.md), etc. — not mocked.
- e2e tests must use the shared transport (`e2e_http.py`), never `requests.*` directly (enforced by CI check).

## Lint / type-check

- `make format` - ruff format (line length 120, not 88). `make lint` - full CI-parity lint (ruff, basedpyright budget gates, circular imports, import safety). `make lint-dev` - faster, changed-files-only version for local iteration.
- basedpyright, ruff-strict, and type-discipline rules are gated by ratcheting budget files (`basedpyright-code-budget.json`, `ruff-strict-budget.json`, `type-discipline-budget.json`). If your change fixes violations, run `make lint-budget-update` and commit the lowered budgets.
- `make pre-commit` runs the CI-equivalent checks on staged files and writes full output to a log file in `.git` (path printed first/last line) — read that log instead of re-running.
- Every lint/type suppression must cite the exact rule and a reason, e.g. `# pyright: ignore[reportArgumentType]  # ...`. `# type: ignore` is banned (disabled in `pyrightconfig.json`).

## Key conventions (see CLAUDE.md/AGENTS.md for full list)

- No mutation: don't reassign variables; annotate with `: Final`; build collections in one shot (comprehensions -> `tuple()`/`MappingProxyType()`/`frozenset()`) instead of seeding-and-mutating (`LIT001`/`LIT002`).
- Fully typed; no bare `Any`/`dict`. Validate untyped inputs via Pydantic in the caller rather than loosening types.
- Composition over inheritance; early returns over nesting; model failures as values rather than raising where a public contract exists.
- Conventional Commits for commits/PR titles; branch off `litellm_internal_staging` (not `main`), branch names use `litellm_` prefix with no `/`.

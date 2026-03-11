# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation
- `make install-dev` - Install core development dependencies
- `make install-proxy-dev` - Install proxy development dependencies with full feature set
- `make install-test-deps` - Install all test dependencies

### Testing
- `make test` - Run all tests
- `make test-unit` - Run unit tests (tests/test_litellm) with 4 parallel workers
- `make test-integration` - Run integration tests (excludes unit tests)
- `pytest tests/` - Direct pytest execution

### Code Quality
- `make lint` - Run all linting (Ruff, MyPy, Black, circular imports, import safety)
- `make format` - Apply Black code formatting
- `make lint-ruff` - Run Ruff linting only
- `make lint-mypy` - Run MyPy type checking only

### Single Test Files
- `poetry run pytest tests/path/to/test_file.py -v` - Run specific test file
- `poetry run pytest tests/path/to/test_file.py::test_function -v` - Run specific test

### Running Scripts
- `poetry run python script.py` - Run Python scripts (use for non-test files)

### GitHub Issue & PR Templates
When contributing to the project, use the appropriate templates:

**Bug Reports** (`.github/ISSUE_TEMPLATE/bug_report.yml`):
- Describe what happened vs. what you expected
- Include relevant log output
- Specify your LiteLLM version

**Feature Requests** (`.github/ISSUE_TEMPLATE/feature_request.yml`):
- Describe the feature clearly
- Explain the motivation and use case

**Pull Requests** (`.github/pull_request_template.md`):
- Add at least 1 test in `tests/litellm/`
- Ensure `make test-unit` passes

## Architecture Overview

LiteLLM is a unified interface for 100+ LLM providers with two main components:

### Core Library (`litellm/`)
- **Main entry point**: `litellm/main.py` - Contains core completion() function
- **Provider implementations**: `litellm/llms/` - Each provider has its own subdirectory
- **Router system**: `litellm/router.py` + `litellm/router_utils/` - Load balancing and fallback logic
- **Type definitions**: `litellm/types/` - Pydantic models and type hints
- **Integrations**: `litellm/integrations/` - Third-party observability, caching, logging
- **Caching**: `litellm/caching/` - Multiple cache backends (Redis, in-memory, S3, etc.)

### Proxy Server (`litellm/proxy/`)
- **Main server**: `proxy_server.py` - FastAPI application
- **Authentication**: `auth/` - API key management, JWT, OAuth2
- **Database**: `db/` - Prisma ORM with PostgreSQL/SQLite support
- **Management endpoints**: `management_endpoints/` - Admin APIs for keys, teams, models
- **Pass-through endpoints**: `pass_through_endpoints/` - Provider-specific API forwarding
- **Guardrails**: `guardrails/` - Safety and content filtering hooks
- **UI Dashboard**: Served from `_experimental/out/` (Next.js build)

## Key Patterns

### Provider Implementation
- Providers inherit from base classes in `litellm/llms/base.py`
- Each provider has transformation functions for input/output formatting
- Support both sync and async operations
- Handle streaming responses and function calling

### Error Handling
- Provider-specific exceptions mapped to OpenAI-compatible errors
- Fallback logic handled by Router system
- Comprehensive logging through `litellm/_logging.py`

### Configuration
- YAML config files for proxy server (see `proxy/example_config_yaml/`)
- Environment variables for API keys and settings
- Database schema managed via Prisma (`proxy/schema.prisma`)

## Development Notes

### Code Style
- Uses Black formatter, Ruff linter, MyPy type checker
- Pydantic v2 for data validation
- Async/await patterns throughout
- Type hints required for all public APIs
- **Avoid imports within methods** — place all imports at the top of the file (module-level). Inline imports inside functions/methods make dependencies harder to trace and hurt readability. The only exception is avoiding circular imports where absolutely necessary.

### Testing Strategy
- Unit tests in `tests/test_litellm/`
- Integration tests for each provider in `tests/llm_translation/`
- Proxy tests in `tests/proxy_unit_tests/`
- Load tests in `tests/load_tests/`
- **Always add tests when adding new entity types or features** — if the existing test file covers other entity types, add corresponding tests for the new one

### UI / Backend Consistency
- When wiring a new UI entity type to an existing backend endpoint, verify the backend API contract (single value vs. array, required vs. optional params) and ensure the UI controls match — e.g., use a single-select dropdown when the backend accepts a single value, not a multi-select

### MCP OAuth / OpenAPI Transport Mapping
- `TRANSPORT.OPENAPI` is a UI-only concept. The backend only accepts `"http"`, `"sse"`, or `"stdio"`. Always map it to `"http"` before any API call (including pre-OAuth temp-session calls).
- FastAPI validation errors return `detail` as an array of `{loc, msg, type}` objects. Error extractors must handle: array (map `.msg`), string, nested `{error: string}`, and fallback.
- When an MCP server already has `authorization_url` stored, skip OAuth discovery (`_discovery_metadata`) — the server URL for OpenAPI MCPs is the spec file, not the API base, and fetching it causes timeouts.
- `client_id` should be optional in the `/authorize` endpoint — if the server has a stored `client_id` in credentials, use that. Never require callers to re-supply it.

### MCP Credential Storage
- OAuth credentials and BYOK credentials share the `litellm_mcpusercredentials` table, distinguished by a `"type"` field in the JSON payload (`"oauth2"` vs plain string).
- When deleting OAuth credentials, check type before deleting to avoid accidentally deleting a BYOK credential for the same `(user_id, server_id)` pair.
- Always pass the raw `expires_at` timestamp to the client — never set it to `None` for expired credentials. Let the frontend compute the "Expired" display state from the timestamp.
- Use `RecordNotFoundError` (not bare `except Exception`) when catching "already deleted" in credential delete endpoints.

### Browser Storage Safety (UI)
- Never write LiteLLM access tokens or API keys to `localStorage` — use `sessionStorage` only. `localStorage` survives browser close and is readable by any injected script (XSS).
- Shared utility functions (e.g. `extractErrorMessage`) belong in `src/utils/` — never define them inline in hooks or duplicate them across files.

### Database Migrations
- Prisma handles schema migrations
- Migration files auto-generated with `prisma migrate dev`
- Always test migrations against both PostgreSQL and SQLite

### Proxy database access
- **Do not write raw SQL** for proxy DB operations. Use Prisma model methods instead of `execute_raw` / `query_raw`.
- Use the generated client: `prisma_client.db.<model>` (e.g. `litellm_tooltable`, `litellm_usertable`) with `.upsert()`, `.find_many()`, `.find_unique()`, `.update()`, `.update_many()` as appropriate. This avoids schema/client drift, keeps code testable with simple mocks, and matches patterns used in spend logs and other proxy code.

### Enterprise Features
- Enterprise-specific code in `enterprise/` directory
- Optional features enabled via environment variables
- Separate licensing and authentication for enterprise features

### HTTP Client Cache Safety
- **Never close HTTP/SDK clients on cache eviction.** `LLMClientCache._remove_key()` must not call `close()`/`aclose()` on evicted clients — they may still be used by in-flight requests. Doing so causes `RuntimeError: Cannot send a request, as the client has been closed.` after the 1-hour TTL expires. Cleanup happens at shutdown via `close_litellm_async_clients()`.

### Troubleshooting: DB schema out of sync after proxy restart
`litellm-proxy-extras` runs `prisma migrate deploy` on startup using **its own** bundled migration files, which may lag behind schema changes in the current worktree. Symptoms: `Unknown column`, `Invalid prisma invocation`, or missing data on new fields.

**Diagnose:** Run `\d "TableName"` in psql and compare against `schema.prisma` — missing columns confirm the issue.

**Fix options:**
1. **Create a Prisma migration** (permanent) — run `prisma migrate dev --name <description>` in the worktree. The generated file will be picked up by `prisma migrate deploy` on next startup.
2. **Apply manually for local dev** — `psql -d litellm -c "ALTER TABLE ... ADD COLUMN IF NOT EXISTS ..."` after each proxy start. Fine for dev, not for production.
3. **Update litellm-proxy-extras** — if the package is installed from PyPI, its migration directory must include the new file. Either update the package or run the migration manually until the next release ships it.
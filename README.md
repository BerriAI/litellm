# LIT-7135 verification

Before and after use clean git archives at the hashes in before-sha and after-sha, built with the repository root Dockerfile. image-provenance.json records both source image IDs. The same PostgreSQL records and real MCP SDK HTTP fixture are used on both sides. Chrome runs headlessly with a separate profile

## Setup and reproduction

1. Build each source archive with `docker build -t litellm-7135:before .` or `docker build -t litellm-7135:after .`
2. Put fresh local-only LITELLM_MASTER_KEY, LITELLM_SALT_KEY and POSTGRES_PASSWORD values in an untracked .env next to compose.yaml
3. Run `REVISION=before docker compose up -d`, wait for http://localhost:47135/health/liveliness, and run `python seed.py` with Python 3.12
4. Log into http://localhost:47135/ui/mcp-servers/ using the local admin key. Open each preview server card, then Settings
5. For preview_auth, change Authentication from None to Basic Auth and enter the fixture credential `preview:correct`
6. For preview_url, change `/wrong` to `/mcp`, leaving Authentication Value blank to retain its stored credential
7. For preview_headers, expand Permission Management and change X-Preview-Key from wrong to correct
8. Do not Save. Before: Unable to load tools persists, with no POST for the edited configuration
9. Run `REVISION=after docker compose up -d --no-deps gateway` and repeat steps 4 through 7. After: Tool Configuration shows echo after typing stops and discovery finishes. Click Flat List to inspect it
10. Save the corrected Basic record. Save returns 202. GET /mcp-rest/tools/list?server_id=<id> returns echo. POST /mcp-rest/tools/call with the payload in happy-path.json returns the supplied message

capture.py documents the exact browser steps and emits sanitized network records. Credentials shown in the fixture are dummy values; actual local proxy and database secrets are excluded. The fixture rejects invalid credentials with 403 to avoid a browser HTTP-auth challenge intercepting fetch

The hidden server tools tab can retry GET requests against the broken saved record while Settings is open. The after screenshots and POST results demonstrate that those errors do not replace the successful edited preview

## Checks

regression-tests.log: 129 dashboard tests pass. backend-tests.log: 151 mapped REST endpoint tests pass. The corresponding before logs demonstrate four failing UI regressions and twelve failing backend cases at the merge base

check-final.log: full make check passes, including Ruff formatting/lint, test-tree lint, strict-rule, type-discipline, test-quality, basedpyright budget, dashboard lint budgets and generated API type sync. ui-build.log: production dashboard build passes. Full backend suites are left to CI

verify-origin.py checks live host-change credential handling. origin-boundary.json records zero UI preview requests before credential entry, followed by a successful tool list after explicit entry

# Local test results for LIT-7135

- 125 affected dashboard tests passed: edit preview rules, edit form integration and existing OAuth cases, tool configuration
- Four new dashboard integration cases fail against merge base 6b721de3e530bc97758d59621e57e3a75a9ebadf: corrected Basic Auth, corrected static header, rapid URL edits with stale responses, saved-response races and reverting changes
- 131 tests passed in tests/test_litellm/proxy/_experimental/mcp_server/test_rest_endpoints.py, including seven new cases that exercise real MCP client configuration and the saved-server registry
- The seven backend regressions fail at the merge base: the client uses the saved URL, HTTP transport and Basic scheme despite the edited SSE configuration and auth choice
- Production Next.js build passed; Docker source builds compile the dashboard too
- Full Ruff, test-tree lint, strict-rule, type-discipline, test-quality and basedpyright gates passed against the merge base, in Python 3.12 with proxy-dev/e2e-dev and a generated Prisma client
- Dashboard formatting, ESLint and whole-dashboard warning budgets pass without increasing budgets

Live before/after evidence and final check logs are attached separately. The fixture is an actual MCP SDK server over HTTP, with a Basic credential or X-Preview-Key header requirement. Its rejection uses HTTP 403 so browser HTTP-auth prompts do not intercept fetch requests. No mocks or model calls are used in this live test.

Final tip: 245369764e80d113e80ec8269846e5257692b8bc

Live cases: Basic Auth correction, URL correction with inherited credential, and static-header correction all return echo before Save. Saving the corrected Basic record returns 202; real tool-list and echo tool-call both return 200. See happy-path.json

Additional public MCP happy path: https://learn.microsoft.com/api/mcp returned three tools; microsoft_docs_search with query Azure Functions overview returned 200 and documentation content. See public-happy-path.json

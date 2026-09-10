# LIT-7135 verification results

Final tip: 3169c80252e6e98f6c68ff77869244dd9788b796
Merge base: 6b721de3e530bc97758d59621e57e3a75a9ebadf

129 affected dashboard tests pass. 151 tests in the mapped backend REST endpoint file pass. Four new UI integration cases fail at the merge base. The backend regression subset at the merge base has twelve expected failures: seven stale connection cases and five rejected-origin inherited-credential cases; fifteen other cases pass

Production dashboard build and both source Docker builds pass. Full make check passes all applicable gates: Ruff lint/format, test-tree lint, strict-rule, type-discipline, test-quality, basedpyright budgets, dashboard lint/budgets and generated API-type sync. No budgets or blanket suppressions were added. Existing diagnostics remain within the repository budgets

Live screenshots and sanitized network traces demonstrate Basic Auth correction, same-origin URL correction with a blank credential field, and static-header correction. Each final-tip POST returns echo and clears the stale failure before Save. Before screenshots show the same corrections leaving Unable to load tools, without preview POSTs

The origin-boundary run changes the hostname. The UI shows an actionable instruction and sends zero preview requests until an explicit credential is entered, then lists echo. A direct preview request with a saved ID but no explicit credential does not inherit the saved secret across hosts; an explicit credential succeeds. Unit coverage also verifies scheme changes, non-default ports and same-origin default-port normalization

After saving corrected Basic Auth, Save returns 202, listing returns 200 with echo, and the echo tool call returns 200 with the expected message. Public Microsoft Learn MCP also returns three tools and a successful read-only documentation search on the final tip

The fixture is an actual MCP SDK server over HTTP. It rejects invalid credentials with 403 so a browser HTTP-auth challenge does not intercept fetch requests. No mocks or model calls are used in live verification. The browser waits for the initial form load before editing fields

Files: check-final.log, regression-tests.log, regression-before.log, backend-tests.log, backend-regression-before.log, ui-build.log, happy-path.json, public-happy-path.json, origin-boundary.json and matching before/after screenshots/network JSON

IPv6 route verification: POST /mcp-rest/test/tools/list with http://[::1]:9/mcp returns a structured connection error with HTTP 200, rather than an origin-parsing 500. See ipv6-preview.json and verify-ipv6.py. Backend cases cover IPv6 same/different origins, default ports, casing and invalid ports

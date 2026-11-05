# Issues API proof

PR: https://github.com/BerriAI/litellm/pull/50187

The narrated video records actual HTTP requests from a local test client. The saved Issues, supporting observations, and historical daily aggregates are explicitly synthetic. This proves the storage and API layer; automatic detection is covered by the detector PR.

- Before: exact PR1 parent `16d1e1ebb91`. Both `/issues` and `/usage/alert-rules` return 404.
- After: `6d31c9536412caab41928b4ed0e1baa7fadae878`, verified clean before and after capture.
- Three Issues share server pagination. Saved key comparison and Activity agree with the fixture range.
- Administrators see two supporting signals; the key owner sees only its own signal, with total one. Provider-wide access returns 403.
- Status changes preserve saved evidence; partial family updates preserve other settings, families and custom rules.
- Two API workers, dedicated Redis and an isolated temporary PostgreSQL schema. Fixtures, schema and processes were removed afterward.

Eight real HTTP/Prisma checks passed at `76275022d44bd4768124492f9517af001e702dd6`; their unchanged report is `storage-checks-at-76275022.json`. The following 6d31 commit only refreshed the generated OpenAPI snapshot. Later import/formatting cleanup should be documented separately rather than attributed to this recording.

The MP4 is 1920×1080, 97.23 seconds, with seven narration clips. `http-trace.json` contains the complete safe request/response bodies. Screenshots display selected actual response fields.

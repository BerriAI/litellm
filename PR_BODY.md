## Relevant issues

Fixes the "active and paused models are indistinguishable in the models table" pain: the table mixes both states, has no status column, and the drawer can only filter by public model name and access group.

## Pre-Submission checklist

- [x] I ran `npm run build` in `ui/litellm-dashboard` without errors
- [x] I ran `npx vitest run` for the models-and-endpoints suite: 205 passed
- [x] I added/updated unit tests (`test_routes_model_info.py`: 4 new tests)
- [x] I verified the change end-to-end against a live proxy (39 deployments: filter returns exactly 9 active / 20 paused; `sortBy=blocked&sortOrder=asc` puts active first; `search=nvidia_nim` finds deployments by provider)

## What changed

### Backend — `GET /v2/model/info`

- New optional `blocked` query param: `true` = only paused deployments, `false` = only active ones. Omitting it keeps the current behavior for every existing caller.
- New `blocked` sort field (`sortBy=blocked&sortOrder=asc` = active first). The existing `status` sort field is unchanged: it still sorts by config-vs-DB source, which is why it never grouped active/paused rows.
- `search` now also matches `litellm_params.model`, so typing a provider or upstream model id (`nvidia_nim`, `openrouter/deepseek`, `openai/gpt-4`…) finds deployments whose public model name doesn't mention them.
  - Router-side matching is case-insensitive; the DB branch uses Prisma JSON `string_contains`, which is case-sensitive on Postgres (same limitation already documented for that query path). Rows already loaded in the router — the common case, including all DB-backed deployments — go through the case-insensitive path.

### Frontend — Models & Endpoints

- New visible **Status** column (`Active` / `Paused` badges), sortable server-side.
- New **Status** filter (All / Active / Paused) in the Filters drawer, wired through URL state (`?status=active|paused`) and the new server param, so pagination totals stay correct while filtered.
- The **Actions** column is now pinned to the right edge, so the pause/resume switches stay visible when the table overflows horizontally.
- The pre-existing hidden "Source" column (DB vs config) is untouched.

### Sentinel safety

The routing-status filter only applies when the parameter is literally `True`/`False`. Direct calls that bypass FastAPI receive the truthy Query sentinel as the default; guarding on identity (same pattern as `exclude_auto_routers`' `is True`) keeps their no-filter behavior. A regression test pins this (`test_model_info_v2_query_sentinel_does_not_filter` still passes).

## Tests

- `tests/test_litellm/proxy/proxy_server/test_routes_model_info.py`: 4 new tests (blocked filter both ways, blocked sort, search over `litellm_params.model`). Full-file run: same pre-existing failures before and after the change (verified by reverting the patch), i.e. no regressions.
- `ui/litellm-dashboard`: 208/208 tests pass in the models-and-endpoints suite (1 updated for the new column, 1 new for the drawer filter, 3 new for the status mapping).

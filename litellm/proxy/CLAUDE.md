# Proxy Server (`litellm/proxy/`)

Rules that apply across the proxy server. Sub-domains (auth, guardrails, management endpoints, etc.) have their own nested CLAUDE.md files.

## Proxy database access
- **Do not write raw SQL** for proxy DB operations. Use Prisma model methods instead of `execute_raw` / `query_raw`.
- Use the generated client: `prisma_client.db.<model>` (e.g. `litellm_tooltable`, `litellm_usertable`) with `.upsert()`, `.find_many()`, `.find_unique()`, `.update()`, `.update_many()` as appropriate. This avoids schema/client drift, keeps code testable with simple mocks, and matches patterns used in spend logs and other proxy code.
- **No N+1 queries.** Never query the DB inside a loop. Batch-fetch with `{"in": ids}` and distribute in-memory.
- **Batch writes.** Use `create_many`/`update_many`/`delete_many` instead of individual calls (these return counts only; `update_many`/`delete_many` no-op silently on missing rows). When multiple separate writes target the same table (e.g. in `batch_()`), order by primary key to avoid deadlocks.
- **Push work to the DB.** Filter, sort, group, and aggregate in SQL, not Python. Verify Prisma generates the expected SQL — e.g. prefer `group_by` over `find_many(distinct=...)` which does client-side processing.
- **Bound large result sets.** Prisma materializes full results in memory. For results over ~10 MB, paginate with `take`/`skip` or `cursor`/`take`, always with an explicit `order`. Prefer cursor-based pagination (`skip` is O(n)). Don't paginate naturally small result sets.
- **Limit fetched columns on wide tables.** Use `select` to fetch only needed fields — returns a partial object, so downstream code must not access unselected fields.
- **Check index coverage.** For new or modified queries, check `schema.prisma` for a supporting index. Prefer extending an existing index (e.g. `@@index([a])` → `@@index([a, b])`) over adding a new one, unless it's a `@@unique`. Only add indexes for large/frequent queries.
- **Keep schema files in sync.** Apply schema changes to all `schema.prisma` copies (`schema.prisma`, `litellm/proxy/`, `litellm-proxy-extras/`, `litellm-js/spend-logs/` for SpendLogs) with a migration under `litellm-proxy-extras/litellm_proxy_extras/migrations/`.

## Database Migrations
- Prisma handles schema migrations
- Migration files auto-generated with `prisma migrate dev`
- Always test migrations against both PostgreSQL and SQLite

## Troubleshooting: DB schema out of sync after proxy restart
`litellm-proxy-extras` runs `prisma migrate deploy` on startup using **its own** bundled migration files, which may lag behind schema changes in the current worktree. Symptoms: `Unknown column`, `Invalid prisma invocation`, or missing data on new fields.

**Diagnose:** Run `\d "TableName"` in psql and compare against `schema.prisma` — missing columns confirm the issue.

**Fix options:**
1. **Create a Prisma migration** (permanent) — run `prisma migrate dev --name <description>` in the worktree. The generated file will be picked up by `prisma migrate deploy` on next startup.
2. **Apply manually for local dev** — `psql -d litellm -c "ALTER TABLE ... ADD COLUMN IF NOT EXISTS ..."` after each proxy start. Fine for dev, not for production.
3. **Update litellm-proxy-extras** — if the package is installed from PyPI, its migration directory must include the new file. Either update the package or run the migration manually until the next release ships it.

# Database

Use [database-access](../../database-access/SKILL.md) to keep repository extraction, query-only ports and caller integration scoped to the requested work

Background in the [Database design](https://app.notion.com/p/Database-40543b8acdab81bd9738d924c7617302)

## Python surface today

- python-proxy only. python-sdk doesn't use the database
- Callers (routes, auth, spend, jobs) reach PostgreSQL partly through `litellm/repositories` and partly through direct Prisma calls and raw SQL
- `PrismaClient` holds the writer and an optional reader
- Everything moves behind repositories

## Shared crates

- `crates/db` exposes typed operations and owns SQL in `queries/`. Rust callers use it directly
- Python passes typed arguments through PyO3, never SQL strings

## Per host

- python-proxy
  - Routes receive `Repositories` and call domain APIs such as `repositories.banners.get_banner()`
  - Each operation has one implementation, Prisma or native. Operations in the same repository may use different backends
  - No duplicate query, backend switch or Prisma fallback
  - A whole transaction stays on one backend
  - `Database` and FastAPI dependency injection are optional conveniences, not prerequisites. Port an operation without changing its callers
  - Existing startup and shutdown code manages Prisma, and the SQLx pool joins it when introduced. Close opened connections if startup fails, finish active work before disconnecting
- rust-proxy
  - Calls `crates/db` through its own SQLx pool
- Prisma and SQLx keep separate pools, even with PgBouncer. Size both layers together and keep reader and writer routing

## Wire contracts

- Same schema for both. Every row both sides write is a wire contract

## Order and gates

1. Extract Prisma calls into typed repositories, alone or in the same PR as a port
2. Port one operation or a whole transaction. Validate against real PostgreSQL and refresh SQLx metadata
3. Switch all callers and delete the Prisma query in the same change
4. Ship native support for ported operations. Roll back releases if needed
5. Retire Prisma queries first, then migrations through a separate SQLx cutover

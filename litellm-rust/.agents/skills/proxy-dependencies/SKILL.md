---
name: proxy-dependencies
description: Inject typed repositories and services into Python proxy handlers, or change database runtime ownership while preserving behavior for future Rust backends
---

# Proxy dependencies

- Follow the root [database-access skill](../../../../.agents/skills/database-access/SKILL.md) for queries, repository extraction and backend migration
- This skill owns dependency wiring and resource lifetime, not SQL or schema ownership

## Handler contracts

- Inject named FastAPI providers through `Annotated[..., Depends(...)]`
- Expose narrow repository protocols and typed service callables, not Prisma clients, models, tables or query dictionaries
- Keep database I/O out of dependency construction
- Keep no-database behavior and authorization precedence specific to the route
- Ordinary helpers and background jobs receive explicit arguments, never unresolved `Depends` defaults
- Override named providers in route tests and keep production parsing and persistence code real

## Database ownership

- `litellm._db.Database` exposes typed repositories and an async close operation
- `from_prisma` adopts an already initialized client without opening a second pool
- FastAPI lifespan enters `_db.lifespan` and publishes its handle in request state
- Drain database consumers before exiting that context, which alone closes the handle
- Do not keep a parallel Prisma disconnect fallback in proxy shutdown
- Legacy globals remain compatibility access for unmigrated code, not a second pool or owner
- Never close a database from a request dependency or repository operation
- SQLx integration is a separate explicit change, preserve transaction boundaries and do not replay failed writes through Prisma

## Migration process

- Inventory imports, direct callers, state writers, side effects and error ordering first
- Migrate one complete feature and its helper calls before expanding
- Preserve response models, OpenAPI, audit scheduling, configuration sync and serialization
- Verify default behavior, authorization, invalid input, storage failure and shutdown ordering
- State which legacy reads and backend leaks remain
- Read [route migration](references/route-migration.md) for the banner example
- Read [runtime ownership](references/runtime-ownership.md) before changing resource lifetime or reload behavior

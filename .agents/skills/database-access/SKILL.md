---
name: database-access
description: Organize database repositories and proxy dependency wiring, manage connection ownership, or port Prisma operations to Rust and SQLx without expanding migration scope
metadata:
  internal: true
---

# Database access

## Situations

- Adding a query or an operation to a repository, read [adding a query](references/python-repository.md#adding-a-query)
- Moving existing Prisma calls behind a repository, read [moving existing access](references/python-repository.md#moving-existing-access-into-a-repository)
- Porting a repository from Prisma to Rust and SQLx, read [porting to SQLx](references/python-repository.md#porting-a-repository-to-sqlx)
- Injecting a repository into a route, read [Python and FastAPI](references/python-fastapi.md) or [Rust and Axum](references/rust-axum.md)

## Adding new database access

- Put Python database access behind an operation in `litellm/repositories`
- Give each repository its own FastAPI provider, so a route builds only the repositories it uses
- A provider borrows existing handles and performs no database I/O during construction
- Define the operations a route calls as a Protocol in `litellm/models/<feature>.py`, next to its domain types
  - The Protocol takes and returns domain types, never Prisma models, raw rows or serialized payloads
  - Parsing and serialization of stored values live in the repository
  - Add a Protocol when a repository gets a provider, not up front for every repository
- Keep query execution inside the repository, with one implementation per repository and no backend switch or fallback
- A repository is either all Prisma or all native, never a mix of both
- Prisma is fine for the implementation, adding a query does not require a Rust port
- Native Rust callers use `litellm-rust/crates/db` directly

## Framework wiring

- Python routes receive a per-repository dependency such as `UserBannerRepositoryDep` and call its operations, such as `banners.get_banner()`
- `DatabaseDep` lives in `litellm/proxy/deps/database.py`, repository providers in `litellm/proxy/deps/repositories.py`
  - A provider returns the feature's Protocol, so routes never see which backend implements it
  - Dependencies used by one feature only, like the banner audit binding, stay next to that feature's routes
- Repositories never import FastAPI, so jobs and helpers can use them without a request
- Use named FastAPI providers with `Annotated[..., Depends(...)]`, ordinary helpers and jobs receive explicit arguments
- Preserve each route's no-database behavior, authorization precedence and background-task scheduling
- Read [Python and FastAPI](references/python-fastapi.md) for provider examples, lifespan state and route verification
- Read [Rust and Axum](references/rust-axum.md) when adding database access to Rust handlers

## Connection ownership

- Keep connection setup and shutdown separate from repositories, which own no pools
- Reuse existing handles, never connect per request or disconnect from a repository operation
- Give each pool one owner for setup and shutdown, borrowed handles never close it
- Finish requests, jobs and queued database work before closing connections
- Clean up resources opened before a startup failure, without a second disconnect fallback
- Add a backend to the existing lifecycle owner and pass its handle only to repositories that need it
- Preserve cancellation, timeout, error and transaction semantics across the native bridge

## Changing ownership or reloading resources

- Inventory initialization, reload, tests, background jobs and shutdown together
- Move every writer when changing authoritative storage, copying a replaceable global into state is insufficient
- A frozen repository does not make its clients immutable
- Decide whether each request captures one resource generation or reads it again during processing
- Build and validate replacements before publishing them, preserving the old generation on failure
- Serialize reloads and coordinate publication with shutdown
- Drain streams and background users before retiring old resources, retaining every generation grows resource use
- Define multi-worker coordination separately from an in-process swap
- Choose mechanisms such as `ArcSwap` only after defining publication, capture and retirement

## Migration context and boundaries

- Long-term direction is to centralize Python database access in repositories, then replace their Prisma implementations with Rust APIs one repository at a time
- Prisma and native repositories coexist during the transition, unrelated existing calls can stay as they are
- Prisma and SQLx keep separate client pools to PgBouncer, which pools connections to PostgreSQL
  - Size both layers together and preserve reader/writer routing and pooling-mode compatibility
- PgBouncer does not share transactions between Prisma and SQLx, keep each atomic unit of work on one backend and transaction
- Never retry a failed Rust write through Prisma when it may already have committed
- Prisma owns schema migrations today, eventual SQLx migration ownership and Prisma removal are separate planned steps, not part of a query port

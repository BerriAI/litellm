# Python repositories

## Adding a query

Use this when a feature needs a read or write that no repository offers yet, including one more operation on an existing repository

### Before starting

- Search `litellm/repositories` for a repository that already owns the table, and extend it instead of adding a second one
- Check `litellm/models/<feature>.py` for an existing Protocol and domain types
- Decide the route's behavior when no database is configured, and where that check sits relative to authorization and validation

### Steps

1. Domain types and Protocol in `litellm/models/<feature>.py`
   - The Protocol takes and returns domain types, never Prisma models, raw rows or serialized payloads
   - Add the Protocol together with the provider, not for repositories nothing injects
2. Repository in `litellm/repositories/<feature>_repository.py`
   - Implement the operation with Prisma, a Rust port is not required
   - Compose the Prisma client and type the tables it touches with `TableActions` from `prisma_protocols.py`, never inherit `PrismaTableRepository`
   - Parse and serialize stored values here, so callers only see domain types
   - Never import FastAPI, connect, disconnect or choose replicas per call
3. Provider and `Annotated` alias in `litellm/proxy/deps/repositories.py`, see [dependency injection](python-fastapi.md#dependency-injection)
4. The route takes `<Feature>RepositoryDep`, handles `None` with the behavior decided above, and calls the operation
5. Jobs and helpers that need the operation receive the repository as an explicit argument, built by their caller from the handle it already holds

### Done when

- Handler tests drive the real route with `dependency_overrides` and the real repository code, see [verification](python-fastapi.md#verification)
- Tests cover the no-database path, error precedence, and what was actually stored
- Persistence changes are checked against real PostgreSQL through a live proxy

### Out of scope

- Porting the query to Rust, see [porting a repository to SQLx](#porting-a-repository-to-sqlx)
- Moving unrelated direct Prisma calls in the same module

## Moving existing access into a repository

Use this when code calls Prisma directly, through `prisma_client.db.<table>` or a helper, and the calls should live behind a repository. The backend stays Prisma

### Before starting

- List every call being moved, its callers, and the transaction or batch it runs in
- List the module's imports, global state writers, side effects such as audit and config sync, and error ordering
- Record current return values, raised errors and response bodies, these are what the change must preserve

### Steps

1. Move the existing Prisma code into the relevant repository in `litellm/repositories` without rewriting the queries
2. Add domain types and a Protocol in `litellm/models/<feature>.py` when a route will inject the repository
3. Replace each direct call with the repository operation, through a provider for routes and explicit arguments for jobs and helpers, see [Python and FastAPI](python-fastapi.md)
4. Calls sharing a transaction move into the same repository or unit of work in one change
5. State which legacy reads remain and why, when the change does not move all of them

### Done when

- Focused regression tests pin the preserved return values, errors and transaction behavior
- Response bodies, status codes and OpenAPI match the base
- Configuration sync and serialization behave the same, including invalid stored data

### Out of scope

- Switching the backend, even when an equivalent Rust query exists
- Converting other routes in the module just to establish the pattern
- Cleanups unrelated to the moved calls

## Porting a repository to SQLx

Status: TBD. No repository has been ported yet. `database.native` is a `NativeDatabase` from `crates/python-bridge/src/db.rs` whose `connect` and `close` are noops that open no pool, and the PyO3 repository surface does not exist. A real pool takes typed writer/reader settings and a token provider, never the Prisma `DATABASE_URL`, which carries Prisma-only params and rotating IAM tokens. The constraints below are settled, the steps are a sketch to revise after the first port lands

Use this only when a port is explicitly requested. Repository extraction does not require one

### When not to port

- An equivalent Rust or SQLx query existing is not a reason to switch a Prisma caller
- Leave existing backend selection unchanged unless integration is explicitly requested
- Never port part of a repository, it is either all Prisma or all native

### Before starting

- Identify the repository's inputs, outputs, errors and transaction boundary
- List repositories that share a transaction with it, they move together
- Check `litellm-rust/crates/db` for an equivalent query and reuse it when its behavior matches

### Steps

1. Rust queries, following `litellm-rust/crates/db/AGENTS.md`
   - Static SQL in `queries/`, compile-checked with SQLx, typed inputs and row projections in `src/`
   - Dynamic filters and sort keys become closed Rust enums that select static SQL, every value is bound
   - Refresh SQLx query metadata after SQL changes
2. PyO3 surface exposing typed operations, never SQL text, table names or column names from Python. TBD: where it lives and how the handle reaches it
3. Native repository implementing the same Protocol in `litellm/repositories`
   - Converts typed PyO3 results to domain types, preserving defaults, invalid-data handling and storage errors
   - Python conversion stays in the bridge, not in `crates/db`
4. Switch the provider in `litellm/proxy/deps/repositories.py`, the Protocol and routes stay unchanged

```diff
 async def get_user_banner_repository(database: DatabaseDep) -> UserBannerStore | None:
-    return None if database is None else UserBannerRepository(database.prisma)
+    return None if database is None else NativeUserBannerRepository(database.native)
```

5. Delete the Prisma repository in the same change and switch every caller, including jobs and helpers

### Done when

- Behavior is verified against real PostgreSQL, covering null handling, conflicts, ordering, errors and transaction semantics
- `DATABASE_URL=... cargo test -p litellm-db --locked -- --include-ignored` passes against a Prisma-migrated database
- Live proxy requests show unchanged responses. TBD: a Python-side real-PostgreSQL test harness

### Rollback

- Roll back the release, never retry an uncertain native write through Prisma

### Out of scope

- SQLx owning schema migrations, and removing Prisma
- Converting unrelated routes to dependency injection

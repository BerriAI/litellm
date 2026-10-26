---
name: database-access
description: Work on database access through Python repositories and typed Rust queries. Explains the incremental migration from Prisma to SQLx without requiring unrelated migrations
---

# Database access

## Direction

- Gradually move application database calls behind `litellm/repositories`
- Repositories provide the Python-facing boundary and may use either Prisma or Rust APIs backed by `litellm-rust/crates/db`
- Keep backend selection inside that boundary so callers do not need to know which implementation handles an operation
- Rust callers use `crates/db` directly without going through Python repositories
- This is an incremental direction, not a requirement to migrate existing code whenever it is touched
- Do not expand a feature or bug fix into a database migration unless needed for the task or explicitly requested

## Current state and future plan

- Python repositories already exist, but database access is not fully centralized and existing interfaces still expose Prisma concepts
- `crates/db` provides typed SQLx queries, inputs, and row projections
- A Rust query existing does not mean its Python binding or repository integration exists yet
- Prisma remains responsible for schema migrations today
- The intended progression is
  - Move database operations behind repositories, initially retaining Prisma where appropriate
  - Replace individual repository implementations with Rust-backed operations
  - Reduce Prisma runtime usage until it is needed only for migrations
  - Move migration ownership to SQLx in a separate, explicitly planned transition, then remove Prisma
- Do not introduce a competing SQLx migration history during a query port

## Migrating one operation

- Identify the existing query, callers, return values, errors, and transaction boundary
- Reuse the relevant repository or add a focused operation to it
  - Moving the existing Prisma implementation behind the repository is a useful standalone step
  - Prefer operation-specific typed inputs and outputs over introducing new Prisma-shaped interfaces
  - Preserve existing caller contracts without broadly redesigning unrelated repositories
- Reuse an existing `crates/db` query when its behavior matches
  - Otherwise add a typed query following the crate's local rules
  - Keep static SQL in `queries/`, compile-check it with SQLx, and keep reusable input and row types in `src/`
- For Python callers, expose the operation through the Python bridge and call it from the repository
  - Keep Python conversion and compatibility handling at the boundary
  - Keep `crates/db` independent of Python
- Verify equivalent results, errors, and transaction behavior against real PostgreSQL before switching callers
  - Include relevant null handling, conflict behavior, ordering, and affected-row semantics
  - Refresh SQLx query metadata when changing SQL
- Switch only the selected operation and remove its superseded implementation when no longer needed

## Connections and transactions

- Prisma and Rust may maintain separate connection pools during the transition
- Both may connect through PgBouncer to manage PostgreSQL connection usage
- Account for both pools across workers and replicas, and verify compatibility with the configured PgBouncer pooling mode
- PgBouncer does not make Prisma and SQLx share a transaction
- Keep each atomic unit of work on one backend and one transaction
  - If an operation belongs to a larger transaction, migrate the whole transaction boundary together or leave it on Prisma
- Select the backend before execution
  - Do not retry a failed Rust write through Prisma when the original write may already have committed

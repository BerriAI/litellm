---
name: database-access
description: Add database queries, extract existing Prisma calls into repositories, or explicitly port queries to Rust and SQLx without expanding migration scope
---

# Database access

## Adding new database access

- Put Python database access behind an operation in `litellm/repositories`
- Give callers typed inputs and outputs, not Prisma clients, models, or query dictionaries
- Keep query execution and backend selection inside the repository
- Prisma is fine for the implementation, adding a query does not require a Rust port
- Native Rust callers use `litellm-rust/crates/db` directly

## Asked to move existing access behind a repository

- Identify the calls being moved and their transaction boundary
- Move the existing Prisma implementation into the relevant repository
- Replace those direct database calls with repository operations
- Preserve return values, errors, and transaction behavior, with focused regression tests
- Keep this an extraction, not a backend migration or unrelated cleanup

## An equivalent Rust or SQLx query already exists

- Leave it and the current backend selection unchanged unless integration is explicitly requested
- Its existence alone does not justify switching a Prisma caller to Rust
- Repository extraction can keep using Prisma even when the Rust query is available

## Asked to port a Prisma query to Rust or SQLx

- Identify the existing inputs, outputs, errors, and transaction boundary
- Check `litellm-rust/crates/db` for an equivalent query and reuse it if its behavior matches
- Otherwise follow `litellm-rust/crates/db/AGENTS.md`
  - Put static SQL in `queries/` and compile-check it with SQLx
  - Keep typed inputs and reusable row projections in `src/`
  - Refresh SQLx query metadata after SQL changes
- Keep Python conversion in the Python bridge, not in `litellm-rust/crates/db`
- When caller integration is requested, expose the Rust operation through the bridge and call it from the repository
  - A query-only port can land without switching callers
- Verify behavior against real PostgreSQL before switching callers
  - Check relevant null handling, conflicts, ordering, errors, and transaction semantics
- Remove the old implementation only when no callers need it

## Migration context and boundaries

- Long-term direction is to centralize Python database access in repositories, then replace their Prisma implementations incrementally with Rust APIs
- Mixed Prisma and Rust implementations are expected during the transition, unrelated existing calls can stay as they are
- Both backends may have separate pools through PgBouncer, size them together and verify pooling-mode compatibility
- PgBouncer does not share transactions between Prisma and SQLx, keep each atomic unit of work on one backend and transaction
- Never retry a failed Rust write through Prisma when it may already have committed
- Prisma owns schema migrations today, eventual SQLx migration ownership and Prisma removal are separate planned steps, not part of a query port

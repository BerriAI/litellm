# Runtime ownership

## Current foundation

- Prisma startup still owns connection setup, authentication refresh, health checks and migrations
- Lifespan enters `_db.lifespan` with the initialized client adapted by `from_prisma`
- The `Database` handle provides typed repositories and `aclose`, without exposing Prisma to handlers
- The existing shutdown sequence drains requests, jobs, logging, spend and audit work before disconnect
- `_db.lifespan` alone closes the handle, including startup or shutdown failure
- No parallel disconnect fallback belongs in proxy shutdown
- This does not remove all proxy globals or implement a Rust database backend

## Adding a backend

- Keep domain operations in repositories and conversion at the native bridge
- Define who creates, borrows and closes each pool, including startup-failure cleanup
- Initialize native resources in the serving process, respecting the existing native runtime's fork restrictions
- Await native shutdown instead of depending on Python garbage collection
- Do not create a pool per request or close a shared pool from a borrowed handle
- Preserve error, cancellation, timeout and transaction semantics across the bridge
- Keep each atomic operation on one backend and transaction
- Follow the root database-access skill for backend selection and schema ownership

## Changing ownership

- Inventory initialization, reload, tests, background jobs and shutdown together
- Migrate every writer of a resource when moving its authoritative storage
- Starlette lifespan state is shallow-copied into request state
- `app.state` and yielded lifespan state are distinct mechanisms
- A frozen container does not make its database client, router or cache immutable
- Do not copy a replaceable global into state while leaving reload to update only the global

## Runtime swapping

- Backend selection at startup does not require hot swapping
- Define whether a request captures one runtime generation or reads again during processing
- Build and validate a replacement before publishing it, leaving the old generation usable on failure
- Serialize reloads and coordinate publication with shutdown
- Drain streams and background users before retiring old resources
- Retaining every old generation until shutdown is simple but can grow resource use without bounds
- Define multi-worker coordination separately from an in-process swap
- Axum `State` and `FromRef` can expose the same narrow dependencies to Rust handlers
- Do not prescribe `ArcSwap` before defining publication, capture and retirement semantics

## Framework references

- [Starlette lifespan](https://www.starlette.io/lifespan/)
- [Axum state](https://docs.rs/axum/latest/axum/extract/struct.State.html)

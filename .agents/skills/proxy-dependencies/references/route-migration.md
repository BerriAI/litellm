# Route migration

## Existing pattern

- `get_database` reads the lifespan-owned handle from request state
- A feature provider builds its narrow repository from the backend handle, such as `UserBannerRepository(database.prisma)`
- A handler receives `UserBannerStore | None`, never a Prisma client
- `UserBannerRepository` keeps the existing Prisma table, serialization and configuration-sync behavior
- The banner audit provider binds the legacy audit callable at request time, without performing audit I/O
- SQLx can later implement the same typed operations without changing handler signatures

## Timing and errors

- FastAPI resolves dependencies before entering the handler
- Handler exception handling cannot catch failures raised during dependency resolution
- A universal required-database dependency can override authorization and validation behavior
- Banner GET without a database returns the default disabled banner
- Banner PATCH checks the admin role before returning the existing missing-database error
- Invalid stored banner data falls back to the default, but storage failures still propagate
- Banner writes finish before the success audit is queued as a FastAPI background task
- Queue post-response work with `BackgroundTasks`, not bare `asyncio.create_task`, so the in-flight drain waits for it before the database closes
- Do not move background work into an awaited provider merely to remove an import

## Verification

- Exercise real FastAPI handlers using `dependency_overrides` for external boundaries
- Restore overrides after each test
- Verify complete response bodies, status codes and OpenAPI against the base
- Include combined failure conditions to pin error precedence
- Check saved values, fresh revisions, audit contents and failed-write behavior
- Check lifespan state with a running lifespan, not only a TestClient that skips startup
- Verify providers do not connect, query, disconnect or audit eagerly
- Use real PostgreSQL and live proxy requests when changing persistence behavior

## Scope

- Preserve direct Python call sites or migrate them explicitly
- Keep repository extraction, backend switching and error normalization separate
- Do not migrate every route in a large module just to establish the pattern
- Do not invent providers for resources without consumers

## Framework references

- [Dependency composition and caching](https://fastapi.tiangolo.com/tutorial/dependencies/sub-dependencies/)
- [Dependency overrides](https://fastapi.tiangolo.com/advanced/testing-dependencies/)

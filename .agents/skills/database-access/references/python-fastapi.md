# Python and FastAPI

## Existing pattern

- `UserBannerStore` in `litellm/models/user_banner.py` is the Protocol the banner routes call
- `get_database` in `litellm/proxy/deps/database.py` reads the lifespan-owned handle from request state
- `get_user_banner_repository` in `litellm/proxy/deps/repositories.py` builds the banner repository from that handle
- Choose each repository's replica routing once in its provider, never at call sites
- A handler receives `UserBannerStore | None`, never a Prisma client or a concrete repository
- `UserBannerRepository` implements `UserBannerStore` structurally, keeping the existing Prisma table, serialization and configuration-sync behavior
- The banner audit provider binds the legacy audit callable at request time, without performing audit I/O
- A native repository can later implement the same Protocol without changing handler signatures, see [porting a repository to SQLx](python-repository.md#porting-a-repository-to-sqlx)

## Dependency injection

- Add one provider and one `Annotated` alias per repository in `litellm/proxy/deps/repositories.py`
- Annotate the provider and alias with the feature's Protocol, not the concrete repository
- A route depends only on the repositories it calls, so a request never builds repositories it does not use
- Do not build repositories once at lifespan, a reconnect can replace the Prisma client they wrap

```python
class UserBannerStore(Protocol):
    async def get_banner(self) -> UserBanner: ...

    async def save_banner(self, banner: UserBanner) -> None: ...
```

```python
async def get_user_banner_repository(database: DatabaseDep) -> UserBannerStore | None:
    return None if database is None else UserBannerRepository(database.prisma)


UserBannerRepositoryDep: TypeAlias = Annotated[UserBannerStore | None, Depends(get_user_banner_repository)]
```

- FastAPI caches each provider within a request, so routes and sub-dependencies share one instance
- Ordinary helpers and jobs receive explicit arguments, never unresolved `Depends` defaults
- A route still reading the `prisma_client` global or building a repository inline switches to the repository's `...RepositoryDep`, keeping its no-database behavior and error precedence

```python
async def get_user_banner(banners: UserBannerRepositoryDep) -> UserBanner:
    if banners is None:
        return UserBanner()
    return await banners.get_banner()
```

## Lifespan state

- Prisma startup owns connection setup, authentication refresh, health checks and migrations
- `Database` holds the initialized Prisma client and the `NativeDatabase` handle, without holding fields per feature
  - The native handle opens no pool yet, its `connect` and `close` are noops
- The proxy lifespan builds the handle with `Database.open`, registers `Database.aclose` on its exit stack and publishes it in request state
  - Callers only open and close `Database`, which closes Prisma and the native handle together, and `open` closes Prisma itself when the native connect fails
- Starlette shallow-copies lifespan state into request state, which is distinct from `app.state`
- The database module never imports a repository, legacy globals borrow its handle without owning shutdown
- Drain requests, jobs, logging, spend and audit work before database teardown
- The exit stack closes the handle on exit, including startup failures after the client is created
- When adding native resources, initialize them in the serving process, respecting runtime fork restrictions
- Await native shutdown instead of depending on Python garbage collection
- Apply the [connection ownership rules](../SKILL.md#connection-ownership) when changing the owner

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
- Keep production parsing and persistence code real
- Restore overrides after each test
- Verify complete response bodies, status codes and OpenAPI against the base
- Include combined failure conditions to pin error precedence
- Check saved values, fresh revisions, audit contents and failed-write behavior
- Check lifespan state with a running lifespan, not only a TestClient that skips startup
- Verify providers do not connect, query, disconnect or audit eagerly
- Use real PostgreSQL and live proxy requests when changing persistence behavior

## Scope

- Preserve direct Python call sites or migrate them explicitly
- Inventory imports, state writers, side effects and error ordering before migrating a feature
- Preserve configuration sync and serialization, and state which legacy reads remain
- Do not migrate every route in a large module just to establish the pattern
- Do not invent providers for resources without consumers

## Framework references

- [Lifespan events](https://fastapi.tiangolo.com/advanced/events/) and [Starlette lifespan state](https://starlette.dev/lifespan/)
- [`SessionDep`-style `Annotated` aliases](https://fastapi.tiangolo.com/tutorial/sql-databases/#create-a-session-dependency), which `DatabaseDep` and `UserBannerRepositoryDep` follow
- [Dependency composition and caching](https://fastapi.tiangolo.com/tutorial/dependencies/sub-dependencies/)
- [Background tasks and dependencies with `yield`](https://fastapi.tiangolo.com/advanced/advanced-dependencies/#background-tasks-and-dependencies-with-yield-technical-details)
- [Dependency overrides](https://fastapi.tiangolo.com/advanced/testing-dependencies/)
- [Testing lifespan](https://fastapi.tiangolo.com/advanced/testing-events/)

import inspect


async def resolve_awaitable(obj):
    if inspect.isawaitable(obj):
        return await obj

    return obj


async def async_any(items):
    for item in items:
        if await resolve_awaitable(item):
            return True

    return False

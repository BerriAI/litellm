import asyncio
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import fakeredis.aioredis
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.routing import Route

from litellm.proxy.middleware.auto_queue_middleware import (
    AutoQueueMiddleware,
    AutoQueueRedis,
    ModelQueue,
)


@pytest_asyncio.fixture
async def redis():
    client = fakeredis.aioredis.FakeRedis()
    try:
        yield client
    finally:
        await client.flushdb()
        await client.aclose()


@pytest_asyncio.fixture
async def aqr_factory(redis):
    def _factory(
        *,
        default_max_concurrent: int = 2,
        ceiling: int = 10,
        scale_up_threshold: int = 3,
        scale_down_step: int = 1,
    ):
        return AutoQueueRedis(
            redis=redis,
            default_max_concurrent=default_max_concurrent,
            ceiling=ceiling,
            scale_up_threshold=scale_up_threshold,
            scale_down_step=scale_down_step,
        )

    return _factory


@pytest.fixture
def queue_factory():
    def _factory(*, max_depth: int = 5):
        return ModelQueue(max_depth=max_depth)

    return _factory


@pytest.fixture
def make_middleware_app():
    def _factory(
        handler: Callable[..., Any],
        *,
        aqr,
        enabled: bool = True,
        max_queue_depth: int = 100,
        extra_routes: Iterable[Route] | None = None,
    ):
        routes = [Route("/v1/chat/completions", handler, methods=["POST"])]
        if extra_routes:
            routes.extend(extra_routes)
        app = Starlette(routes=routes)
        return AutoQueueMiddleware(
            app,
            aqr=aqr,
            enabled=enabled,
            max_queue_depth=max_queue_depth,
        )

    return _factory


@pytest_asyncio.fixture
async def asgi_client_factory():
    clients: list[AsyncClient] = []

    async def _factory(app):
        client = AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        )
        clients.append(client)
        return client

    try:
        yield _factory
    finally:
        for client in clients:
            await client.aclose()


@pytest_asyncio.fixture
async def drive_asgi():
    async def _drive(app, *, scope, messages):
        sent: list[dict[str, Any]] = []
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        for message in messages:
            await queue.put(message)

        async def receive():
            return await queue.get()

        async def send(message):
            sent.append(message)

        await app(scope, receive, send)
        return sent

    return _drive


@pytest_asyncio.fixture
async def eventually():
    import inspect

    async def _eventually(predicate, *, timeout: float = 2.0, interval: float = 0.01):
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        last_value = None
        while loop.time() < deadline:
            last_value = predicate()
            if inspect.isawaitable(last_value):
                last_value = await last_value
            if last_value:
                return last_value
            await asyncio.sleep(interval)
        raise AssertionError(f"Condition not met within {timeout}s; last_value={last_value!r}")

    return _eventually


@pytest.fixture
def key_metadata_factory(monkeypatch):
    import litellm.proxy._types as proxy_types
    import litellm.proxy.proxy_server as proxy_server

    def _factory(
        *,
        timeout: int | float | None = None,
        priority: int | None = None,
        explode: bool = False,
    ) -> None:
        if explode:
            def _boom(*args, **kwargs):
                raise RuntimeError("metadata lookup failed")

            monkeypatch.setattr(proxy_types, "hash_token", _boom)
            return

        metadata: dict[str, Any] = {}
        if timeout is not None:
            metadata["queue_timeout_seconds"] = timeout
        if priority is not None:
            metadata["queue_priority"] = priority

        fake_key = SimpleNamespace(metadata=metadata)
        monkeypatch.setattr(proxy_types, "hash_token", lambda token: f"hashed::{token}")
        monkeypatch.setattr(
            proxy_server,
            "user_api_key_cache",
            SimpleNamespace(get_cache=lambda key: fake_key),
        )

    return _factory

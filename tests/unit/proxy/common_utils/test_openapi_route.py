import asyncio
import itertools
import threading
from _collections_abc import dict_items
from collections.abc import Callable
from typing import Final

import httpx
import pytest
from fastapi import FastAPI

from litellm.proxy.common_utils.custom_openapi_spec import JsonValue
from litellm.proxy.common_utils.openapi_route import OpenAPISchema, serve_openapi_off_the_event_loop

UNBLOCK_TIMEOUT_SECONDS: Final = 5


def _route_paths(app: FastAPI) -> Callable[[], tuple[str, ...]]:
    return lambda: tuple(str(getattr(route, "path", "")) for route in app.routes)


def _client(app: FastAPI, root_path: str = "") -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app, root_path=root_path), base_url="http://proxy")


async def _let_other_tasks_run() -> None:
    for _ in range(50):
        await asyncio.sleep(0)


async def test_other_requests_are_answered_while_the_schema_builds():
    app: Final = FastAPI()
    app.add_api_route("/ping", lambda: "pong")
    build_started: Final = threading.Event()
    ping_answered: Final = threading.Event()

    def build(paths: tuple[str, ...]) -> OpenAPISchema:
        build_started.set()
        return {"ping_answered_during_build": ping_answered.wait(UNBLOCK_TIMEOUT_SECONDS)}

    serve_openapi_off_the_event_loop(app, _route_paths(app), build)
    async with _client(app) as client:
        schema_request: Final = asyncio.ensure_future(client.get("/openapi.json"))
        await asyncio.to_thread(build_started.wait, UNBLOCK_TIMEOUT_SECONDS)
        ping: Final = await client.get("/ping")
        ping_answered.set()
        schema: Final = await schema_request

    assert (ping.json(), schema.json()) == ("pong", {"ping_answered_during_build": True})


async def test_a_route_added_during_the_build_is_in_the_next_schema():
    app: Final = FastAPI()
    build_started: Final = threading.Event()
    may_finish: Final = threading.Event()

    def build(paths: tuple[str, ...]) -> OpenAPISchema:
        build_started.set()
        may_finish.wait(UNBLOCK_TIMEOUT_SECONDS)
        return {"paths": [*paths]}

    serve_openapi_off_the_event_loop(app, _route_paths(app), build)
    async with _client(app) as client:
        first_request: Final = asyncio.ensure_future(client.get("/openapi.json"))
        await asyncio.to_thread(build_started.wait, UNBLOCK_TIMEOUT_SECONDS)
        app.add_api_route("/registered-mid-build", lambda: "late")
        may_finish.set()
        first: Final = await first_request
        second: Final = await client.get("/openapi.json")

    assert ("/registered-mid-build" in first.json()["paths"], "/registered-mid-build" in second.json()["paths"]) == (
        False,
        True,
    )


async def test_concurrent_cold_requests_share_one_build():
    app: Final = FastAPI()
    build_numbers: Final = itertools.count()
    build_started: Final = threading.Event()
    may_finish: Final = threading.Event()

    def build(paths: tuple[str, ...]) -> OpenAPISchema:
        build_started.set()
        may_finish.wait(UNBLOCK_TIMEOUT_SECONDS)
        return {"build": next(build_numbers)}

    serve_openapi_off_the_event_loop(app, _route_paths(app), build)
    async with _client(app) as client:
        first_request: Final = asyncio.ensure_future(client.get("/openapi.json"))
        await asyncio.to_thread(build_started.wait, UNBLOCK_TIMEOUT_SECONDS)
        second_request: Final = asyncio.ensure_future(client.get("/openapi.json"))
        await _let_other_tasks_run()
        may_finish.set()
        responses: Final = await asyncio.gather(first_request, second_request)

    assert [response.json() for response in responses] == [{"build": 0}, {"build": 0}]


async def test_warm_requests_reuse_the_rendered_schema():
    app: Final = FastAPI()
    renders: Final = itertools.count()

    class RenderCountingSchema(dict[str, JsonValue]):
        def items(self) -> "dict_items[str, JsonValue]":
            _ = next(renders)
            return super().items()

    serve_openapi_off_the_event_loop(app, _route_paths(app), lambda paths: RenderCountingSchema(paths=[*paths]))
    async with _client(app) as client:
        bodies: Final = [(await client.get("/openapi.json")).content for _ in range(3)]

    assert (len(set(bodies)), next(renders)) == (1, 1)


async def test_a_rebuilt_schema_is_served_instead_of_the_previous_one():
    app: Final = FastAPI()
    builds: Final = itertools.count()
    serve_openapi_off_the_event_loop(app, _route_paths(app), lambda paths: {"build": next(builds)})

    async with _client(app) as client:
        first: Final = await client.get("/openapi.json")
        app.openapi_schema = None
        second: Final = await client.get("/openapi.json")

    assert (first.json(), second.json()) == ({"build": 0}, {"build": 1})



async def test_an_app_that_replaces_its_openapi_function_serves_that_schema():
    app: Final = FastAPI()
    serve_openapi_off_the_event_loop(app, _route_paths(app), lambda paths: {"built_by": "the proxy"})
    app.openapi = lambda: {"built_by": "the operator"}

    async with _client(app) as client:
        response: Final = await client.get("/openapi.json")

    assert response.json() == {"built_by": "the operator"}

@pytest.mark.parametrize(
    ("servers", "expected"),
    [
        ([{"url": "https://docs.example"}], [{"url": "/api"}, {"url": "https://docs.example"}]),
        ([{"url": "/api"}], [{"url": "/api"}]),
    ],
)
async def test_the_root_path_is_listed_first_among_the_servers(servers: JsonValue, expected: JsonValue):
    app: Final = FastAPI()
    serve_openapi_off_the_event_loop(app, _route_paths(app), lambda paths: {"servers": servers})

    async with _client(app, root_path="/api") as client:
        response: Final = await client.get("/openapi.json")

    assert response.json()["servers"] == expected

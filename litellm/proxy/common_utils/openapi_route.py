import asyncio
import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Final, TypeAlias, TypeVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import TypeAdapter
from starlette.routing import BaseRoute, Route

from litellm.proxy.common_utils.custom_openapi_spec import JsonObject

OpenAPISchema: TypeAlias = JsonObject
Inputs: Final = TypeVar("Inputs")
_SERVERS: Final = TypeAdapter(list[dict[str, object]])


@dataclass(frozen=True, slots=True, eq=False)
class _SameSchema:
    schema: Mapping[str, object]

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _SameSchema) and other.schema is self.schema

    def __hash__(self) -> int:
        return id(self.schema)


def serve_openapi_off_the_event_loop(
    app: FastAPI,
    snapshot: Callable[[], Inputs],
    build: Callable[[Inputs], OpenAPISchema],
) -> None:
    openapi_url: Final = app.openapi_url
    if openapi_url is None:
        return
    build_lock: Final = asyncio.Lock()
    installed_openapi: Final = app.openapi

    async def built_schema() -> OpenAPISchema:
        async with build_lock:
            if app.openapi_schema:
                return app.openapi_schema
            inputs: Final = snapshot()
            schema: Final = await asyncio.to_thread(build, inputs)
            if app.openapi_schema is None and snapshot() == inputs:
                app.openapi_schema = schema
            return schema

    async def served_schema() -> OpenAPISchema:
        if app.openapi != installed_openapi:
            return app.openapi()
        return app.openapi_schema or await built_schema()

    async def openapi(request: Request) -> Response:
        schema: Final = await served_schema()
        scope_root_path: Final[object] = request.scope.get("root_path")
        root_path: Final = scope_root_path.rstrip("/") if isinstance(scope_root_path, str) else ""
        servers_root_path: Final = root_path if app.root_path_in_servers else ""
        return Response(_rendered(_SameSchema(schema), servers_root_path), media_type="application/json")

    app.router.routes[:] = [  # rebind-ok: the app owns its route table
        Route(openapi_url, openapi, include_in_schema=False) if _serves(route, openapi_url) else route
        for route in app.router.routes
    ]


@functools.lru_cache(maxsize=1)
def _rendered(served: _SameSchema, root_path: str) -> bytes | memoryview:
    return JSONResponse(_with_root_path_server(served.schema, root_path) if root_path else served.schema).body


def _with_root_path_server(schema: Mapping[str, object], root_path: str) -> Mapping[str, object]:
    servers: Final = _SERVERS.validate_python(schema.get("servers", []))
    if any(server.get("url") == root_path for server in servers):
        return schema
    return {**schema, "servers": [{"url": root_path}, *servers]}


def _serves(route: BaseRoute, path: str) -> bool:
    return isinstance(route, Route) and route.path == path

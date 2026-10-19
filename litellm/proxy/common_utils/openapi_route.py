import asyncio
from collections.abc import Callable, Mapping
from typing import Final, TypeAlias, TypeVar

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter
from starlette.routing import BaseRoute, Route

from litellm.proxy.common_utils.custom_openapi_spec import JsonObject

OpenAPISchema: TypeAlias = JsonObject
Inputs: Final = TypeVar("Inputs")
_SERVERS: Final = TypeAdapter(list[dict[str, object]])


def serve_openapi_off_the_event_loop(
    app: FastAPI,
    snapshot: Callable[[], Inputs],
    build: Callable[[Inputs], OpenAPISchema],
) -> None:
    openapi_url: Final = app.openapi_url
    if openapi_url is None:
        return
    build_lock: Final = asyncio.Lock()

    async def built_schema() -> OpenAPISchema:
        async with build_lock:
            if app.openapi_schema:
                return app.openapi_schema
            inputs: Final = snapshot()
            schema: Final = await asyncio.to_thread(build, inputs)
            if app.openapi_schema is None and snapshot() == inputs:
                app.openapi_schema = schema
            return schema

    async def openapi(request: Request) -> JSONResponse:
        schema: Final = app.openapi_schema or await built_schema()
        scope_root_path: Final[object] = request.scope.get("root_path")
        root_path: Final = scope_root_path.rstrip("/") if isinstance(scope_root_path, str) else ""
        if root_path and app.root_path_in_servers:
            return JSONResponse(_with_root_path_server(schema, root_path))
        return JSONResponse(schema)

    app.router.routes[:] = [  # rebind-ok: the app owns its route table
        Route(openapi_url, openapi, include_in_schema=False) if _serves(route, openapi_url) else route
        for route in app.router.routes
    ]


def _with_root_path_server(schema: Mapping[str, object], root_path: str) -> Mapping[str, object]:
    servers: Final = _SERVERS.validate_python(schema.get("servers", []))
    if any(server.get("url") == root_path for server in servers):
        return schema
    return {**schema, "servers": [{"url": root_path}, *servers]}


def _serves(route: BaseRoute, path: str) -> bool:
    return isinstance(route, Route) and route.path == path

from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
from fastapi.routing import APIRoute

from litellm.proxy._lazy_openapi_snapshot import _routes_with_stable_unique_ids


def test_routes_with_stable_unique_ids_splits_multi_method_routes() -> None:
    app = FastAPI()

    async def proxy_route() -> dict:
        return {}

    app.add_api_route(
        "/proxy/{endpoint:path}",
        proxy_route,
        methods=["GET", "POST", "DELETE"],
    )

    routes = [route for route in app.routes if isinstance(route, APIRoute)]
    stable_routes = _routes_with_stable_unique_ids(routes)

    assert [route.methods for route in stable_routes] == [
        {"DELETE"},
        {"GET"},
        {"POST"},
    ]

    openapi = get_openapi(title="test", version="1", routes=stable_routes)
    path_ops = openapi["paths"]["/proxy/{endpoint}"]

    operation_ids = {
        method: operation["operationId"] for method, operation in path_ops.items()
    }
    assert operation_ids == {
        "delete": "proxy_route_proxy__endpoint__delete",
        "get": "proxy_route_proxy__endpoint__get",
        "post": "proxy_route_proxy__endpoint__post",
    }

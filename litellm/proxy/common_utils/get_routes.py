"""
Utility class for getting routes from a FastAPI app.
"""

from collections.abc import Sequence
from typing import Final, Protocol

from starlette.routing import BaseRoute
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_logger


class NamedEndpoint(Protocol):
    __name__: str


class RouteInfo(TypedDict):
    path: ReadOnly[str | None]
    methods: ReadOnly[Sequence[str] | None]
    name: ReadOnly[str | None]
    endpoint: ReadOnly[str | None]
    mounted_app: NotRequired[ReadOnly[bool]]


class GetRoutes:
    @staticmethod
    def get_app_routes(
        route: BaseRoute,
        endpoint_route: NamedEndpoint,
    ) -> list[RouteInfo]:
        """
        Get routes for a regular route.
        """
        route_info: Final[RouteInfo] = {
            "path": getattr(route, "path", None),
            "methods": getattr(route, "methods", None),
            "name": getattr(route, "name", None),
            "endpoint": (endpoint_route.__name__ if getattr(route, "endpoint", None) else None),
        }
        return [route_info]

    @staticmethod
    def get_routes_for_mounted_app(
        route: BaseRoute,
    ) -> list[RouteInfo]:
        """
        Get routes for a mounted sub-application.
        """
        mount_path: Final[str] = getattr(route, "path", "")
        sub_app: Final[object] = getattr(route, "app", None)
        if not sub_app or not hasattr(sub_app, "routes"):
            return []
        sub_routes: Final[Sequence[object]] = getattr(sub_app, "routes", ())
        return [
            sub_route_info
            for sub_route in sub_routes
            if (sub_route_info := GetRoutes._mounted_sub_route_info(mount_path, sub_route)) is not None
        ]

    @staticmethod
    def _mounted_sub_route_info(mount_path: str, sub_route: object) -> RouteInfo | None:
        endpoint_func: Final[object] = getattr(sub_route, "endpoint", None) or getattr(sub_route, "app", None)
        if endpoint_func is None:
            return None
        sub_route_path: Final[str] = getattr(sub_route, "path", "")
        return {
            "path": mount_path.rstrip("/") + sub_route_path,
            "methods": getattr(sub_route, "methods", ["GET", "POST"]),
            "name": getattr(sub_route, "name", None),
            "endpoint": GetRoutes._safe_get_endpoint_name(endpoint_func),
            "mounted_app": True,
        }

    @staticmethod
    def _safe_get_endpoint_name(endpoint_function: object) -> str | None:
        """
        Safely get the name of the endpoint function.
        """
        try:
            if hasattr(endpoint_function, "__name__"):
                endpoint_name: Final[str] = getattr(endpoint_function, "__name__", "")
                return endpoint_name
            elif hasattr(endpoint_function, "__class__") and hasattr(endpoint_function.__class__, "__name__"):
                return endpoint_function.__class__.__name__
            else:
                return None
        except Exception:
            verbose_logger.exception("Error getting endpoint name for route: %s", endpoint_function)
            return None

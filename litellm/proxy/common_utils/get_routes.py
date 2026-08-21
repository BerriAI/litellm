"""
Utility class for getting routes from a FastAPI app.
"""

from collections.abc import Sequence
from typing import Any, Final

from starlette.routing import BaseRoute
from typing_extensions import ReadOnly, TypedDict

from litellm._logging import verbose_logger


class RouteInfo(TypedDict, total=False):
    """One entry of the app's route listing."""

    path: ReadOnly[object]
    methods: ReadOnly[object]
    name: ReadOnly[object]
    endpoint: ReadOnly[str | None]
    mounted_app: ReadOnly[bool]


class GetRoutes:
    @staticmethod
    def get_app_routes(
        route: BaseRoute,
        endpoint_route: Any,
    ) -> Sequence[RouteInfo]:
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
    ) -> Sequence[RouteInfo]:
        """
        Get routes for a mounted sub-application.
        """
        routes: Final[list[RouteInfo]] = []
        mount_path: Final = getattr(route, "path", "")
        for sub_route in GetRoutes._mounted_app_routes(route):
            endpoint_func: object = getattr(sub_route, "endpoint", None) or getattr(sub_route, "app", None)

            if endpoint_func is not None:
                sub_route_path = getattr(sub_route, "path", "")
                full_path = mount_path.rstrip("/") + sub_route_path

                route_info: RouteInfo = {
                    "path": full_path,
                    "methods": getattr(sub_route, "methods", ["GET", "POST"]),
                    "name": getattr(sub_route, "name", None),
                    "endpoint": GetRoutes._safe_get_endpoint_name(endpoint_func),
                    "mounted_app": True,
                }
                routes.append(route_info)
        return routes

    @staticmethod
    def _mounted_app_routes(route: BaseRoute) -> Sequence[BaseRoute]:
        """The routes of the sub-application mounted at ``route``, if it mounts one."""
        sub_app: Final[object] = getattr(route, "app", None)
        if sub_app and hasattr(sub_app, "routes"):
            return getattr(sub_app, "routes")
        return ()

    @staticmethod
    def _safe_get_endpoint_name(endpoint_function: object) -> str | None:
        """
        Safely get the name of the endpoint function.
        """
        try:
            if hasattr(endpoint_function, "__name__"):
                return getattr(endpoint_function, "__name__")
            elif hasattr(endpoint_function, "__class__") and hasattr(endpoint_function.__class__, "__name__"):
                return endpoint_function.__class__.__name__
            else:
                return None
        except Exception:
            verbose_logger.exception("Error getting endpoint name for route: %s", endpoint_function)
            return None

"""
Utility class for getting routes from a FastAPI app.
"""

from typing import Any, Final

from starlette.routing import BaseRoute

from litellm._logging import verbose_logger


class GetRoutes:
    @staticmethod
    def get_app_routes(
        route: BaseRoute,
        endpoint_route: Any,
    ) -> list[dict[str, Any]]:
        """
        Get routes for a regular route.
        """
        routes: Final[list[dict[str, Any]]] = []
        route_info: Final = {
            "path": getattr(route, "path", None),
            "methods": getattr(route, "methods", None),
            "name": getattr(route, "name", None),
            "endpoint": (endpoint_route.__name__ if getattr(route, "endpoint", None) else None),
        }
        routes.append(route_info)
        return routes

    @staticmethod
    def get_routes_for_mounted_app(
        route: BaseRoute,
    ) -> list[dict[str, Any]]:
        """
        Get routes for a mounted sub-application.
        """
        routes: Final[list[dict[str, Any]]] = []
        mount_path: Final = getattr(route, "path", "")
        sub_app: Final = getattr(route, "app", None)
        if sub_app and hasattr(sub_app, "routes"):
            for sub_route in sub_app.routes:
                # Get endpoint - either from endpoint attribute or app attribute
                endpoint_func = getattr(sub_route, "endpoint", None) or getattr(sub_route, "app", None)

                if endpoint_func is not None:
                    sub_route_path = getattr(sub_route, "path", "")
                    full_path = mount_path.rstrip("/") + sub_route_path

                    route_info = {
                        "path": full_path,
                        "methods": getattr(sub_route, "methods", ["GET", "POST"]),
                        "name": getattr(sub_route, "name", None),
                        "endpoint": GetRoutes._safe_get_endpoint_name(endpoint_func),
                        "mounted_app": True,
                    }
                    routes.append(route_info)
        return routes

    @staticmethod
    def _safe_get_endpoint_name(endpoint_function: Any) -> str | None:
        """
        Safely get the name of the endpoint function.
        """
        try:
            if hasattr(endpoint_function, "__name__"):
                return getattr(endpoint_function, "__name__")
            elif hasattr(endpoint_function, "__class__") and hasattr(endpoint_function.__class__, "__name__"):
                return getattr(endpoint_function.__class__, "__name__")
            else:
                return None
        except Exception:
            verbose_logger.exception("Error getting endpoint name for route: %s", endpoint_function)
            return None

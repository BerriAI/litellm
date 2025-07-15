"""
Utility class for getting routes from a FastAPI app.
"""

from typing import Any, Dict, List

from starlette.routing import BaseRoute


class GetRoutes:
    @staticmethod
    def get_app_routes(
        route: BaseRoute,
        endpoint_route: Any,
    ) -> List[Dict[str, Any]]:
        """
        Get routes for a regular route.
        """
        routes: List[Dict[str, Any]] = []
        route_info = {
            "path": getattr(route, "path", None),
            "methods": getattr(route, "methods", None),
            "name": getattr(route, "name", None),
            "endpoint": (
                endpoint_route.__name__
                if getattr(route, "endpoint", None)
                else None
            ),
        }
        routes.append(route_info)
        return routes
    
    @staticmethod
    def get_routes_for_mounted_app(
        route: BaseRoute,
    ) -> List[Dict[str, Any]]:
        """
        Get routes for a mounted sub-application.
        """
        routes: List[Dict[str, Any]] = []
        mount_path = getattr(route, 'path', '')
        sub_app = getattr(route, 'app', None)
        if sub_app and hasattr(sub_app, 'routes'):
            for sub_route in sub_app.routes:
                # Get endpoint - either from endpoint attribute or app attribute
                endpoint_func = getattr(sub_route, "endpoint", None) or getattr(sub_route, "app", None)
                
                if endpoint_func is not None:
                    sub_route_path = getattr(sub_route, "path", "")
                    full_path = mount_path.rstrip('/') + sub_route_path
                    
                    route_info = {
                        "path": full_path,
                        "methods": getattr(sub_route, "methods", ["GET", "POST"]),
                        "name": getattr(sub_route, "name", None),
                        "endpoint": endpoint_func.__name__ if callable(endpoint_func) else None,
                        "mounted_app": True,
                    }
                    routes.append(route_info)
        return routes
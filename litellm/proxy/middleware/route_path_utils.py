"""Shared ASGI scope helper for middleware that needs the request path Starlette
will actually route on, i.e. with any ``SERVER_ROOT_PATH``/``root_path`` prefix
stripped the same way Starlette's own router does before matching."""

from typing import Final

from starlette.types import Scope


def get_route_path(scope: Scope) -> str:
    path: Final[str] = scope["path"]
    root_path: Final[str] = scope.get("root_path", "")
    if not root_path or not path.startswith(root_path):
        return path
    if path == root_path:
        return ""
    if path[len(root_path)] == "/":
        return path[len(root_path) :]
    return path

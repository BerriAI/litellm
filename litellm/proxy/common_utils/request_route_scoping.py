"""
Pure route predicates shared by the request-size limit's two enforcement layers.

The ASGI middleware sees a raw scope and the auth layer sees a Starlette
Request, so neither can reuse the other's helpers. Both need the same answers
about a path, so they live here with no framework or proxy dependencies.
"""

import re
from typing import Final

_FILE_UPLOAD_ROUTE_PATTERN: Final = re.compile(r"^(?:/files|(?:/[^/]+)?/v1/files)$")


def strip_root_path(path: str, root_path: str) -> str:
    """
    Normalize a sub-path deployment, e.g. ``/genai/v1/files`` -> ``/v1/files``.

    Strips only on whole segment boundaries, so a sibling path like ``/apifoo``
    is left intact under ``root_path="/api"``.
    """
    normalized_root: Final = root_path.rstrip("/")
    if normalized_root and (path == normalized_root or path.startswith(normalized_root + "/")):
        return path[len(normalized_root) :] or "/"
    return path


def is_file_upload_route(method: str, route: str) -> bool:
    """
    True for the routes that upload a file body: ``POST`` to ``/files``,
    ``/v1/files``, or ``/{provider}/v1/files``.

    Matches the paths ``create_file`` is mounted on, and deliberately not
    ``/v1/vector_stores/{vector_store_id}/files``, which takes a JSON body
    referencing an already-uploaded file.
    """
    return method.upper() == "POST" and _FILE_UPLOAD_ROUTE_PATTERN.match(route) is not None

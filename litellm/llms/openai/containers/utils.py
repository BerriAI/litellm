"""Shared helpers for OpenAI-compatible container API URL construction."""

import httpx


def join_container_api_base_path(api_base: str, path_suffix: str) -> str:
    """Append ``path_suffix`` to the path of ``api_base``, keeping the query string last.

    Azure (and some bases) pass ``api_base`` like
    ``https://host/openai/v1/containers?api-version=v1``. Naive string concat would
    produce ``...?api-version=v1/cntr_...`` which is invalid; this uses ``httpx.URL``
    so the result is ``.../containers/cntr_.../files?api-version=v1``.
    """
    if not path_suffix.startswith("/"):
        path_suffix = f"/{path_suffix}"
    parsed = httpx.URL(api_base)
    new_path = f"{parsed.path.rstrip('/')}{path_suffix}"
    return str(parsed.copy_with(path=new_path))

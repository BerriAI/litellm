"""Request parsing helpers for LiteLLM container proxy endpoints."""

from typing import Any, Dict

from fastapi import HTTPException, Request


def get_container_list_query_params(request: Request) -> Dict[str, Any]:
    """Return validated OpenAI-compatible container list query parameters.

    The container SDK functions accept ``after``, ``limit`` and ``order`` as
    top-level keyword arguments. Proxy handlers must not pass them inside a
    nested ``query_params`` object, because the SDK silently ignores that
    object.
    """

    data: Dict[str, Any] = {}

    after = request.query_params.get("after")
    if after is not None:
        data["after"] = after

    limit_value = request.query_params.get("limit")
    if limit_value is not None:
        try:
            limit = int(limit_value)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="limit must be an integer") from exc
        if not 1 <= limit <= 100:
            raise HTTPException(status_code=422, detail="limit must be between 1 and 100")
        data["limit"] = limit

    order = request.query_params.get("order")
    if order is not None:
        if order not in {"asc", "desc"}:
            raise HTTPException(status_code=422, detail="order must be 'asc' or 'desc'")
        data["order"] = order

    return data

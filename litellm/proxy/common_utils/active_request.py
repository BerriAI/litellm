from contextvars import ContextVar
from typing import Final

from fastapi import Request

active_request: Final[ContextVar[Request | None]] = ContextVar("litellm_active_request", default=None)


async def active_request_disconnected() -> bool:
    request: Final = active_request.get()
    if request is None:
        return False
    try:
        return await request.is_disconnected()
    except Exception:  # noqa: BLE001  # disconnect probing must never fail the request
        return False

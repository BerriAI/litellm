"""Startup hooks for configured callback instances."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from fastapi import FastAPI


def setup_proxy_callbacks(app: "FastAPI", callbacks: Iterable[object]) -> None:
    """Run each configured logger once; a failed setup must fail startup."""
    seen: set[int] = set()
    for callback in tuple(callbacks):
        if isinstance(callback, CustomLogger) and id(callback) not in seen:
            seen.add(id(callback))
            callback.setup_proxy(app)
    # A hook may have added routes after another startup component built OpenAPI.
    app.openapi_schema = None


async def close_proxy_callbacks(callbacks: Iterable[object]) -> None:
    """One failing callback must not prevent the remaining cleanup hooks."""
    seen: set[int] = set()
    for callback in tuple(callbacks):
        if isinstance(callback, CustomLogger) and id(callback) not in seen:
            seen.add(id(callback))
            try:
                await callback.async_close()
            except Exception:
                verbose_proxy_logger.exception("Error closing callback %s", type(callback).__name__)

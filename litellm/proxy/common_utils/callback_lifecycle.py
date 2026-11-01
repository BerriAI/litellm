"""Startup hooks for configured callback instances."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:
    from fastapi import FastAPI


def setup_proxy_callbacks(app: "FastAPI", callbacks: Iterable[object]) -> None:
    """Run each configured logger once; a failed setup must fail startup."""
    for callback in _unique_callbacks(callbacks):
        callback.setup_proxy(app)
    app.openapi_schema = None  # rebind-ok: routes added by startup hooks invalidate FastAPI's cached schema


async def close_proxy_callbacks(callbacks: Iterable[object]) -> None:
    """One failing callback must not prevent the remaining cleanup hooks."""
    for callback in _unique_callbacks(callbacks):
        try:
            await callback.async_close()
        except Exception:  # noqa: BLE001  # Third-party cleanup failures must not skip other callbacks
            verbose_proxy_logger.exception("Error closing callback %s", type(callback).__name__)


def _unique_callbacks(callbacks: Iterable[object]) -> tuple[CustomLogger, ...]:
    return tuple({id(callback): callback for callback in callbacks if isinstance(callback, CustomLogger)}.values())

"""Startup and shutdown hooks for the callbacks the proxy is configured with."""

import asyncio
from collections.abc import Iterable
from typing import Final

from fastapi import APIRouter, Depends, FastAPI
from fastapi.routing import APIRoute

from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

CALLBACK_CLOSE_TIMEOUT_SECONDS: Final = 5.0


def _custom_loggers(callbacks: Iterable[object]) -> tuple[CustomLogger, ...]:
    return tuple({id(callback): callback for callback in callbacks if isinstance(callback, CustomLogger)}.values())


def _key_auth_gap(callback: CustomLogger, router: APIRouter) -> str | None:
    unprotected: Final = tuple(route for route in router.routes if not isinstance(route, APIRoute))
    if not unprotected:
        return None
    return (
        f"Callback {type(callback).__name__} returned routes that cannot carry LiteLLM key auth: "
        f"{unprotected}. Register HTTP routes with add_api_route or the router decorators."
    )


async def _setup_proxy_callback(app: FastAPI, callback: CustomLogger) -> None:
    router: Final = await callback.async_setup_proxy()
    if router is None:
        return
    gap: Final = _key_auth_gap(callback, router)
    if gap is not None:
        raise ValueError(gap)
    app.include_router(router, dependencies=[Depends(user_api_key_auth)])
    app.openapi_schema = None  # rebind-ok: FastAPI caches the schema, so new routes must invalidate it


async def setup_proxy_callbacks(app: FastAPI, callbacks: Iterable[object]) -> None:
    for callback in _custom_loggers(callbacks):
        await _setup_proxy_callback(app, callback)


async def _close_proxy_callback(callback: CustomLogger, timeout: float) -> None:
    try:
        await asyncio.wait_for(callback.async_close_proxy(), timeout=timeout)
    except asyncio.TimeoutError:
        verbose_proxy_logger.warning(
            "Callback %s did not finish closing within %.1fs", type(callback).__name__, timeout
        )
    except Exception:  # noqa: BLE001  # one callback's cleanup failure must not skip the others
        verbose_proxy_logger.exception("Error closing callback %s", type(callback).__name__)


async def close_proxy_callbacks(callbacks: Iterable[object], timeout: float = CALLBACK_CLOSE_TIMEOUT_SECONDS) -> None:
    await asyncio.gather(*(_close_proxy_callback(callback, timeout) for callback in _custom_loggers(callbacks)))

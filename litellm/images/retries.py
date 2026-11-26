import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

import httpx

import litellm
from litellm.utils import calculate_retry_after, should_retry

_T = TypeVar("_T")


def _retry_delay(error: Exception, retries_left: int, max_retries: int) -> float | None:
    if isinstance(error, httpx.HTTPStatusError):
        if not should_retry(error.response.status_code):
            return None
        return calculate_retry_after(retries_left, max_retries, error.response.headers)
    if isinstance(error, litellm.Timeout | httpx.TransportError):
        return calculate_retry_after(retries_left, max_retries)
    return None


def retried(send: Callable[[], _T], max_retries: int) -> Callable[[], _T]:
    def send_with_retries() -> _T:
        for retries_left in range(max_retries, 0, -1):
            try:
                return send()
            except Exception as error:
                if (delay := _retry_delay(error, retries_left, max_retries)) is None:
                    raise
                time.sleep(delay)
        return send()

    return send_with_retries


def async_retried(send: Callable[[], Awaitable[_T]], max_retries: int) -> Callable[[], Awaitable[_T]]:
    async def send_with_retries() -> _T:
        for retries_left in range(max_retries, 0, -1):
            try:
                return await send()
            except Exception as error:
                if (delay := _retry_delay(error, retries_left, max_retries)) is None:
                    raise
                await asyncio.sleep(delay)
        return await send()

    return send_with_retries

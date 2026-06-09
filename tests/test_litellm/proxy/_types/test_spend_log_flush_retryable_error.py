from __future__ import annotations

import asyncio

import httpx
import prisma.errors
import pytest

from litellm.proxy._types import is_spend_log_flush_retryable_error


@pytest.mark.parametrize(
    "error,expected",
    [
        (httpx.ConnectError("connect"), True),
        (httpx.ReadError("read"), True),
        (httpx.ReadTimeout("read timeout"), True),
        (httpx.PoolTimeout("pool timeout"), True),
        (asyncio.TimeoutError(), True),
        (prisma.errors.PrismaError("deadlock detected"), True),
        (prisma.errors.PrismaError("pool timeout waiting for connection"), True),
        (ValueError("bad data"), False),
    ],
)
def test_is_spend_log_flush_retryable_error(error: Exception, expected: bool) -> None:
    assert is_spend_log_flush_retryable_error(error) is expected

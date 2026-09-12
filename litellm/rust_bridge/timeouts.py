"""Shared timeout conversion for the native Rust bridges."""

from __future__ import annotations

import httpx


def timeout_to_seconds(timeout: float | httpx.Timeout | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, httpx.Timeout):
        return timeout.read
    return float(timeout)

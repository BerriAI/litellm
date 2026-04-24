"""Shared HTTP transport for the Mavvrik integration.

Provides a single async function with retry and exponential backoff used
by both Client (Mavvrik API calls) and Uploader (GCS calls).

  http_request(method, url, *, headers, json, params, content, timeout, label)
    → httpx.Response

Retry behaviour:
  - 5xx responses and network errors: retry up to MAX_RETRIES times
  - 4xx responses: returned immediately (client-side error, no retry)
  - Backoff: RETRY_BACKOFF_BASE * 2^attempt seconds between retries
"""

import asyncio
from typing import Any, Dict, Optional

import httpx

from litellm._logging import verbose_proxy_logger

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 1.0  # seconds; doubles each retry


async def http_request(
    method: str,
    url: str,
    *,
    headers: Optional[Dict[str, str]] = None,
    json: Optional[Any] = None,
    params: Optional[Dict[str, str]] = None,
    content: Optional[bytes] = None,
    timeout: float = 30.0,
    label: str = "",
) -> httpx.Response:
    """Execute an HTTP request with retry and exponential backoff.

    Args:
        method:  HTTP verb — GET, POST, PUT, PATCH, etc.
        url:     Full URL.
        headers: Request headers.
        json:    JSON-serialisable body (mutually exclusive with content).
        params:  URL query parameters.
        content: Raw bytes body (mutually exclusive with json).
        timeout: Per-request timeout in seconds.
        label:   Short name used in log and error messages (e.g. "register",
                 "initiate"). Falls back to method when empty.

    Returns:
        httpx.Response. Callers check the status themselves.

    Raises:
        RuntimeError: after MAX_RETRIES failed attempts on 5xx or network errors.
    """
    tag = label or method
    last_exc: Exception = RuntimeError("unknown error")

    async with httpx.AsyncClient() as http:
        for attempt in range(_MAX_RETRIES):
            try:
                resp = await http.request(
                    method,
                    url,
                    headers=headers,
                    json=json,
                    params=params,
                    content=content,
                    timeout=timeout,
                )

                if resp.status_code < 500:
                    return resp  # success or 4xx — return immediately, no retry

                last_exc = RuntimeError(
                    f"{tag} failed: {resp.status_code} {resp.text[:200]}"
                )

            except httpx.RequestError as exc:
                last_exc = exc

            if attempt < _MAX_RETRIES - 1:
                wait = _RETRY_BACKOFF_BASE * (2**attempt)
                verbose_proxy_logger.warning(
                    "mavvrik: %s attempt %d/%d failed, retrying in %.1fs: %s",
                    tag,
                    attempt + 1,
                    _MAX_RETRIES,
                    wait,
                    last_exc,
                )
                await asyncio.sleep(wait)

    raise RuntimeError(
        f"mavvrik: {tag} failed after {_MAX_RETRIES} attempts: {last_exc}"
    )

from __future__ import annotations

from typing import Final


def uses_aiohttp_transport() -> bool:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    return AsyncHTTPHandler.should_use_aiohttp_transport()


def _non_positive_timeout(timeout: object) -> bool:
    import httpx

    seconds: Final = (
        (timeout.connect, timeout.read, timeout.write, timeout.pool)
        if isinstance(timeout, httpx.Timeout)
        else (timeout,)
    )
    return any(isinstance(value, int | float) and value <= 0 for value in seconds)


def sends_through_python(client: object, timeout: object = None) -> bool:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, rsa_key_exchange_and_chacha20_excluded
    from litellm.rust_bridge import detectors

    return (
        isinstance(client, AsyncHTTPHandler)
        or detectors.unsupported_tls()
        or rsa_key_exchange_and_chacha20_excluded()
        or _non_positive_timeout(timeout)
    )

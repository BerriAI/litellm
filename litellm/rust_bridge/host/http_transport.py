from __future__ import annotations


def uses_aiohttp_transport() -> bool:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    return AsyncHTTPHandler.should_use_aiohttp_transport()


def sends_through_python(client: object) -> bool:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
    from litellm.rust_bridge import detectors

    return isinstance(client, AsyncHTTPHandler) or detectors.unsupported_tls()

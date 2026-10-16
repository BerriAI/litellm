from __future__ import annotations


def uses_aiohttp_transport() -> bool:
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    return AsyncHTTPHandler.should_use_aiohttp_transport()

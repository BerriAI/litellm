import os
import ssl
from typing import Final

import httpx


def _get_http_proxy(proxy_url: str) -> httpx.Proxy:
    from litellm.secret_managers.main import str_to_bool

    if str_to_bool(os.getenv("DISABLE_OUTBOUND_PROXY_TLS_VERIFICATION", "False")):
        # For the case when proxy uses self-signed certificate.
        proxy_ssl_context: Final = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        proxy_ssl_context.check_hostname = False
        proxy_ssl_context.verify_mode = ssl.CERT_NONE
        return httpx.Proxy(url=proxy_url, ssl_context=proxy_ssl_context)

    return httpx.Proxy(url=proxy_url, ssl_context=None)


def _rewrite_request(request: httpx.Request) -> httpx.Request:
    headers: Final = request.headers.copy()
    headers["X-Forwarded-Proto"] = request.url.scheme
    url: Final = request.url.copy_with(scheme="http")

    return httpx.Request(
        method=request.method,
        url=url,
        headers=headers,
        stream=request.stream,
        extensions=request.extensions,
    )


class AsyncProxyTransport(httpx.AsyncHTTPTransport):
    def __init__(self, proxy_url: str) -> None:
        super().__init__(proxy=_get_http_proxy(proxy_url))

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await super().handle_async_request(_rewrite_request(request))


class ProxyTransport(httpx.HTTPTransport):
    def __init__(self, proxy_url: str) -> None:
        super().__init__(proxy=_get_http_proxy(proxy_url))

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return super().handle_request(_rewrite_request(request))

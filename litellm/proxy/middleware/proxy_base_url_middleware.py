import os
import typing
from urllib.parse import urlparse


class ProxyBaseUrlMiddleware:
    """
    Middleware to correctly set the request host and scheme based on PROXY_BASE_URL
    or X-Forwarded-* headers without modifying `client` IP (which breaks IP security).
    """

    def __init__(self, app: typing.Callable) -> None:
        self.app = app

    async def __call__(  # noqa: PLR0915
        self, scope: dict, receive: typing.Callable, send: typing.Callable
    ) -> None:
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)

        # Fix UI Trailing Slash 307 Redirects
        # If requesting an extensionless path under /ui, rewrite internal path to /index.html
        path = scope.get("path", "")
        if path.startswith("/ui/") and "." not in path.split("/")[-1]:
            if not path.endswith("/"):
                # Rewrite internal path for StaticFiles to serve the index.html directly
                # without issuing a 307 Temporary Redirect to `/ui/route/`
                scope["path"] = path + "/index.html"

        headers = dict(scope.get("headers", []))

        # 1. Check PROXY_BASE_URL
        proxy_base_url = os.getenv("PROXY_BASE_URL")
        if proxy_base_url:
            parsed = urlparse(proxy_base_url)
            if parsed.scheme:
                scope["scheme"] = parsed.scheme

            if parsed.netloc:
                new_host_value = parsed.netloc.encode("latin-1")
                new_headers = []
                host_replaced = False
                for k, v in scope.get("headers", []):
                    if k == b"host":
                        new_headers.append((b"host", new_host_value))
                        host_replaced = True
                    else:
                        new_headers.append((k, v))

                if not host_replaced:
                    new_headers.append((b"host", new_host_value))

                scope["headers"] = new_headers

                port = parsed.port
                if not port:
                    port = 443 if parsed.scheme == "https" else 80

                hostname = parsed.hostname or ""
                scope["server"] = (hostname, port)

            return await self.app(scope, receive, send)

        # 2. Otherwise parse X-Forwarded-* headers
        x_forwarded_proto = (
            headers.get(b"x-forwarded-proto", b"")
            .decode("latin-1")
            .split(",")[0]
            .strip()
        )
        x_forwarded_host = (
            headers.get(b"x-forwarded-host", b"")
            .decode("latin-1")
            .split(",")[0]
            .strip()
        )
        x_forwarded_port = (
            headers.get(b"x-forwarded-port", b"")
            .decode("latin-1")
            .split(",")[0]
            .strip()
        )

        if x_forwarded_proto:
            scope["scheme"] = x_forwarded_proto

        if x_forwarded_host:
            host_header = x_forwarded_host
            if x_forwarded_port and ":" not in host_header:
                host_header = f"{host_header}:{x_forwarded_port}"

            new_headers = []
            host_replaced = False
            for k, v in scope.get("headers", []):
                if k == b"host":
                    new_headers.append((b"host", host_header.encode("latin-1")))
                    host_replaced = True
                else:
                    new_headers.append((k, v))

            if not host_replaced:
                new_headers.append((b"host", host_header.encode("latin-1")))
            scope["headers"] = new_headers

            port = 0
            host = x_forwarded_host
            if ":" in x_forwarded_host:
                host, port_str = x_forwarded_host.split(":", 1)
                try:
                    port = int(port_str)
                except ValueError:
                    pass
            elif x_forwarded_port:
                try:
                    port = int(x_forwarded_port)
                except ValueError:
                    pass
            else:
                port = 443 if scope.get("scheme", "") == "https" else 80

            scope["server"] = (host, port)

        return await self.app(scope, receive, send)

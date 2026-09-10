"""Integration and unit tests for ext_proc_proxy."""

import asyncio
import os
import ssl
import unittest
from typing import Optional

import aiohttp
from aiohttp import web

from envoy_ext_proc_proxy.cert_utils import (
    create_server_ssl_context,
    generate_self_signed_cert,
)
from envoy_ext_proc_proxy.config import ProxyConfig
from envoy_ext_proc_proxy.proxy import create_proxy_app


class TestProxyIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration test suite for the proxy."""

    @classmethod
    def setUpClass(cls):
        # Generate self-signed certificate for the proxy server
        cls.proxy_cert, cls.proxy_key = generate_self_signed_cert(hostname="localhost")

        # Generate separate certificate for the mock HTTPS upstream server
        cls.upstream_cert, cls.upstream_key = generate_self_signed_cert(hostname="localhost")

        # SSL context for the proxy server
        cls.proxy_server_ssl = create_server_ssl_context(cert_file=cls.proxy_cert, key_file=cls.proxy_key)

        # SSL context for the mock upstream server
        cls.upstream_server_ssl = create_server_ssl_context(cert_file=cls.upstream_cert, key_file=cls.upstream_key)

        # Client SSL context that trusts the proxy certificate (for testing proxy HTTPS connection)
        cls.client_ssl_to_proxy = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cls.proxy_cert)

        # Upstream client SSL context for proxy to verify upstream cert
        cls.proxy_to_upstream_ssl = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=cls.upstream_cert)

    @classmethod
    def tearDownClass(cls):
        for f in (cls.proxy_cert, cls.proxy_key, cls.upstream_cert, cls.upstream_key):
            if os.path.exists(f):
                os.remove(f)

    async def asyncSetUp(self):
        self.runners = []

    async def asyncTearDown(self):
        for runner in reversed(self.runners):
            await runner.cleanup()

    async def _start_mock_upstream(
        self,
        handler,
        use_ssl: bool = False,
    ) -> int:
        """Start a mock upstream server and return its port."""
        app = web.Application()
        app.router.add_route("*", "/{path_info:.*}", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        self.runners.append(runner)

        ssl_ctx = self.upstream_server_ssl if use_ssl else None
        site = web.TCPSite(runner, host="127.0.0.1", port=0, ssl_context=ssl_ctx)
        await site.start()
        port = runner.addresses[0][1]
        return port

    async def _start_proxy(
        self,
        upstream_ssl_context: Optional[ssl.SSLContext] = None,
        upstream_timeout: float = 2.0,
    ) -> int:
        """Start the proxy server and return its port."""
        config = ProxyConfig(
            host="127.0.0.1",
            port=0,
            cert=self.proxy_cert,
            key=self.proxy_key,
            upstream_timeout=upstream_timeout,
        )
        app = create_proxy_app(config=config, upstream_ssl_context=upstream_ssl_context)
        runner = web.AppRunner(app, keepalive_timeout=75.0)
        await runner.setup()
        self.runners.append(runner)

        site = web.TCPSite(
            runner,
            host="127.0.0.1",
            port=0,
            ssl_context=self.proxy_server_ssl,
        )
        await site.start()
        proxy_port = runner.addresses[0][1]
        return proxy_port

    async def test_x_forwarded_proto_http(self):
        """Test proxying to HTTP upstream when X-Forwarded-Proto is http."""

        async def mock_handler(request: web.Request):
            body = await request.text()
            return web.json_response(
                {
                    "scheme": request.scheme,
                    "method": request.method,
                    "path": request.path,
                    "query": dict(request.query),
                    "received_body": body,
                    "custom_header": request.headers.get("X-Test-Header"),
                    "x_forwarded_proto": request.headers.get("X-Forwarded-Proto"),
                }
            )

        http_port = await self._start_mock_upstream(mock_handler, use_ssl=False)
        proxy_port = await self._start_proxy(upstream_ssl_context=self.proxy_to_upstream_ssl)

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/api/test?foo=bar"
            headers = {
                "Host": f"127.0.0.1:{http_port}",
                "X-Forwarded-Proto": "http",
                "X-Test-Header": "HelloProxy",
            }
            async with session.post(
                url,
                headers=headers,
                data="sample payload",
                ssl=self.client_ssl_to_proxy,
            ) as resp:
                self.assertEqual(resp.status, 200)
                data = await resp.json()
                self.assertEqual(data["scheme"], "http")
                self.assertEqual(data["method"], "POST")
                self.assertEqual(data["path"], "/api/test")
                self.assertEqual(data["query"], {"foo": "bar"})
                self.assertEqual(data["received_body"], "sample payload")
                self.assertEqual(data["custom_header"], "HelloProxy")
                self.assertIsNone(data["x_forwarded_proto"])

    async def test_x_forwarded_proto_https_with_tls_verify(self):
        """Test proxying to HTTPS upstream with TLS verification when X-Forwarded-Proto is https."""

        async def mock_handler(request: web.Request):
            return web.Response(text="secure response", status=200)

        https_port = await self._start_mock_upstream(mock_handler, use_ssl=True)
        # Proxy configured with CA certificate to verify the upstream server
        proxy_port = await self._start_proxy(upstream_ssl_context=self.proxy_to_upstream_ssl)

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/secure-path"
            headers = {
                "Host": f"localhost:{https_port}",
                "X-Forwarded-Proto": "https",
            }
            async with session.get(url, headers=headers, ssl=self.client_ssl_to_proxy) as resp:
                self.assertEqual(resp.status, 200)
                text = await resp.text()
                self.assertEqual(text, "secure response")

    async def test_missing_x_forwarded_proto_defaults_to_https(self):
        """Test that missing X-Forwarded-Proto defaults to HTTPS."""

        async def mock_handler(request: web.Request):
            return web.Response(text="default https response", status=200)

        https_port = await self._start_mock_upstream(mock_handler, use_ssl=True)
        proxy_port = await self._start_proxy(upstream_ssl_context=self.proxy_to_upstream_ssl)

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/default-proto"
            headers = {
                "Host": f"localhost:{https_port}",
            }
            async with session.get(url, headers=headers, ssl=self.client_ssl_to_proxy) as resp:
                self.assertEqual(resp.status, 200)
                text = await resp.text()
                self.assertEqual(text, "default https response")

    async def test_invalid_x_forwarded_proto(self):
        """Test that invalid X-Forwarded-Proto returns 400 Bad Request."""
        proxy_port = await self._start_proxy()

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/test"
            headers = {
                "Host": "localhost:8080",
                "X-Forwarded-Proto": "ftp",
            }
            async with session.get(url, headers=headers, ssl=self.client_ssl_to_proxy) as resp:
                self.assertEqual(resp.status, 400)
                text = await resp.text()
                self.assertIn("Invalid X-Forwarded-Proto header value", text)

    async def test_header_filtering_and_exclusion(self):
        """Test hop-by-hop headers and X-Forwarded-For/Host exclusion."""
        received_upstream_headers = {}

        async def mock_handler(request: web.Request):
            nonlocal received_upstream_headers
            received_upstream_headers = dict(request.headers)
            response = web.Response(text="header test ok")
            response.headers["X-Upstream-Header"] = "UpstreamValue"
            response.headers["Connection"] = "close"
            return response

        http_port = await self._start_mock_upstream(mock_handler, use_ssl=False)
        proxy_port = await self._start_proxy(upstream_ssl_context=self.proxy_to_upstream_ssl)

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/headers"
            headers = {
                "Host": f"127.0.0.1:{http_port}",
                "X-Forwarded-Proto": "http",
                "X-Forwarded-For": "198.51.100.1",
                "X-Forwarded-Host": "malicious.example.com",
                "X-Valid-Header": "KeepThis",
                "Connection": "X-Custom-Hop",
                "X-Custom-Hop": "HopVal",
            }
            async with session.get(url, headers=headers, ssl=self.client_ssl_to_proxy) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("X-Upstream-Header"), "UpstreamValue")

                # Verify excluded headers were not sent upstream
                self.assertNotIn("X-Forwarded-For", received_upstream_headers)
                self.assertNotIn("X-Forwarded-Host", received_upstream_headers)
                self.assertNotIn("X-Forwarded-Proto", received_upstream_headers)
                self.assertNotIn("X-Custom-Hop", received_upstream_headers)
                self.assertEqual(received_upstream_headers.get("X-Valid-Header"), "KeepThis")

    async def test_large_payload_streaming(self):
        """Test streaming large request and response bodies."""

        async def echo_handler(request: web.Request):
            resp = web.StreamResponse(status=200)
            await resp.prepare(request)
            async for chunk in request.content.iter_any():
                await resp.write(chunk)
            await resp.write_eof()
            return resp

        http_port = await self._start_mock_upstream(echo_handler, use_ssl=False)
        proxy_port = await self._start_proxy(upstream_ssl_context=self.proxy_to_upstream_ssl)

        import io

        payload = b"A" * (2 * 1024 * 1024)  # 2MB payload

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/stream-echo"
            headers = {
                "Host": f"127.0.0.1:{http_port}",
                "X-Forwarded-Proto": "http",
            }
            async with session.post(
                url, headers=headers, data=io.BytesIO(payload), ssl=self.client_ssl_to_proxy
            ) as resp:
                self.assertEqual(resp.status, 200)
                body = await resp.read()
                self.assertEqual(len(body), len(payload))
                self.assertEqual(body, payload)

    async def test_client_keepalive(self):
        """Test persistent client keep-alive connection reuse."""
        request_count = 0

        async def count_handler(request: web.Request):
            nonlocal request_count
            request_count += 1
            return web.Response(text=f"count={request_count}")

        http_port = await self._start_mock_upstream(count_handler, use_ssl=False)
        proxy_port = await self._start_proxy(upstream_ssl_context=self.proxy_to_upstream_ssl)

        # Single persistent ClientSession connecting to proxy over HTTPS
        connector = aiohttp.TCPConnector(ssl=self.client_ssl_to_proxy)
        async with aiohttp.ClientSession(connector=connector) as session:
            url = f"https://127.0.0.1:{proxy_port}/keepalive"
            headers = {
                "Host": f"127.0.0.1:{http_port}",
                "X-Forwarded-Proto": "http",
            }
            for i in range(1, 6):
                async with session.get(url, headers=headers) as resp:
                    self.assertEqual(resp.status, 200)
                    text = await resp.text()
                    self.assertEqual(text, f"count={i}")

    async def test_upstream_tls_verification_failure(self):
        """Test that untrusted upstream TLS certificate triggers 502 Bad Gateway."""

        async def mock_handler(request: web.Request):
            return web.Response(text="secret", status=200)

        # Upstream uses self-signed cert
        https_port = await self._start_mock_upstream(mock_handler, use_ssl=True)
        # Proxy uses default SSL context (which does NOT trust upstream self-signed cert)
        proxy_port = await self._start_proxy(upstream_ssl_context=None)

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/untrusted"
            headers = {
                "Host": f"localhost:{https_port}",
                "X-Forwarded-Proto": "https",
            }
            async with session.get(url, headers=headers, ssl=self.client_ssl_to_proxy) as resp:
                self.assertEqual(resp.status, 502)
                text = await resp.text()
                self.assertIn("TLS certificate verification failed", text)

    async def test_upstream_connection_refused(self):
        """Test that connection failure to upstream returns 502 Bad Gateway."""
        proxy_port = await self._start_proxy(upstream_ssl_context=self.proxy_to_upstream_ssl)

        # Use an unused port
        unused_port = 59123

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/down"
            headers = {
                "Host": f"127.0.0.1:{unused_port}",
                "X-Forwarded-Proto": "http",
            }
            async with session.get(url, headers=headers, ssl=self.client_ssl_to_proxy) as resp:
                self.assertEqual(resp.status, 502)
                text = await resp.text()
                self.assertIn("Failed to connect to upstream", text)

    async def test_upstream_timeout(self):
        """Test that upstream timeout returns 504 Gateway Timeout."""

        async def slow_handler(request: web.Request):
            await asyncio.sleep(2.0)
            return web.Response(text="slow")

        http_port = await self._start_mock_upstream(slow_handler, use_ssl=False)
        # Proxy with a 0.5s upstream timeout
        proxy_port = await self._start_proxy(
            upstream_ssl_context=self.proxy_to_upstream_ssl,
            upstream_timeout=0.5,
        )

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/slow"
            headers = {
                "Host": f"127.0.0.1:{http_port}",
                "X-Forwarded-Proto": "http",
            }
            async with session.get(url, headers=headers, ssl=self.client_ssl_to_proxy) as resp:
                self.assertEqual(resp.status, 504)
                text = await resp.text()
                self.assertIn("504 Gateway Timeout", text)

    async def test_head_request(self):
        """Test HEAD request proxying."""

        async def mock_handler(request: web.Request):
            resp = web.Response(text="some body content")
            resp.headers["X-Custom-Head"] = "HeadValue"
            return resp

        http_port = await self._start_mock_upstream(mock_handler, use_ssl=False)
        proxy_port = await self._start_proxy(upstream_ssl_context=self.proxy_to_upstream_ssl)

        async with aiohttp.ClientSession() as session:
            url = f"https://127.0.0.1:{proxy_port}/head-test"
            headers = {
                "Host": f"127.0.0.1:{http_port}",
                "X-Forwarded-Proto": "http",
            }
            async with session.head(url, headers=headers, ssl=self.client_ssl_to_proxy) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get("X-Custom-Head"), "HeadValue")
                body = await resp.read()
                self.assertEqual(body, b"")


if __name__ == "__main__":
    unittest.main()

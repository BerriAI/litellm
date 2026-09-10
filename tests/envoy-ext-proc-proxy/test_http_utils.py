"""Unit tests for http_utils.py."""

import asyncio
import ssl
import unittest

import aiohttp
from aiohttp.client_reqrep import ConnectionKey
import multidict

from envoy_ext_proc_proxy.config import ProxyConfig
from envoy_ext_proc_proxy.http_utils import (
    EXCLUDED_REQUEST_HEADERS,
    HOP_BY_HOP_HEADERS,
    classify_upstream_error,
    create_client_session,
    filter_request_headers,
    filter_response_headers,
    is_bodyless_response,
)


class TestHttpUtils(unittest.IsolatedAsyncioTestCase):
    """Test suite for HTTP utility header filtering functions."""

    def test_filter_request_headers_dict(self):
        """Test filtering request headers from a standard dict."""
        headers = {
            ":method": "GET",
            ":path": "/index.html",
            ":authority": "example.com",
            "Host": "example.com",
            "User-Agent": "test-agent",
            "Connection": "keep-alive, X-Custom-Hop",
            "X-Custom-Hop": "StripThis",
            "Transfer-Encoding": "chunked",
            "X-Forwarded-For": "1.2.3.4",
            "X-Forwarded-Host": "attacker.com",
            "X-Forwarded-Proto": "http",
            "X-Ai-Proxy-Request-Id": "req-12345",
            "X-Valid-Header": "KeepThis",
        }

        filtered = filter_request_headers(headers)
        self.assertIsInstance(filtered, multidict.CIMultiDict)
        self.assertEqual(filtered.get("Host"), "example.com")
        self.assertEqual(filtered.get("User-Agent"), "test-agent")
        self.assertEqual(filtered.get("X-Valid-Header"), "KeepThis")

        # Excluded and stripped headers
        self.assertNotIn(":method", filtered)
        self.assertNotIn(":path", filtered)
        self.assertNotIn("Connection", filtered)
        self.assertNotIn("X-Custom-Hop", filtered)
        self.assertNotIn("Transfer-Encoding", filtered)
        self.assertNotIn("X-Forwarded-For", filtered)
        self.assertNotIn("X-Forwarded-Host", filtered)
        self.assertNotIn("X-Forwarded-Proto", filtered)
        self.assertNotIn("X-Ai-Proxy-Request-Id", filtered)

    def test_filter_request_headers_cimultidict(self):
        """Test filtering request headers from a CIMultiDictProxy."""
        md = multidict.CIMultiDict(
            [
                ("Host", "example.com"),
                ("Accept", "text/html"),
                ("Connection", "upgrade, X-Hop"),
                ("Upgrade", "websocket"),
                ("X-Hop", "hopval"),
                ("X-Forwarded-For", "127.0.0.1"),
                ("X-Forwarded-Host", "example.org"),
                ("X-Forwarded-Proto", "https"),
                ("X-Ai-Proxy-Request-Id", "uuid-789"),
            ]
        )
        proxy = multidict.CIMultiDictProxy(md)

        filtered = filter_request_headers(proxy)
        self.assertEqual(filtered.get("Host"), "example.com")
        self.assertEqual(filtered.get("Accept"), "text/html")
        self.assertNotIn("Connection", filtered)
        self.assertNotIn("Upgrade", filtered)
        self.assertNotIn("X-Hop", filtered)
        self.assertNotIn("X-Forwarded-For", filtered)
        self.assertNotIn("X-Forwarded-Host", filtered)
        self.assertNotIn("X-Forwarded-Proto", filtered)
        self.assertNotIn("X-Ai-Proxy-Request-Id", filtered)

    def test_filter_response_headers_dict(self):
        """Test filtering response headers from a dict."""
        headers = {
            ":status": "200",
            "Content-Type": "application/json",
            "Content-Length": "42",
            "Connection": "close, X-Dynamic-Hop",
            "X-Dynamic-Hop": "DynamicValue",
            "Keep-Alive": "timeout=5",
            "Transfer-Encoding": "chunked",
            "X-Custom-Response": "PreserveThis",
        }

        filtered = filter_response_headers(headers)
        self.assertEqual(filtered.get("Content-Type"), "application/json")
        self.assertEqual(filtered.get("Content-Length"), "42")
        self.assertEqual(filtered.get("X-Custom-Response"), "PreserveThis")

        self.assertNotIn(":status", filtered)
        self.assertNotIn("Connection", filtered)
        self.assertNotIn("X-Dynamic-Hop", filtered)
        self.assertNotIn("Keep-Alive", filtered)
        self.assertNotIn("Transfer-Encoding", filtered)

    def test_filter_response_headers_cimultidict(self):
        """Test filtering response headers from a CIMultiDict."""
        md = multidict.CIMultiDict(
            [
                ("Server", "aiohttp"),
                ("Set-Cookie", "c1=v1"),
                ("Set-Cookie", "c2=v2"),
                ("Connection", "X-Hop1, X-Hop2"),
                ("X-Hop1", "val1"),
                ("X-Hop2", "val2"),
                ("Proxy-Authenticate", "Basic"),
            ]
        )
        proxy = multidict.CIMultiDictProxy(md)

        filtered = filter_response_headers(proxy)
        self.assertEqual(filtered.get("Server"), "aiohttp")
        self.assertEqual(filtered.getall("Set-Cookie"), ["c1=v1", "c2=v2"])

        self.assertNotIn("Connection", filtered)
        self.assertNotIn("X-Hop1", filtered)
        self.assertNotIn("X-Hop2", filtered)
        self.assertNotIn("Proxy-Authenticate", filtered)

    def test_filter_headers_multiple_connection_headers(self):
        """Test that tokens from multiple Connection headers are all stripped."""
        md = multidict.CIMultiDict(
            [
                ("Connection", "X-Hop1"),
                ("Connection", "X-Hop2, close"),
                ("X-Hop1", "strip-1"),
                ("X-Hop2", "strip-2"),
                ("X-Kept", "keep-this"),
            ]
        )
        filtered = filter_response_headers(md)
        self.assertEqual(filtered.get("X-Kept"), "keep-this")
        self.assertNotIn("Connection", filtered)
        self.assertNotIn("X-Hop1", filtered)
        self.assertNotIn("X-Hop2", filtered)

    def test_is_bodyless_response(self):
        """Test is_bodyless_response predicate for HEAD, 204, 304."""
        self.assertTrue(is_bodyless_response("HEAD", 200))
        self.assertTrue(is_bodyless_response("head", 404))
        self.assertTrue(is_bodyless_response("GET", 204))
        self.assertTrue(is_bodyless_response("GET", 304))
        self.assertTrue(is_bodyless_response("POST", 204))
        self.assertFalse(is_bodyless_response("GET", 200))
        self.assertFalse(is_bodyless_response("POST", 200))
        self.assertFalse(is_bodyless_response("GET", 404))

    def test_classify_upstream_error(self):
        """Test error classification for various exception types."""
        # TLS error
        tls_err = ssl.SSLCertVerificationError("cert failed")
        status, msg = classify_upstream_error(tls_err)
        self.assertEqual(status, 502)
        self.assertIn("TLS certificate verification failed", msg)

        # Connection error with valid ConnectionKey
        conn_key = ConnectionKey(
            host="localhost",
            port=8080,
            is_ssl=False,
            ssl=None,
            proxy=None,
            proxy_auth=None,
            proxy_headers_hash=None,
        )
        conn_err = aiohttp.ClientConnectorError(conn_key, OSError("cannot connect"))
        status, msg = classify_upstream_error(conn_err)
        self.assertEqual(status, 502)
        self.assertIn("Failed to connect to upstream", msg)

        # Timeout error
        timeout_err = asyncio.TimeoutError()
        status, msg = classify_upstream_error(timeout_err)
        self.assertEqual(status, 504)
        self.assertIn("Gateway Timeout", msg)

        # Generic ClientError
        client_err = aiohttp.ClientError("some client error")
        status, msg = classify_upstream_error(client_err)
        self.assertEqual(status, 502)
        self.assertIn("Upstream client error", msg)

        # Generic Exception
        gen_err = RuntimeError("something broke")
        status, msg = classify_upstream_error(gen_err)
        self.assertEqual(status, 502)
        self.assertIn("Upstream error", msg)

    async def test_create_client_session_auto_decompress_false(self):
        """Test create_client_session creates a session with auto_decompress=False."""
        config = ProxyConfig(upstream_timeout=45.0)
        session = create_client_session(config)
        self.assertFalse(session.auto_decompress)
        self.assertEqual(session.timeout.total, 45.0)
        await session.close()


if __name__ == "__main__":
    unittest.main()

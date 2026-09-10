"""Unit tests for __main__.py."""

import asyncio
import os
import unittest
from unittest.mock import patch

import grpc

from envoy_ext_proc_proxy import main, run_servers
from envoy_ext_proc_proxy.cert_utils import generate_self_signed_cert
from envoy_ext_proc_proxy.config import ProxyConfig


class TestMain(unittest.IsolatedAsyncioTestCase):
    """Test suite for __main__ entry point."""

    @classmethod
    def setUpClass(cls):
        cls.cert_file, cls.key_file = generate_self_signed_cert(hostname="localhost")

    @classmethod
    def tearDownClass(cls):
        for f in (cls.cert_file, cls.key_file):
            if os.path.exists(f):
                os.remove(f)

    async def test_run_servers(self):
        """Test that run_servers loads certs and calls run_proxy and run_grpc_server."""
        config = ProxyConfig(
            cert=self.cert_file,
            key=self.key_file,
            port=0,
            ext_proc_port=0,
        )

        proxy_called = False
        grpc_called = False

        async def mock_run_proxy(config, ssl_context, **kwargs):
            nonlocal proxy_called
            proxy_called = True
            self.assertIsNotNone(ssl_context)
            self.assertIn("session_registry", kwargs)

        async def mock_run_grpc_server(config, server_credentials, **kwargs):
            nonlocal grpc_called
            grpc_called = True
            self.assertIsInstance(server_credentials, grpc.ServerCredentials)
            self.assertIn("session_registry", kwargs)

        with (
            patch("envoy_ext_proc_proxy.run_proxy", side_effect=mock_run_proxy),
            patch("envoy_ext_proc_proxy.run_grpc_server", side_effect=mock_run_grpc_server),
        ):
            await run_servers(config)

        self.assertTrue(proxy_called)
        self.assertTrue(grpc_called)

    def test_main_cli_invocation(self):
        """Test main() parses args and invokes run_servers."""

        def fake_run(coro):
            coro.close()

        with (
            patch("sys.argv", ["ext_proc_proxy", "--self-signed"]),
            patch("envoy_ext_proc_proxy.asyncio.run", side_effect=fake_run) as mock_run,
        ):
            main()
            self.assertTrue(mock_run.called)


if __name__ == "__main__":
    unittest.main()

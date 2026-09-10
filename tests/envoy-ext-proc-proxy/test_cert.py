"""Tests for certificate utilities."""

import os
import ssl
import tempfile
import unittest

import grpc

from envoy_ext_proc_proxy.cert_utils import (
    create_grpc_server_credentials,
    create_server_ssl_context,
    generate_self_signed_cert,
    get_or_create_server_cert_and_key,
)


class TestCertUtils(unittest.TestCase):
    """Test certificate generation and SSL context creation."""

    def test_generate_self_signed_cert_temp(self):
        """Test generating self-signed certificate in temporary directory."""
        cert_path, key_path = generate_self_signed_cert()
        try:
            self.assertTrue(os.path.exists(cert_path))
            self.assertTrue(os.path.exists(key_path))
            self.assertGreater(os.path.getsize(cert_path), 0)
            self.assertGreater(os.path.getsize(key_path), 0)

            # Check that SSLContext can load the generated cert and key
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
        finally:
            if os.path.exists(cert_path):
                os.remove(cert_path)
            if os.path.exists(key_path):
                os.remove(key_path)

    def test_generate_self_signed_cert_custom_path(self):
        """Test generating self-signed certificate at specified paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cert_path = os.path.join(temp_dir, "custom_cert.pem")
            key_path = os.path.join(temp_dir, "custom_key.pem")

            res_cert, res_key = generate_self_signed_cert(
                cert_path=cert_path,
                key_path=key_path,
                hostname="test.example.com",
            )
            self.assertEqual(res_cert, cert_path)
            self.assertEqual(res_key, key_path)
            self.assertTrue(os.path.exists(cert_path))
            self.assertTrue(os.path.exists(key_path))

            ctx = create_server_ssl_context(cert_file=cert_path, key_file=key_path)
            self.assertIsInstance(ctx, ssl.SSLContext)

    def test_get_or_create_server_cert_and_key_self_signed(self):
        """Test get_or_create_server_cert_and_key with generate_self_signed=True."""
        cert_path, key_path = get_or_create_server_cert_and_key(generate_self_signed=True)
        try:
            self.assertTrue(os.path.exists(cert_path))
            self.assertTrue(os.path.exists(key_path))
            ctx = create_server_ssl_context(cert_file=cert_path, key_file=key_path)
            self.assertIsInstance(ctx, ssl.SSLContext)
            self.assertEqual(ctx.minimum_version, ssl.TLSVersion.TLSv1_2)
        finally:
            if os.path.exists(cert_path):
                os.remove(cert_path)
            if os.path.exists(key_path):
                os.remove(key_path)

    def test_get_or_create_server_cert_and_key_missing_args(self):
        """Test error handling when cert or key files are missing in get_or_create."""
        with self.assertRaises(ValueError):
            get_or_create_server_cert_and_key(cert_file=None, key_file=None)

        with self.assertRaises(FileNotFoundError):
            get_or_create_server_cert_and_key(cert_file="/nonexistent/cert.pem", key_file="/nonexistent/key.pem")

    def test_create_server_ssl_context_missing_files(self):
        """Test error handling in create_server_ssl_context when files do not exist."""
        with self.assertRaises(FileNotFoundError):
            create_server_ssl_context(cert_file="/nonexistent/cert.pem", key_file="/nonexistent/key.pem")

    def test_create_grpc_server_credentials(self):
        """Test creating gRPC server credentials from cert and key files."""
        cert_path, key_path = generate_self_signed_cert()
        try:
            creds = create_grpc_server_credentials(cert_file=cert_path, key_file=key_path)
            self.assertIsInstance(creds, grpc.ServerCredentials)
        finally:
            if os.path.exists(cert_path):
                os.remove(cert_path)
            if os.path.exists(key_path):
                os.remove(key_path)

    def test_create_grpc_server_credentials_missing_files(self):
        """Test error handling in create_grpc_server_credentials when files do not exist."""
        with self.assertRaises(FileNotFoundError):
            create_grpc_server_credentials(cert_file="/nonexistent/cert.pem", key_file="/nonexistent/key.pem")

    def test_generate_self_signed_cert_openssl_not_found(self):
        """Test error message when openssl binary is not found."""
        from unittest.mock import patch

        with patch("subprocess.run", side_effect=FileNotFoundError):
            with self.assertRaises(RuntimeError) as ctx:
                generate_self_signed_cert()
            self.assertIn("not found in PATH", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

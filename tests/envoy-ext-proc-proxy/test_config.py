"""Tests for configuration parsing."""

import unittest

from envoy_ext_proc_proxy.config import ProxyConfig, parse_args


class TestConfig(unittest.TestCase):
    """Test CLI argument parsing and config defaults."""

    def test_default_config(self):
        """Test default config with self_signed flag."""
        config = parse_args(["--self-signed"])
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 8443)
        self.assertTrue(config.self_signed)
        self.assertEqual(config.keepalive_timeout, 75.0)
        self.assertEqual(config.upstream_timeout, 60.0)
        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.ext_proc_host, "0.0.0.0")
        self.assertEqual(config.ext_proc_port, 50051)
        self.assertEqual(config.ext_proc_target, "http://127.0.0.1:8080")

    def test_custom_config(self):
        """Test custom CLI arguments."""
        config = parse_args(
            [
                "--host",
                "127.0.0.1",
                "-p",
                "9443",
                "--cert",
                "/path/to/cert.pem",
                "--key",
                "/path/to/key.pem",
                "--keepalive-timeout",
                "30.0",
                "--upstream-timeout",
                "15.0",
                "--log-level",
                "DEBUG",
                "--ext-proc-host",
                "127.0.0.1",
                "--ext-proc-port",
                "50052",
                "--ext-proc-target",
                "https://backend.internal:8443",
            ]
        )
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 9443)
        self.assertEqual(config.cert, "/path/to/cert.pem")
        self.assertEqual(config.key, "/path/to/key.pem")
        self.assertFalse(config.self_signed)
        self.assertEqual(config.keepalive_timeout, 30.0)
        self.assertEqual(config.upstream_timeout, 15.0)
        self.assertEqual(config.log_level, "DEBUG")
        self.assertEqual(config.ext_proc_host, "127.0.0.1")
        self.assertEqual(config.ext_proc_port, 50052)
        self.assertEqual(config.ext_proc_target, "https://backend.internal:8443")

    def test_missing_cert_and_self_signed(self):
        """Test error when neither cert/key nor --self-signed is given."""
        with self.assertRaises(SystemExit):
            parse_args([])

    def test_invalid_ext_proc_target(self):
        """Test error when --ext-proc-target is invalid."""
        with self.assertRaises(SystemExit):
            parse_args(["--self-signed", "--ext-proc-target", "ftp://invalid-target"])

        with self.assertRaises(SystemExit):
            parse_args(["--self-signed", "--ext-proc-target", "no-scheme-target"])


if __name__ == "__main__":
    unittest.main()

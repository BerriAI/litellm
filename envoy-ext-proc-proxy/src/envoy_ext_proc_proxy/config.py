"""Configuration and argument parsing for ext_proc_proxy."""

import argparse
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class ProxyConfig:
    """Proxy and ext_proc server configuration."""

    # HTTPS Proxy settings
    host: str = "0.0.0.0"
    port: int = 8443
    cert: str | None = None
    key: str | None = None
    self_signed: bool = False
    keepalive_timeout: float = 75.0
    upstream_timeout: float = 60.0
    log_level: str = "INFO"

    # Envoy ext_proc server settings
    ext_proc_host: str = "0.0.0.0"
    ext_proc_port: int = 50051
    ext_proc_target: str = "http://127.0.0.1:8080"


def parse_args(args: list[str] | None = None) -> ProxyConfig:
    """Parse command line arguments into ProxyConfig."""
    parser = argparse.ArgumentParser(
        description="HTTPS-terminating HTTP/HTTPS proxy and Envoy ext_proc server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # HTTPS Proxy options
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host/IP address to bind the HTTPS proxy server to",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8443,
        help="Port to listen for incoming HTTPS proxy connections",
    )
    parser.add_argument(
        "--cert",
        help="Path to server certificate PEM file",
    )
    parser.add_argument(
        "--key",
        help="Path to server private key PEM file",
    )
    parser.add_argument(
        "--self-signed",
        action="store_true",
        help="Generate a self-signed certificate for testing",
    )
    parser.add_argument(
        "--keepalive-timeout",
        type=float,
        default=75.0,
        help="Keep-alive timeout for client connections in seconds",
    )
    parser.add_argument(
        "--upstream-timeout",
        type=float,
        default=60.0,
        help="Timeout for upstream requests in seconds",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    # Envoy ext_proc options
    parser.add_argument(
        "--ext-proc-host",
        default="0.0.0.0",
        help="Host/IP address to bind the Envoy ext_proc gRPC server to",
    )
    parser.add_argument(
        "--ext-proc-port",
        type=int,
        default=50051,
        help="Port to listen for Envoy ext_proc gRPC connections",
    )
    parser.add_argument(
        "--ext-proc-target",
        default="http://127.0.0.1:8080",
        help="Target URL for the ext_proc HTTP server (e.g. http://127.0.0.1:8080)",
    )

    parsed = parser.parse_args(args)

    if not parsed.self_signed and (not parsed.cert or not parsed.key):
        parser.error("Either specify --self-signed or provide both --cert and --key.")

    # Validate ext_proc_target URL
    parsed_url = urlparse(parsed.ext_proc_target)
    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        parser.error(
            f"Invalid --ext-proc-target '{parsed.ext_proc_target}'. "
            "Must be a valid HTTP or HTTPS URL, e.g. http://127.0.0.1:8080"
        )

    return ProxyConfig(
        host=parsed.host,
        port=parsed.port,
        cert=parsed.cert,
        key=parsed.key,
        self_signed=parsed.self_signed,
        keepalive_timeout=parsed.keepalive_timeout,
        upstream_timeout=parsed.upstream_timeout,
        log_level=parsed.log_level,
        ext_proc_host=parsed.ext_proc_host,
        ext_proc_port=parsed.ext_proc_port,
        ext_proc_target=parsed.ext_proc_target,
    )

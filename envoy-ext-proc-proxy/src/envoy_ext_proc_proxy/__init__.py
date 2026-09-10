"""ext_proc_proxy - HTTPS/HTTP reverse and forward proxy with Envoy ext_proc support."""

import os
import sys

# Ensure generated protobuf stubs are importable
GEN_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "protogen"))
if os.path.exists(GEN_DIR) and GEN_DIR not in sys.path:
    sys.path.insert(0, GEN_DIR)

import asyncio
import logging

from envoy_ext_proc_proxy.cert_utils import (
    create_grpc_server_credentials,
    create_server_ssl_context,
    get_or_create_server_cert_and_key,
)
from envoy_ext_proc_proxy.config import ProxyConfig, parse_args
from envoy_ext_proc_proxy.ext_proc_server import run_grpc_server
from envoy_ext_proc_proxy.proxy import run_proxy
from envoy_ext_proc_proxy.session_registry import SessionRegistry

logger = logging.getLogger("ext_proc_proxy.main")


async def run_servers(config: ProxyConfig) -> None:
    """Generate or load TLS certificates and concurrently run both servers."""
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    logger.info("Initializing server TLS certificates...")
    cert_file, key_file = get_or_create_server_cert_and_key(
        cert_file=config.cert,
        key_file=config.key,
        generate_self_signed=config.self_signed,
        hostname=config.host if config.host not in ("0.0.0.0", "::") else "localhost",
    )
    ssl_context = create_server_ssl_context(
        cert_file=cert_file,
        key_file=key_file,
    )
    server_credentials = create_grpc_server_credentials(
        cert_file=cert_file,
        key_file=key_file,
    )

    session_registry = SessionRegistry()

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(
                run_proxy(
                    config=config,
                    ssl_context=ssl_context,
                    session_registry=session_registry,
                )
            )
            tg.create_task(
                run_grpc_server(
                    config=config,
                    server_credentials=server_credentials,
                    session_registry=session_registry,
                )
            )
    except* (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Shutting down servers...")


def main() -> None:
    """Main entry point function."""
    config = parse_args()
    try:
        asyncio.run(run_servers(config))
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()

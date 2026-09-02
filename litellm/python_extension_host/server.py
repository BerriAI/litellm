# pyright: reportAny=false, reportMissingModuleSource=false, reportUnknownMemberType=false
from __future__ import annotations

import argparse
import asyncio
import os
from typing import Final

import grpc

from litellm.python_extension.generated.v1 import extension_host_pb2_grpc as pb_grpc

from .cache_client import create_gateway_channel, grpc_target
from .service import PythonExtensionHostService


async def serve(
    listen: str = "127.0.0.1:50051",
    token: str | None = None,
    gateway_endpoint: str | None = None,
) -> None:
    resolved_token: Final = token or os.environ.get("LITELLM_EXTENSION_HOST_TOKEN")
    if not resolved_token:
        raise ValueError("LITELLM_EXTENSION_HOST_TOKEN is required")
    gateway_channel: Final = create_gateway_channel(
        gateway_endpoint or os.environ.get("LITELLM_GATEWAY_SERVICES_ENDPOINT")
    )
    gateway_stub: Final = pb_grpc.GatewayServicesStub(gateway_channel) if gateway_channel is not None else None
    server: Final = grpc.aio.server()
    pb_grpc.add_PythonExtensionHostServicer_to_server(
        PythonExtensionHostService(resolved_token, gateway_stub=gateway_stub), server
    )
    if server.add_insecure_port(grpc_target(listen)) == 0:
        raise RuntimeError(f"failed to bind extension host to {listen}")
    await server.start()
    try:
        await server.wait_for_termination()
    finally:
        await server.stop(grace=5)
        if gateway_channel is not None:
            await gateway_channel.close()


def main() -> None:
    parser: Final = argparse.ArgumentParser(description="Run the LiteLLM Python extension host")
    parser.add_argument("--listen", default="127.0.0.1:50051")
    parser.add_argument("--gateway-endpoint")
    args: Final = parser.parse_args()
    asyncio.run(serve(listen=args.listen, gateway_endpoint=args.gateway_endpoint))


if __name__ == "__main__":
    main()

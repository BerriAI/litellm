import os

from uvicorn import run as run_uvicorn

from litellm.proxy.experimental_codex_gateway.app import create_gateway_app
from litellm.proxy.experimental_codex_gateway.settings import GatewaySettings


def main() -> None:
    settings = GatewaySettings.from_environment()
    host = os.environ.get("CODEX_GATEWAY_HOST", "127.0.0.1")
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("CODEX_GATEWAY_HOST must be a loopback address")
    port = int(os.environ.get("CODEX_GATEWAY_PORT", "4000"))
    run_uvicorn(create_gateway_app(settings=settings), host=host, port=port, access_log=False)

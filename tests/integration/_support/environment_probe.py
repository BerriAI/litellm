import os
from typing import Final

from fastapi.responses import JSONResponse

PROBE_ROUTE: Final = "/health/integration-environment"
PROBE_HOOK: Final = "integration._support.environment_probe:configure_environment_probe"


async def worker_environment(names: str) -> JSONResponse:
    return JSONResponse(
        content={"pid": os.getpid(), "environment": {name: os.environ.get(name) for name in names.split(",")}}
    )


def configure_environment_probe() -> None:
    from litellm.proxy.proxy_server import app

    app.add_api_route(PROBE_ROUTE, worker_environment, include_in_schema=False)

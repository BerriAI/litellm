import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Final

from fastapi import Response
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter

READINESS_BODY: Final = TypeAdapter(dict[str, object])


def configure_executor() -> None:
    asyncio.get_running_loop().set_default_executor(
        ThreadPoolExecutor(max_workers=32, thread_name_prefix="integration-provider")
    )


async def worker_readiness() -> JSONResponse:
    from litellm.proxy.health_endpoints._health_endpoints import health_readiness

    response: Final = Response()
    body: Final = READINESS_BODY.validate_python(await health_readiness(response))
    return JSONResponse(
        content=body, status_code=response.status_code, headers={"x-integration-worker-pid": str(os.getpid())}
    )


def configure_readiness() -> None:
    from litellm.proxy.proxy_server import app

    app.add_api_route("/health/integration-readiness", worker_readiness, include_in_schema=False)

from __future__ import annotations

import asyncio
import json
import os
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Final
from unittest.mock import patch

import httpx

from tests.sdk_function_trace.fixtures import OCR_MODEL
from tests.sdk_function_trace.profiler import FunctionTraceEvent, profile_python


def _proxy_app_without_license_check():
    from litellm.proxy.auth.litellm_license import LicenseCheck

    with patch.object(LicenseCheck, "is_premium", return_value=False):
        from litellm.proxy.proxy_server import app

    return app


async def _run_python_ocr_proxy_trace() -> tuple[FunctionTraceEvent, ...]:
    import litellm
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    app = _proxy_app_without_license_check()

    provider_requests = 0

    async def provider_response(request: httpx.Request) -> httpx.Response:
        nonlocal provider_requests
        provider_requests += 1
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(
                {
                    "pages": [{"index": 0, "markdown": "hello"}],
                    "model": OCR_MODEL,
                    "usage_info": {"pages_processed": 1},
                }
            ).encode(),
            request=request,
        )

    provider_client = AsyncHTTPHandler(timeout=5)
    original_provider_client: Final = provider_client.client
    provider_client.client = httpx.AsyncClient(
        transport=httpx.MockTransport(provider_response), timeout=5
    )
    async with AsyncExitStack() as stack:
        stack.push_async_callback(original_provider_client.aclose)
        stack.push_async_callback(provider_client.client.aclose)
        proxy_client: Final = await stack.enter_async_context(
            httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy.test"
            )
        )
        with (
            patch.dict(os.environ, {"MISTRAL_API_KEY": "test-key"}),
            patch("litellm.proxy.proxy_server.llm_router", None),
            patch("litellm.proxy.proxy_server.user_model", f"mistral/{OCR_MODEL}"),
            patch(
                "litellm.llms.custom_httpx.llm_http_handler.get_async_httpx_client",
                return_value=provider_client,
            ),
            profile_python(
                source_root=Path(litellm.__file__).parent, threads=True
            ) as profiler,
        ):
            response: Final = await proxy_client.post(
                "/ocr",
                json={
                    "model": f"mistral/{OCR_MODEL}",
                    "document": {
                        "type": "document_url",
                        "document_url": "https://example.com/document.pdf",
                    },
                    "pages": [0],
                },
            )

    if response.status_code != 200:
        raise RuntimeError(
            f"FastAPI /ocr returned {response.status_code}: {response.text}"
        )
    if provider_requests != 1:
        raise AssertionError(
            f"expected one provider request, received {provider_requests}"
        )
    return tuple(profiler.events)


def run_python_ocr_proxy_trace() -> tuple[FunctionTraceEvent, ...]:
    with patch.dict(
        os.environ,
        {"LITELLM_LOCAL_MODEL_COST_MAP": "True", "MISTRAL_API_KEY": "test-key"},
    ):
        os.environ.pop("LITELLM_LICENSE", None)
        return asyncio.run(_run_python_ocr_proxy_trace())

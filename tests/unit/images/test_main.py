from datetime import datetime
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx import llm_http_handler as llm_http_handler_module
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

PNG_BYTES: Final = b"\x89PNG\r\n\x1a\nfakepng"


def _edit_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"created": 1712697600, "data": [{"b64_json": "aW1n"}]})


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ("openai/dall-e-2", "openai/unpriced-image-model"))
async def test_router_image_edit_bills_the_deployment_price_with_a_logger_built_before_routing(
    monkeypatch: pytest.MonkeyPatch, model: str
):
    client: Final = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(_edit_ok))
    monkeypatch.setattr(llm_http_handler_module, "get_async_httpx_client", lambda **_kwargs: client)
    router: Final = litellm.Router(
        model_list=[
            {
                "model_name": "priced-edit-deployment",
                "litellm_params": {
                    "model": model,
                    "api_base": "https://edit.example/v1",
                    "api_key": "sk-test",
                    "output_cost_per_image": 0.5,
                },
            }
        ]
    )
    request: Final = {
        "model": "priced-edit-deployment",
        "prompt": "add a hat",
        "image": PNG_BYTES,
        "size": "1024x1024",
        "litellm_call_id": "proxy-call-id",
    }
    logging_obj, routed_request = litellm.utils.function_setup(
        original_function="aimage_edit",
        rules_obj=litellm.utils.Rules(),
        start_time=datetime.now(),
        **request,
    )

    response: Final = await router.aimage_edit(**routed_request, litellm_logging_obj=logging_obj)

    assert response._hidden_params["response_cost"] == pytest.approx(0.5)

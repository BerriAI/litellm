import json
from typing import Final

import httpx
import pytest

from litellm import aimage_generation
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.mark.parametrize(
    "model,expected_endpoint",
    [
        ("fal_ai/fal-ai/flux-pro/v1.1-ultra", "fal-ai/flux-pro/v1.1-ultra"),
        (
            "fal_ai/fal-ai/stable-diffusion-v35-medium",
            "fal-ai/stable-diffusion-v35-medium",
        ),
        ("fal_ai/fal-ai/nano-banana", "fal-ai/nano-banana"),
        (
            "fal_ai/fal-ai/gemini-25-flash-image",
            "fal-ai/gemini-25-flash-image",
        ),
    ],
)
@pytest.mark.asyncio
async def test_fal_ai_image_generation_basic(model: str, expected_endpoint: str) -> None:
    requests: Final[list[httpx.Request]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "images": [
                    {
                        "url": "https://example.com/generated-image.png",
                        "width": 1024,
                        "height": 768,
                        "content_type": "image/jpeg",
                    }
                ],
                "seed": 42,
            },
        )

    response: Final = await aimage_generation(
        model=model,
        prompt="A cute baby sea otter",
        api_key="test-fal-ai-key-12345",
        client=AsyncHTTPHandler(transport=httpx.MockTransport(handler)),
    )

    assert response.data
    assert response.data[0].url == "https://example.com/generated-image.png"
    assert len(requests) == 1
    assert requests[0].url.host == "fal.run"
    assert expected_endpoint in requests[0].url.path
    assert requests[0].headers["Authorization"] == "Key test-fal-ai-key-12345"
    assert json.loads(requests[0].content)["prompt"] == "A cute baby sea otter"


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)

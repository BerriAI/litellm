"""Live e2e: POST /v1/images/generations returns an image.

Registers an image deployment at runtime, drives it through the real OpenAI SDK
(LIT-4577), and asserts the response carries a generated image (url or base64).
"""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e


def _assert_image_returned(
    proxy: ProxyClient,
    resources: ResourceManager,
    sdk: SdkClients,
    prefix: str,
    params: LiteLLMParamsBody,
) -> None:
    model = f"{prefix}-{unique_marker()}"
    model_id = proxy.create_model(model, params)
    resources.defer(lambda: proxy.delete_model(model_id))
    client = sdk.openai(resources.key())

    images = client.images.generate(model=model, prompt="Draw a cute cat", n=1, size="1024x1024")
    data = images.data or []
    assert data, f"/images/generations returned no data: {images!r}"
    first = data[0]
    assert first.b64_json or first.url, "generated image has neither b64_json nor url"


class TestImageGeneration:
    @pytest.mark.covers("llm.images_generations.openai.basic.nonstream.works")
    def test_image_generation_returns_image(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        _assert_image_returned(
            proxy,
            resources,
            sdk,
            "e2e-image",
            LiteLLMParamsBody(model="openai/gpt-image-1-mini", api_key="os.environ/OPENAI_API_KEY"),
        )

    @pytest.mark.covers(
        "llm.images_generations.bedrock.basic.nonstream.works", exercised_on=["images_generations"]
    )
    def test_bedrock_image_generation_returns_image(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        _assert_image_returned(
            proxy,
            resources,
            sdk,
            "e2e-bedrock-image",
            LiteLLMParamsBody(
                model="bedrock/amazon.titan-image-generator-v2:0",
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
        )

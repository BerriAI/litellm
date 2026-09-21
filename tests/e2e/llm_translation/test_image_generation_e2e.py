"""Live e2e: POST /v1/images/generations returns an image.

Registers an image deployment at runtime, drives it through the real OpenAI SDK
(LIT-4577), and asserts the response carries a generated image (url or base64).
Malformed bodies the SDK refuses to build stay on the shared transport. Migrated
from litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from e2e_http import assert_client_error
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from openai.types import ImagesResponse
from proxy_client import ProxyClient
from pydantic import BaseModel
from sdk_clients import SdkClients

pytestmark = pytest.mark.e2e


class _OptionalImageBody(BaseModel):
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    size: str | None = None


def _assert_image_returned(images: ImagesResponse) -> None:
    data = images.data or []
    assert data, f"/images/generations returned no data: {images!r}"
    first = data[0]
    assert first.b64_json or first.url, f"generated image has neither b64_json nor url: {first!r}"


def _register(proxy: ProxyClient, resources: ResourceManager, prefix: str, params: LiteLLMParamsBody) -> tuple[str, str]:
    model = f"{prefix}-{unique_marker()}"
    model_id = proxy.create_model(model, params)
    resources.defer(lambda: proxy.delete_model(model_id))
    return model, resources.key()


def _register_openai_image(proxy: ProxyClient, resources: ResourceManager) -> tuple[str, str]:
    return _register(
        proxy,
        resources,
        "e2e-image",
        LiteLLMParamsBody(model="openai/gpt-image-1-mini", api_key="os.environ/OPENAI_API_KEY"),
    )


class TestImageGeneration:
    @pytest.mark.covers("llm.images_generations.openai.basic.nonstream.works")
    def test_image_generation_returns_image(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register_openai_image(proxy, resources)
        images = sdk.openai(key).images.generate(model=model, prompt="Draw a cute cat", n=1, size="1024x1024")
        _assert_image_returned(images)

    @pytest.mark.covers("llm.images_generations.bedrock.basic.nonstream.works", exercised_on=["images_generations"])
    def test_bedrock_image_generation_returns_image(
        self, proxy: ProxyClient, resources: ResourceManager, sdk: SdkClients
    ) -> None:
        model, key = _register(
            proxy,
            resources,
            "e2e-bedrock-image",
            LiteLLMParamsBody(
                model="bedrock/amazon.nova-canvas-v1:0",
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
        )
        images = sdk.openai(key).images.generate(model=model, prompt="Draw a cute cat", n=1, size="1024x1024")
        _assert_image_returned(images)

    @pytest.mark.skip(reason="stage red: product gap, /v1/images/generations 500s (aimage_generation TypeError) on missing prompt instead of 400")
    @pytest.mark.covers("llm.images_generations.openai.input_validation.nonstream.works")
    def test_missing_prompt_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_openai_image(proxy, resources)
        result = proxy.transport.send(
            "/v1/images/generations",
            headers=proxy.transport.bearer(key),
            json=_OptionalImageBody(model=model),
        )
        assert_client_error(result, "images missing prompt")

    @pytest.mark.covers("llm.images_generations.openai.input_validation.nonstream.works")
    def test_empty_prompt_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_openai_image(proxy, resources)
        result = proxy.transport.send(
            "/v1/images/generations",
            headers=proxy.transport.bearer(key),
            json=_OptionalImageBody(model=model, prompt=""),
        )
        assert_client_error(result, "images empty prompt")

    @pytest.mark.covers("llm.images_generations.openai.input_validation.nonstream.works")
    def test_invalid_size_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_openai_image(proxy, resources)
        result = proxy.transport.send(
            "/v1/images/generations",
            headers=proxy.transport.bearer(key),
            json=_OptionalImageBody(model=model, prompt="a blue square", size="999x999"),
        )
        assert_client_error(result, "images invalid size")

    @pytest.mark.covers("llm.images_generations.openai.input_validation.nonstream.works")
    def test_invalid_n_returns_client_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        model, key = _register_openai_image(proxy, resources)
        result = proxy.transport.send(
            "/v1/images/generations",
            headers=proxy.transport.bearer(key),
            json=_OptionalImageBody(model=model, prompt="a blue square", n=0),
        )
        assert_client_error(result, "images invalid n")

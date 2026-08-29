"""Live e2e: POST /v1/images/generations returns an image.

Registers an OpenAI image deployment at runtime and asserts the response carries a
generated image (url or base64). Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest
from e2e_config import unique_marker
from e2e_http import (
    assert_client_error,
    require_successful_call,
)
from endpoints_client import EndpointsClient, ImagesResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from pydantic import BaseModel

pytestmark = pytest.mark.e2e


class _OptionalImageBody(BaseModel):
    model: str | None = None
    prompt: str | None = None
    n: int | None = None
    size: str | None = None


def _assert_image_returned(body: str) -> None:
    parsed = ImagesResult.model_validate_json(body)
    assert parsed.data, f"/images/generations returned no data: {body[:300]}"
    first = parsed.data[0]
    assert first.b64_json or first.url, (
        f"generated image has neither b64_json nor url: {body[:300]}"
    )


def _register_openai_image(
    endpoints_client: EndpointsClient, resources: ResourceManager
) -> tuple[str, str]:
    model = f"e2e-image-{unique_marker()}"
    model_id = endpoints_client.create_model(
        model,
        LiteLLMParamsBody(model="openai/gpt-image-1-mini", api_key="os.environ/OPENAI_API_KEY"),
    )
    resources.defer(lambda: endpoints_client.delete_model(model_id))
    return model, resources.key()


class TestImageGeneration:
    @pytest.mark.covers("llm.images_generations.openai.basic.nonstream.works")
    def test_image_generation_returns_image(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_openai_image(endpoints_client, resources)
        result = endpoints_client.images(key, model, "Draw a cute cat")
        require_successful_call(result)
        _assert_image_returned(result.body)

    @pytest.mark.covers("llm.images_generations.bedrock.basic.nonstream.works", exercised_on=["images_generations"])
    def test_bedrock_image_generation_returns_image(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-bedrock-image-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="bedrock/amazon.nova-canvas-v1:0",
                aws_access_key_id="os.environ/AWS_ACCESS_KEY_ID",
                aws_secret_access_key="os.environ/AWS_SECRET_ACCESS_KEY",
                aws_region_name="os.environ/AWS_REGION",
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.images(key, model, "Draw a cute cat")
        require_successful_call(result)
        _assert_image_returned(result.body)

    @pytest.mark.skip(reason="stage red: product gap, /v1/images/generations 500s (aimage_generation TypeError) on missing prompt instead of 400")
    @pytest.mark.covers("llm.images_generations.openai.input_validation.nonstream.works")
    def test_missing_prompt_returns_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_openai_image(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/images/generations",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalImageBody(model=model),
        )
        assert_client_error(result, "images missing prompt")

    @pytest.mark.covers("llm.images_generations.openai.input_validation.nonstream.works")
    def test_empty_prompt_returns_client_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_openai_image(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/images/generations",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalImageBody(model=model, prompt=""),
        )
        assert_client_error(result, "images empty prompt")

    @pytest.mark.covers("llm.images_generations.openai.input_validation.nonstream.works")
    def test_invalid_size_returns_client_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_openai_image(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/images/generations",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalImageBody(model=model, prompt="a blue square", size="999x999"),
        )
        assert_client_error(result, "images invalid size")

    @pytest.mark.covers("llm.images_generations.openai.input_validation.nonstream.works")
    def test_invalid_n_returns_client_error(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model, key = _register_openai_image(endpoints_client, resources)
        result = endpoints_client.proxy.transport.send(
            "/v1/images/generations",
            headers=endpoints_client.proxy.transport.bearer(key),
            json=_OptionalImageBody(model=model, prompt="a blue square", n=0),
        )
        assert_client_error(result, "images invalid n")

"""Live e2e: POST /v1/images/generations returns an image.

Registers an OpenAI image deployment at runtime and asserts the response carries a
generated image (url or base64). Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest

from e2e_config import require_env, unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient, ImagesResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


def _assert_image_returned(body: str) -> None:
    parsed = ImagesResult.model_validate_json(body)
    assert parsed.data, f"/images/generations returned no data: {body[:300]}"
    first = parsed.data[0]
    assert first.b64_json or first.url, (
        f"generated image has neither b64_json nor url: {body[:300]}"
    )


class TestImageGeneration:
    def test_image_generation_returns_image(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-image-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="openai/gpt-image-1-mini", api_key="os.environ/OPENAI_API_KEY"
            ),
        )
        resources.defer(lambda: endpoints_client.delete_model(model_id))
        key = resources.key()

        result = endpoints_client.images(key, model, "Draw a cute cat")
        require_successful_call(result)
        _assert_image_returned(result.body)

    @pytest.mark.covers("llm.images_generations.bedrock.basic.nonstream.works", exercised_on=["images_generations"])
    def test_bedrock_image_generation_returns_image(
        self, endpoints_client: EndpointsClient, resources: ResourceManager
    ) -> None:
        require_env("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION")
        model = f"e2e-bedrock-image-{unique_marker()}"
        model_id = endpoints_client.create_model(
            model,
            LiteLLMParamsBody(
                model="bedrock/amazon.titan-image-generator-v2:0",
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

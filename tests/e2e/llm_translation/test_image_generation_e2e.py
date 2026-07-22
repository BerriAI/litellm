"""Live e2e: POST /v1/images/generations returns an image.

Registers an OpenAI image deployment at runtime and asserts the response carries a
generated image (url or base64). Migrated from
litellm-regression-tests/tests/test_inference_endpoints.py.
"""

from __future__ import annotations

import pytest

from e2e_config import unique_marker
from e2e_http import require_successful_call
from endpoints_client import EndpointsClient, ImagesResult
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class TestImageGeneration:
    @pytest.mark.covers("llm.images_generations.openai.basic.nonstream.works")
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
        parsed = ImagesResult.model_validate_json(result.body)
        assert parsed.data, f"/images/generations returned no data: {result.body[:300]}"
        first = parsed.data[0]
        assert first.b64_json or first.url, (
            f"generated image has neither b64_json nor url: {result.body[:300]}"
        )

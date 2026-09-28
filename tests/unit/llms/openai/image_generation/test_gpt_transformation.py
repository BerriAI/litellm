from unittest.mock import MagicMock

import httpx
import pytest

from litellm.llms.azure.image_generation.gpt_transformation import AzureGPTImageGenerationConfig
from litellm.llms.openai.image_generation.gpt_transformation import GPTImageGenerationConfig
from litellm.types.utils import ImageResponse


@pytest.mark.parametrize("config", [GPTImageGenerationConfig(), AzureGPTImageGenerationConfig()])
def test_transform_image_generation_response_keeps_provider_echo(config):
    raw_response = httpx.Response(
        status_code=200,
        json={
            "created": 1788457009,
            "data": [{"b64_json": "/9j/4AAQSkZJRg=="}],
            "output_format": "jpeg",
            "background": "opaque",
            "quality": "low",
            "size": "1024x1024",
        },
        request=httpx.Request("POST", "https://api.openai.com/v1/images/generations"),
    )

    image_response = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "a red apple", "output_format": "jpeg"},
        optional_params={"output_format": "jpeg"},
        litellm_params={},
        encoding=None,
    )

    assert image_response.output_format == "jpeg"
    assert image_response.quality == "low"
    assert image_response.background == "opaque"

from unittest.mock import MagicMock

import httpx

from litellm.llms.openai.image_generation.gpt_transformation import (
    GPTImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


def test_transform_image_generation_response_preserves_output_format():
    config = GPTImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={"created": 1, "data": [{"b64_json": "image-data"}]},
    )

    result = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data={"prompt": "A white cat"},
        optional_params={"output_format": "webp"},
        litellm_params={},
        encoding=None,
    )

    assert result.output_format == "webp"

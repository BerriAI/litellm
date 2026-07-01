import os
import sys
from typing import cast

import httpx

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.siliconflow.image_generation.transformation import (
    SiliconFlowImageGenerationConfig,
)
from litellm.types.utils import ImageResponse


class LoggingStub:
    def post_call(self, original_response: str) -> None:
        return None


class TestSiliconFlowImageGenerationConfig:
    def setup_method(self):
        self.config = SiliconFlowImageGenerationConfig()

    def test_map_openai_params_maps_size_and_n(self):
        optional_params = self.config.map_openai_params(
            non_default_params={"size": "1024x1024", "n": 2},
            optional_params={},
            model="Qwen/Qwen-Image",
            drop_params=True,
        )

        assert optional_params["image_size"] == "1024x1024"
        assert optional_params["batch_size"] == 2

    def test_transform_image_generation_response_reads_images(self):
        raw_response = httpx.Response(
            200,
            json={
                "images": [
                    {"url": "https://example.com/a.png"},
                    {"url": "https://example.com/b.png"},
                ]
            },
        )

        result = self.config.transform_image_generation_response(
            model="Qwen/Qwen-Image",
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=cast(LiteLLMLoggingObj, LoggingStub()),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

        assert result.data is not None
        first_image = result.data[0]
        second_image = result.data[1]
        assert len(result.data) == 2
        assert first_image.url == "https://example.com/a.png"
        assert second_image.url == "https://example.com/b.png"

from typing import List

from litellm.types.llms.openai import OpenAIImageVariationOptionalParams
from litellm.types.utils import ImageResponse

from ...base_llm.image_variations.transformation import BaseImageVariationConfig


class OpenAIImageVariationConfig(BaseImageVariationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageVariationOptionalParams]:
        return ["n", "size", "response_format", "user"]

    def transform_request_image_variation(self, *args, **kwargs) -> dict:
        return {}

    def transform_response_image_variation(self, *args, **kwargs) -> ImageResponse:
        return ImageResponse(**kwargs)

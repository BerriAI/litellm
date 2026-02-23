from typing import List

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams


class XInferenceImageGenerationConfig(BaseImageGenerationConfig):
    """
    XInference image generation config

    https://inference.readthedocs.io/en/v1.1.1/reference/generated/xinference.client.handlers.ImageModelHandle.text_to_image.html#xinference.client.handlers.ImageModelHandle.text_to_image
    """

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        return ["n", "response_format", "size", "response_format"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

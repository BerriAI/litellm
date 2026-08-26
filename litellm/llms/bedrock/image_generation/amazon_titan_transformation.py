"""
Transformation logic for Amazon Titan Image Generation.
"""

import types
from typing import Final

from openai.types.image import Image

from litellm.types.llms.bedrock import (
    AmazonNovaCanvasImageGenerationConfig,
    AmazonTitanImageGenerationRequestBody,
    AmazonTitanTextToImageParams,
)
from litellm.types.utils import ImageResponse
from litellm.utils import get_model_info


class AmazonTitanImageGenerationConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=stability.stable-diffusion-xl-v0
    """

    cfg_scale: int | None = None
    seed: float | None = None
    steps: list[str] | None = None
    width: int | None = None
    height: int | None = None

    def __init__(
        self,
        cfg_scale: int | None = None,
        seed: float | None = None,
        steps: list[str] | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        locals_: Final = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    @classmethod
    def _is_titan_model(cls, model: str | None = None) -> bool:
        """
        Returns True if the model is a Titan model

        Titan models follow this pattern:

        """
        if model and "amazon.titan" in model:
            return True
        return False

    @classmethod
    def get_supported_openai_params(cls, model: str | None = None) -> list:
        return ["size", "n", "quality"]

    @classmethod
    def map_openai_params(
        cls,
        non_default_params: dict,
        optional_params: dict,
    ):
        from typing import Any

        image_generation_config: Final[dict[str, Any]] = {}
        for k, v in non_default_params.items():
            if k == "size" and v is not None:
                width, height = v.split("x")
                image_generation_config["width"] = int(width)
                image_generation_config["height"] = int(height)
            elif k == "n" and v is not None:
                image_generation_config["numberOfImages"] = v
            elif k == "quality" and v is not None:  # 'auto', 'hd', 'standard', 'high', 'medium', 'low'
                if v in ("hd", "premium", "high"):
                    image_generation_config["quality"] = "premium"
                elif v in ("standard", "medium", "low"):
                    image_generation_config["quality"] = "standard"

        if image_generation_config:
            optional_params["imageGenerationConfig"] = image_generation_config
        return optional_params

    @classmethod
    def transform_request_body(
        cls,
        text: str,
        optional_params: dict,
    ) -> AmazonTitanImageGenerationRequestBody:
        from typing import Any

        image_generation_config = optional_params.pop("imageGenerationConfig", {})
        negative_text: Final = optional_params.pop("negativeText", None)
        text_to_image_params: Final[dict[str, Any]] = {"text": text}
        if negative_text:
            text_to_image_params["negativeText"] = negative_text
        task_type: Final = optional_params.pop("taskType", "TEXT_IMAGE")
        user_specified_image_generation_config: Final = optional_params.pop("imageGenerationConfig", {})
        image_generation_config = {
            **image_generation_config,
            **user_specified_image_generation_config,
        }
        return AmazonTitanImageGenerationRequestBody(
            taskType=task_type,
            textToImageParams=AmazonTitanTextToImageParams(**text_to_image_params),
            imageGenerationConfig=AmazonNovaCanvasImageGenerationConfig(**image_generation_config),
        )

    @classmethod
    def transform_response_dict_to_openai_response(
        cls, model_response: ImageResponse, response_dict: dict
    ) -> ImageResponse:
        image_list: Final[list[Image]] = []
        for image in response_dict["images"]:
            _image = Image(b64_json=image)
            image_list.append(_image)

        model_response.data = image_list

        return model_response

    @classmethod
    def cost_calculator(
        cls,
        model: str,
        image_response: ImageResponse,
        size: str | None = None,
        optional_params: dict | None = None,
    ) -> float:
        model_info: Final = get_model_info(model=model)
        output_cost_per_image: Final = model_info.get("output_cost_per_image") or 0.0
        if not image_response.data:
            return 0.0
        num_images: Final = len(image_response.data)
        return output_cost_per_image * num_images

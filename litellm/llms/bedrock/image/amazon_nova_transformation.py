import types
from typing import List, Optional

from openai.types.image import Image

from litellm.types.llms.bedrock import (
    AmazonNovaTextToImageRequest, AmazonNovaTextToImageResponse,
)
from litellm.types.utils import ImageResponse


class AmazonNovaConfig:
    """
    Reference: https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/model-catalog/serverless/amazon.nova-canvas-v1:0

    """

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
    def get_supported_openai_params(cls, model: Optional[str] = None) -> List:
        """
        No additional OpenAI params are mapped for Nova
        """
        return []

    @classmethod
    def _is_nova_model(cls, model: Optional[str] = None) -> bool:
        """
        Returns True if the model is a Nova model

        Nova models follow this pattern:

        """
        if model:
            if "amazon.nova" in model:
                return True
        return False

    @classmethod
    def transform_request_body(
        cls, text: str, optional_params: dict
    ) -> AmazonNovaTextToImageRequest:
        """
        Transform the request body for Nova models
        """
        data = AmazonNovaTextToImageRequest(textToImageParams={"text": text}, **optional_params)
        return data

    @classmethod
    def map_openai_params(cls, non_default_params: dict, optional_params: dict) -> dict:
        """
        Map the OpenAI params to the Bedrock params

        No OpenAI params are mapped for Nova, so directly return the optional_params
        """
        return optional_params

    @classmethod
    def transform_response_dict_to_openai_response(
        cls, model_response: ImageResponse, response_dict: dict
    ) -> ImageResponse:
        """
        Transform the response dict to the OpenAI response
        """

        nova_response = AmazonNovaTextToImageResponse(**response_dict)
        openai_images: List[Image] = []
        for _img in nova_response.get("images", []):
            openai_images.append(Image(b64_json=_img))

        model_response.data = openai_images
        return model_response

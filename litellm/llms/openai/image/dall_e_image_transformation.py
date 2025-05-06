import types
from typing import List, Optional

from litellm.types.utils import ImageResponse


class DallEImageConfig:
    """
    Configuration class for OpenAI's DALL-E models.
    
    Reference: https://platform.openai.com/docs/api-reference/images/create
    
    Supported parameters for OpenAI GPT-Image-1:
    - `prompt` (required): The prompt to generate images from.
    - `model` (required): The model to use for image generation.
    - `n`: Number of images to generate. Defaults to 1.
    - `quality`: The quality of the image that will be generated. The quality of the image that will be generated.
        - auto (default value) will automatically select the best quality for the given model.
        - high, medium and low are supported for gpt-image-1.
        - hd and standard are supported for dall-e-3.
        - standard is the only option for dall-e-2.
    - `response_format`: The format in which generated images with dall-e-2 and dall-e-3 are returned. Must be one of url or b64_json. URLs are only valid for 60 minutes after the image has been generated. This parameter isn't supported for gpt-image-1 which will always return base64-encoded images.
    - `size`: The size of the generated images. Must be one of 1024x1024, 1536x1024 (landscape), 1024x1536 (portrait), or auto (default value) for gpt-image-1, one of 256x256, 512x512, or 1024x1024 for dall-e-2, and one of 1024x1024, 1792x1024, or 1024x1792 for dall-e-3.
    - `style`: The style of the generated images. This parameter is only supported for dall-e-3. Must be one of vivid or natural. Vivid causes the model to lean towards generating hyper-real and dramatic images. Natural causes the model to produce more natural, less hyper-real looking images.
    - `user`: A unique identifier representing your end-user.
    
    Note: 'response_format' is not supported by gpt-image-1
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
        Returns the list of OpenAI parameters supported by the model.
        """
        return ["n", "quality", "response_format", "size", "style", "user"]
    
    @classmethod
    def _is_dall_e_image_model(cls, model: Optional[str] = None) -> bool:
        """
        Returns True if the model is a DALL-E Image model
        """
        if model:
            if "dall-e" in model:
                return True
        return False
    
    @classmethod
    def map_openai_params(cls, non_default_params: dict, optional_params: dict) -> dict:
        """Map openai params"""
        optional_params.update(non_default_params)
        return optional_params
    
    @classmethod
    def transform_response_dict_to_openai_response(
        cls, model_response: ImageResponse, response_dict: dict
    ) -> ImageResponse:
        """
        Transform the response dict to the OpenAI response.
        Since we're already dealing with OpenAI/Azure providers, minimal transformation is needed.
        """
        # The response structure should already be compatible with OpenAI format
        return model_response
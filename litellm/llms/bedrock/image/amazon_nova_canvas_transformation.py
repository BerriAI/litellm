import types
from typing import Any, Dict, List, Optional

from openai.types.image import Image

from litellm.types.llms.bedrock import (
    AmazonNovaCanvasColorGuidedGenerationParams,
    AmazonNovaCanvasColorGuidedRequest,
    AmazonNovaCanvasImageGenerationConfig,
    AmazonNovaCanvasRequestBase,
    AmazonNovaCanvasTextToImageParams,
    AmazonNovaCanvasTextToImageRequest,
    AmazonNovaCanvasTextToImageResponse,
    AmazonNovaCanvasInpaintingParams,
    AmazonNovaCanvasInpaintingRequest,
)
from litellm.types.utils import ImageResponse


class AmazonNovaCanvasConfig:
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
        """ """
        return ["n", "size", "quality"]

    @classmethod
    def _is_nova_model(cls, model: Optional[str] = None) -> bool:
        """
        Returns True if the model is a Nova Canvas model

        Nova models follow this pattern:

        """
        if model and "amazon.nova-canvas" in model:
            return True
        return False

    @classmethod
    def transform_request_body(
        cls, text: str, optional_params: dict
    ) -> AmazonNovaCanvasRequestBase:
        """
        Transform the request body for Amazon Nova Canvas model
        """
        task_type = optional_params.pop("taskType", "TEXT_IMAGE")
        image_generation_config = optional_params.pop("imageGenerationConfig", {})
        image_generation_config = {**image_generation_config, **optional_params}
        if task_type == "TEXT_IMAGE":
            text_to_image_params: Dict[str, Any] = image_generation_config.pop(
                "textToImageParams", {}
            )
            text_to_image_params = {"text": text, **text_to_image_params}
            try:
                text_to_image_params_typed = AmazonNovaCanvasTextToImageParams(
                    **text_to_image_params  # type: ignore
                )
            except Exception as e:
                raise ValueError(
                    f"Error transforming text to image params: {e}. Got params: {text_to_image_params}, Expected params: {AmazonNovaCanvasTextToImageParams.__annotations__}"
                )

            try:
                image_generation_config_typed = AmazonNovaCanvasImageGenerationConfig(
                    **image_generation_config
                )
            except Exception as e:
                raise ValueError(
                    f"Error transforming image generation config: {e}. Got params: {image_generation_config}, Expected params: {AmazonNovaCanvasImageGenerationConfig.__annotations__}"
                )

            return AmazonNovaCanvasTextToImageRequest(
                textToImageParams=text_to_image_params_typed,
                taskType=task_type,
                imageGenerationConfig=image_generation_config_typed,
            )
        if task_type == "COLOR_GUIDED_GENERATION":
            color_guided_generation_params: Dict[
                str, Any
            ] = image_generation_config.pop("colorGuidedGenerationParams", {})
            color_guided_generation_params = {
                "text": text,
                **color_guided_generation_params,
            }
            try:
                color_guided_generation_params_typed = AmazonNovaCanvasColorGuidedGenerationParams(
                    **color_guided_generation_params  # type: ignore
                )
            except Exception as e:
                raise ValueError(
                    f"Error transforming color guided generation params: {e}. Got params: {color_guided_generation_params}, Expected params: {AmazonNovaCanvasColorGuidedGenerationParams.__annotations__}"
                )

            try:
                image_generation_config_typed = AmazonNovaCanvasImageGenerationConfig(
                    **image_generation_config
                )
            except Exception as e:
                raise ValueError(
                    f"Error transforming image generation config: {e}. Got params: {image_generation_config}, Expected params: {AmazonNovaCanvasImageGenerationConfig.__annotations__}"
                )

            return AmazonNovaCanvasColorGuidedRequest(
                taskType=task_type,
                colorGuidedGenerationParams=color_guided_generation_params_typed,
                imageGenerationConfig=image_generation_config_typed,
            )
        if task_type == "INPAINTING":
            inpainting_params: Dict[str, Any] = image_generation_config.pop(
                "inpaintingParams", {}
            )
            inpainting_params = {"text": text, **inpainting_params}
            try:
                inpainting_params_typed = AmazonNovaCanvasInpaintingParams(
                    **inpainting_params  # type: ignore
                )
            except Exception as e:
                raise ValueError(
                    f"Error transforming inpainting params: {e}. Got params: {inpainting_params}, Expected params: {AmazonNovaCanvasInpaintingParams.__annotations__}"
                )

            try:
                image_generation_config_typed = AmazonNovaCanvasImageGenerationConfig(
                    **image_generation_config
                )
            except Exception as e:
                raise ValueError(
                    f"Error transforming image generation config: {e}. Got params: {image_generation_config}, Expected params: {AmazonNovaCanvasImageGenerationConfig.__annotations__}"
                )

            return AmazonNovaCanvasInpaintingRequest(
                taskType=task_type,
                inpaintingParams=inpainting_params_typed,
                imageGenerationConfig=image_generation_config_typed,
            )
        raise NotImplementedError(f"Task type {task_type} is not supported")

    @classmethod
    def map_openai_params(cls, non_default_params: dict, optional_params: dict) -> dict:
        """
        Map the OpenAI params to the Bedrock params
        """
        _size = non_default_params.get("size")
        if _size is not None:
            width, height = _size.split("x")
            optional_params["width"], optional_params["height"] = int(width), int(
                height
            )
        if non_default_params.get("n") is not None:
            optional_params["numberOfImages"] = non_default_params.get("n")
        if non_default_params.get("quality") is not None:
            if non_default_params.get("quality") in ("hd", "premium"):
                optional_params["quality"] = "premium"
            if non_default_params.get("quality") == "standard":
                optional_params["quality"] = "standard"
        return optional_params

    @classmethod
    def transform_response_dict_to_openai_response(
        cls, model_response: ImageResponse, response_dict: dict
    ) -> ImageResponse:
        """
        Transform the response dict to the OpenAI response
        """

        nova_response = AmazonNovaCanvasTextToImageResponse(**response_dict)
        openai_images: List[Image] = []
        for _img in nova_response.get("images", []):
            openai_images.append(Image(b64_json=_img))

        model_response.data = openai_images
        return model_response

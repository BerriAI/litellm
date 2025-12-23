import copy
import os
import types
from typing import List, Optional

from openai.types.image import Image

from litellm.llms.bedrock.common_utils import get_cached_model_info
from litellm.types.utils import ImageResponse


class AmazonStabilityConfig:
    """
    Reference: https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=stability.stable-diffusion-xl-v0

    Supported Params for the Amazon / Stable Diffusion models:

    - `cfg_scale` (integer): Default `7`. Between [ 0 .. 35 ]. How strictly the diffusion process adheres to the prompt text (higher values keep your image closer to your prompt)

    - `seed` (float): Default: `0`. Between [ 0 .. 4294967295 ]. Random noise seed (omit this option or use 0 for a random seed)

    - `steps` (array of strings): Default `30`. Between [ 10 .. 50 ]. Number of diffusion steps to run.

    - `width` (integer): Default: `512`. multiple of 64 >= 128. Width of the image to generate, in pixels, in an increment divible by 64.
        Engine-specific dimension validation:

        - SDXL Beta: must be between 128x128 and 512x896 (or 896x512); only one dimension can be greater than 512.
        - SDXL v0.9: must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, or 896x1152
        - SDXL v1.0: same as SDXL v0.9
        - SD v1.6: must be between 320x320 and 1536x1536

    - `height` (integer): Default: `512`. multiple of 64 >= 128. Height of the image to generate, in pixels, in an increment divible by 64.
        Engine-specific dimension validation:

        - SDXL Beta: must be between 128x128 and 512x896 (or 896x512); only one dimension can be greater than 512.
        - SDXL v0.9: must be one of 1024x1024, 1152x896, 1216x832, 1344x768, 1536x640, 640x1536, 768x1344, 832x1216, or 896x1152
        - SDXL v1.0: same as SDXL v0.9
        - SD v1.6: must be between 320x320 and 1536x1536
    """

    cfg_scale: Optional[int] = None
    seed: Optional[float] = None
    steps: Optional[List[str]] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def __init__(
        self,
        cfg_scale: Optional[int] = None,
        seed: Optional[float] = None,
        steps: Optional[List[str]] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> None:
        locals_ = locals().copy()
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
    def get_supported_openai_params(cls, model: Optional[str] = None) -> List:
        return ["size"]

    @classmethod
    def map_openai_params(
        cls,
        non_default_params: dict,
        optional_params: dict,
    ):
        _size = non_default_params.get("size")
        if _size is not None:
            width, height = _size.split("x")
            optional_params["width"] = int(width)
            optional_params["height"] = int(height)

        return optional_params

    @classmethod
    def transform_request_body(
        cls,
        text: str,
        optional_params: dict,
    ) -> dict:
        inference_params = copy.deepcopy(optional_params)
        inference_params.pop(
            "user", None
        )  # make sure user is not passed in for bedrock call

        prompt = text.replace(os.linesep, " ")
        ## LOAD CONFIG
        config = cls.get_config()
        for k, v in config.items():
            if (
                k not in inference_params
            ):  # completion(top_k=3) > anthropic_config(top_k=3) <- allows for dynamic variables to be passed in
                inference_params[k] = v

        return {
            "text_prompts": [{"text": prompt, "weight": 1}],
             **inference_params,
        }

    @classmethod
    def transform_response_dict_to_openai_response(
        cls, model_response: ImageResponse, response_dict: dict
    ) -> ImageResponse:
        image_list: List[Image] = []
        for artifact in response_dict["artifacts"]:
            _image = Image(b64_json=artifact["base64"])
            image_list.append(_image)

        model_response.data = image_list

        return model_response

    @classmethod
    def cost_calculator(
        cls,
        model: str,
        image_response: ImageResponse,
        size: Optional[str] = None,
        optional_params: Optional[dict] = None,
    ) -> float:
        optional_params = optional_params or {}

        # see model_prices_and_context_window.json for details on how steps is used
        # Reference pricing by steps for stability 1: https://aws.amazon.com/bedrock/pricing/
        _steps = optional_params.get("steps", 50)
        steps = "max-steps" if _steps > 50 else "50-steps"

        # size is stored in model_prices_and_context_window.json as 1024-x-1024
        # current size has 1024x1024
        size = size or "1024-x-1024"
        model = f"{size}/{steps}/{model}"

        get_model_info = get_cached_model_info()
        model_info = get_model_info(
            model=model,
            custom_llm_provider="bedrock",
        )

        output_cost_per_image: float = model_info.get("output_cost_per_image") or 0.0
        num_images: int = 0
        if image_response.data:
            num_images = len(image_response.data)
        return output_cost_per_image * num_images
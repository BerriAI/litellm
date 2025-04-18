from typing import Any, List, Optional
from PIL import Image
import io
import base64

from litellm.llms.base_llm.image_variations.transformation import LiteLLMLoggingObj
from litellm.types.utils import FileTypes, ImageResponse

from ...base_llm.image_variations.transformation import BaseImageVariationConfig
from ..common_utils import LLMError


class DiffusersImageVariationConfig(BaseImageVariationConfig):
    def get_supported_diffusers_params(self) -> List[str]:
        """Return supported parameters for diffusers pipeline"""
        return [
            "prompt",
            "height",
            "width",
            "num_inference_steps",
            "guidance_scale",
            "negative_prompt",
            "num_images_per_prompt",
            "eta",
            "seed",
        ]

    def transform_request_image_variation(
        self,
        model: Optional[str],
        image: FileTypes,
        optional_params: dict,
        headers: dict,
    ) -> dict:
        """Convert input to format expected by diffusers"""
        # Convert image to PIL if needed
        if not isinstance(image, Image.Image):
            if isinstance(image, str):  # file path
                image = Image.open(image)
            elif isinstance(image, bytes):  # raw bytes
                image = Image.open(io.BytesIO(image))

        return {
            "image": image,
            "model": model or "runwayml/stable-diffusion-v1-5",
            "params": {
                k: v
                for k, v in optional_params.items()
                if k in self.get_supported_diffusers_params()
            },
        }

    def transform_response_image_variation(
        self,
        model: Optional[str],
        raw_response: Any,  # Not used for local
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        image: FileTypes,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
    ) -> ImageResponse:
        """Convert diffusers output to standardized ImageResponse"""
        # For diffusers, model_response should be PIL Image or list of PIL Images
        if isinstance(model_response, list):
            images = model_response
        else:
            images = [model_response]

        # Convert to base64
        image_data = []
        for img in images:
            buffered = io.BytesIO()
            img.save(buffered, format="PNG")
            image_data.append(
                {"b64_json": base64.b64encode(buffered.getvalue()).decode("utf-8")}
            )

        return ImageResponse(created=int(time.time()), data=image_data)

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict
    ) -> LLMError:
        """Return generic LLM error for diffusers"""
        return LLMError(status_code=status_code, message=error_message, headers=headers)

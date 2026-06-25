from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.aiml import AimlImageGenerationRequestParams
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


OPENAI_STYLE_IMAGE_MODEL_PREFIXES: tuple[str, ...] = ("openai/",)


class AimlImageGenerationConfig(BaseImageGenerationConfig):
    DEFAULT_BASE_URL: str = "https://api.aimlapi.com"
    IMAGE_GENERATION_ENDPOINT: str = "v1/images/generations"

    @staticmethod
    def _is_openai_style_model(model: str) -> bool:
        """
        OpenAI image models routed through AI/ML API (e.g. ``openai/gpt-image-2``)
        use the upstream OpenAI request schema, not the flux-style schema used by
        the rest of the AI/ML catalog.
        """
        return model.startswith(OPENAI_STYLE_IMAGE_MODEL_PREFIXES)

    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        https://api.aimlapi.com/v1/images/generations
        """
        if self._is_openai_style_model(model):
            return [
                "n",
                "size",
                "quality",
                "response_format",
                "output_format",
                "background",
                "moderation",
                "output_compression",
            ]
        return ["n", "response_format", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        is_openai_style = self._is_openai_style_model(model)

        for k in non_default_params.keys():
            if k in optional_params.keys():
                continue
            if k not in supported_params:
                if drop_params:
                    continue
                raise ValueError(
                    f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                )

            if is_openai_style:
                optional_params[k] = non_default_params[k]
                continue

            if k == "n":
                optional_params["num_images"] = non_default_params[k]
            elif k == "response_format":
                optional_params["output_format"] = non_default_params[k]
            elif k == "size":
                size_value = non_default_params[k]
                if isinstance(size_value, str) and "x" in size_value:
                    width, height = map(int, size_value.split("x"))
                    optional_params["image_size"] = {
                        "width": width,
                        "height": height,
                    }
                else:
                    optional_params["image_size"] = size_value
            else:
                optional_params[k] = non_default_params[k]

        return optional_params

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request
        """
        complete_url: str = (
            api_base or get_secret_str("AIML_API_BASE") or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")
        # Strip /v1 suffix if present since IMAGE_GENERATION_ENDPOINT already includes v1
        if complete_url.endswith("/v1"):
            complete_url = complete_url[:-3]
        complete_url = f"{complete_url}/{self.IMAGE_GENERATION_ENDPOINT}"
        return complete_url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        final_api_key: Optional[str] = (
            api_key
            or get_secret_str("AIML_API_KEY")
            or get_secret_str("AIMLAPI_KEY")  # Alternative name
        )
        if not final_api_key:
            raise ValueError("AIML_API_KEY or AIMLAPI_KEY is not set")

        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to the AI/ML image generation request body

        https://api.aimlapi.com/v1/images/generations
        """
        if self._is_openai_style_model(model):
            return {"model": model, "prompt": prompt, **optional_params}

        aiml_image_generation_request_body: AimlImageGenerationRequestParams = (
            AimlImageGenerationRequestParams(
                prompt=prompt,
                model=model,
                **optional_params,
            )
        )
        return dict(aiml_image_generation_request_body)

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Transform the image generation response to the litellm image response

        https://api.aimlapi.com/v1/images/generations
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        # AI/ML API can return images in multiple formats:
        # 1. Top-level data array with url (OpenAI-like format)
        # 2. output.choices array with image_base64
        # 3. images array with url (and optional width, height, content_type)

        if "data" in response_data and isinstance(response_data["data"], list):
            # Handle OpenAI-like format: {"data": [{"url": "...", "width": 1024, "height": 768, "content_type": "image/jpeg"}]}
            for image in response_data["data"]:
                if "url" in image:
                    model_response.data.append(
                        ImageObject(
                            b64_json=None,
                            url=image["url"],
                            revised_prompt=image.get("revised_prompt"),
                        )
                    )
                elif "b64_json" in image or "image_base64" in image:
                    model_response.data.append(
                        ImageObject(
                            b64_json=image.get("b64_json") or image.get("image_base64"),
                            url=None,
                            revised_prompt=image.get("revised_prompt"),
                        )
                    )
        elif "output" in response_data and "choices" in response_data["output"]:
            for choice in response_data["output"]["choices"]:
                if "image_base64" in choice:
                    model_response.data.append(
                        ImageObject(
                            b64_json=choice["image_base64"],
                            url=None,
                        )
                    )
                elif "url" in choice:
                    model_response.data.append(
                        ImageObject(
                            b64_json=None,
                            url=choice["url"],
                        )
                    )
        elif "images" in response_data:
            # Handle alternative format: {"images": [{"url": "...", "width": 1024, "height": 768, "content_type": "image/jpeg"}]}
            for image in response_data["images"]:
                if "url" in image:
                    model_response.data.append(
                        ImageObject(
                            b64_json=None,
                            url=image["url"],
                        )
                    )
                elif "image_base64" in image:
                    model_response.data.append(
                        ImageObject(
                            b64_json=image["image_base64"],
                            url=None,
                        )
                    )
        return model_response

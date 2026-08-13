from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.llms.recraft import RecraftImageGenerationRequestParams
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class RecraftImageGenerationConfig(BaseImageGenerationConfig):
    DEFAULT_BASE_URL: str = "https://external.api.recraft.ai"
    IMAGE_GENERATION_ENDPOINT: str = "v1/images/generations"

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        """
        https://www.recraft.ai/docs#generate-image
        """
        return ["n", "response_format", "size", "style"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params: Final = self.get_supported_openai_params(model)
        for k in non_default_params:
            if k not in optional_params:
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        """
        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        complete_url: str = api_base or get_secret_str("RECRAFT_API_BASE") or self.DEFAULT_BASE_URL

        complete_url = complete_url.rstrip("/")
        complete_url = f"{complete_url}/{self.IMAGE_GENERATION_ENDPOINT}"
        return complete_url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key: Final[str | None] = api_key or get_secret_str("RECRAFT_API_KEY")
        if not final_api_key:
            raise ValueError("RECRAFT_API_KEY is not set")

        headers["Authorization"] = f"Bearer {final_api_key}"
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
        Transform the image generation request to the recraft image generation request body

        https://www.recraft.ai/docs#generate-image
        """
        recratft_image_generation_request_body: Final[RecraftImageGenerationRequestParams] = (
            RecraftImageGenerationRequestParams(
                prompt=prompt,
                model=model,
                **optional_params,
            )
        )
        return dict(recratft_image_generation_request_body)

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
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        """
        Transform the image generation response to the litellm image response

        https://www.recraft.ai/docs#generate-image
        """
        try:
            response_data: Final = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        if not model_response.data:
            model_response.data = []

        for image_data in response_data["data"]:
            model_response.data.append(
                ImageObject(
                    url=image_data.get("url", None),
                    b64_json=image_data.get("b64_json", None),
                )
            )

        return model_response

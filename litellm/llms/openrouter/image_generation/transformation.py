"""
OpenRouter Image Generation Support

OpenRouter provides a dedicated image generation endpoint at /api/v1/images.

Request format:
{
    "model": "bytedance-seed/seedream-4.5",
    "prompt": "A beautiful sunset over mountains",
    "n": 1,
    "size": "1024x1024",
    "quality": "auto"
}

Response format:
{
    "created": 1748372400,
    "data": [{"b64_json": "<base64>", "media_type": "image/png"}],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 4175,
        "total_tokens": 4175,
        "cost": 0.04
    }
}
"""

from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


QUALITY_ALIASES: dict[str, str] = {
    "standard": "low",
    "hd": "high",
}


class OpenRouterImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(self, model: str) -> List[OpenAIImageGenerationOptionalParams]:
        return [
            "n",
            "quality",
            "size",
            "background",
            "output_compression",
            "output_format",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)

        for key, value in non_default_params.items():
            if key in supported_params:
                if key == "quality":
                    optional_params["quality"] = QUALITY_ALIASES.get(value, value)
                else:
                    optional_params[key] = value
            elif not drop_params:
                optional_params[key] = value

        return optional_params

    def _set_usage_and_cost(
        self,
        model_response: ImageResponse,
        response_json: dict,
        model: str,
    ) -> None:
        usage_data = response_json.get("usage", {})
        if usage_data:
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            total_tokens = usage_data.get("total_tokens", 0)

            model_response.usage = ImageUsage(
                input_tokens=prompt_tokens,
                input_tokens_details=ImageUsageInputTokensDetails(
                    image_tokens=0,
                    text_tokens=prompt_tokens,
                ),
                output_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

            cost = usage_data.get("cost")
            if cost is not None:
                if not hasattr(model_response, "_hidden_params"):
                    model_response._hidden_params = {}
                if "additional_headers" not in model_response._hidden_params:
                    model_response._hidden_params["additional_headers"] = {}
                model_response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] = float(
                    cost
                )

            cost_details = usage_data.get("cost_details", {})
            if cost_details:
                if "response_cost_details" not in model_response._hidden_params:
                    model_response._hidden_params["response_cost_details"] = {}
                model_response._hidden_params["response_cost_details"].update(cost_details)

        model_response._hidden_params["model"] = response_json.get("model", model)

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        if api_base:
            base = api_base.rstrip("/")
            if base.endswith("/images"):
                return base
            if base.endswith("/chat/completions"):
                # Backwards compatibility: image generation previously routed
                # through /chat/completions, so older configs may still point
                # api_base at that path.
                base = base[: -len("/chat/completions")]
            return f"{base}/images"

        return "https://openrouter.ai/api/v1/images"

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
        api_key = api_key or litellm.api_key or get_secret_str("OPENROUTER_API_KEY")
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
            }
        )
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        request_body: dict[str, object] = {
            "model": model,
            "prompt": prompt,
        }

        for key, value in optional_params.items():
            if key not in ("model", "prompt"):
                request_body[key] = value

        return request_body

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
        try:
            response_json = raw_response.json()
        except Exception as e:
            raise OpenRouterException(
                message=f"Error parsing OpenRouter response: {str(e)}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        if not model_response.data:
            model_response.data = []

        try:
            for item in response_json.get("data", []):
                b64 = item.get("b64_json")
                url = item.get("url")
                model_response.data.append(
                    ImageObject(
                        b64_json=b64,
                        url=url,
                        revised_prompt=None,
                    )
                )

            self._set_usage_and_cost(model_response, response_json, model)
            return model_response

        except Exception as e:
            raise OpenRouterException(
                message=f"Error transforming OpenRouter image generation response: {str(e)}",
                status_code=500,
                headers={},
            )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

"""
OpenRouter Image Generation Support

OpenRouter serves image generation from a dedicated endpoint,
``POST https://openrouter.ai/api/v1/images``. Models whose only output modality is
``image`` (``krea/*``, ``openai/gpt-image-*``, ``bytedance-seed/*``, ...) are reachable
*only* there; sending them to ``/chat/completions`` answers with an opaque HTTP 500.
Hybrid text+image models such as ``google/gemini-2.5-flash-image`` are served from both,
so this config routes every OpenRouter image generation through ``/images``.

Request format:
{
    "model": "krea/krea-2-large",
    "prompt": "a red panda astronaut floating in space",
    "n": 1,
    "aspect_ratio": "1:1"
}

Params a model does not advertise are tolerated rather than rejected, so provider-native
knobs (``resolution``, ``seed``, ``output_format``, ...) can be set on the deployment and
pass straight through.

Response format:
{
    "created": 0,
    "data": [{"b64_json": "iVBORw0KGgo...", "media_type": "image/png"}],
    "usage": {
        "prompt_tokens": 0,
        "completion_tokens": 4175,
        "total_tokens": 4175,
        "cost": 0.06,
        "completion_tokens_details": {"image_tokens": 4175},
        "cost_details": {"upstream_inference_cost": 0.06}
    }
}
"""

import re
from typing import TYPE_CHECKING, Any, Final, Optional, Union

import httpx
from pydantic import ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError
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
from litellm.types.llms.openrouter import OpenRouterImagesResponse
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


DEFAULT_OPENROUTER_API_BASE: Final = "https://openrouter.ai/api/v1"

SUPPORTED_ASPECT_RATIOS: Final[tuple[tuple[int, int], ...]] = (
    (1, 1),
    (2, 3),
    (3, 2),
    (3, 4),
    (4, 3),
    (4, 5),
    (5, 4),
    (9, 16),
    (16, 9),
    (21, 9),
)

PIXEL_SIZE_PATTERN: Final = re.compile(r"(\d+)x(\d+)")

NON_BODY_PARAMS: Final = frozenset({"model", "prompt", "extra_headers"})


class OpenRouterImageGenerationConfig(BaseImageGenerationConfig):
    """Maps ``/v1/images/generations`` onto OpenRouter's ``/api/v1/images`` endpoint."""

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return ["n", "quality", "response_format", "size"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Translate OpenAI image params into OpenRouter's image API params.

        ``size`` becomes ``aspect_ratio``, which no OpenRouter image model accepts as pixel
        dimensions. ``quality`` passes through untouched because OpenRouter reuses OpenAI's
        ``auto``/``low``/``medium``/``high`` enum for it. ``response_format`` is dropped
        because the endpoint always answers with base64 image data.
        """
        mapped = dict(optional_params)

        for key, value in non_default_params.items():
            if key == "size":
                aspect_ratio = self._map_size_to_aspect_ratio(str(value))
                if aspect_ratio is not None:
                    mapped["aspect_ratio"] = aspect_ratio
            elif key == "response_format":
                self._reject_unsupported_response_format(value=str(value), model=model, drop_params=drop_params)
            else:
                mapped[key] = value

        return mapped

    def _map_size_to_aspect_ratio(self, size: str) -> Optional[str]:
        """
        Map an OpenAI ``WxH`` size onto the closest aspect ratio OpenRouter accepts.

        Returns ``None`` for ``auto`` and for anything that is not a pixel size, so the
        model's own default applies instead of a guessed ratio.
        """
        match = PIXEL_SIZE_PATTERN.fullmatch(size.strip())
        if match is None:
            return None

        width, height = int(match.group(1)), int(match.group(2))
        if width == 0 or height == 0:
            return None

        target = width / height
        closest = min(SUPPORTED_ASPECT_RATIOS, key=lambda ratio: abs(ratio[0] / ratio[1] - target))
        return f"{closest[0]}:{closest[1]}"

    def _reject_unsupported_response_format(self, value: str, model: str, drop_params: bool) -> None:
        if value == "b64_json" or drop_params:
            return
        raise UnsupportedParamsError(
            model=model,
            llm_provider="openrouter",
            message=(
                f"OpenRouter's image API always returns base64 image data, so response_format="
                f"'{value}' is not supported. Request 'b64_json', or set `drop_params: true` to ignore it."
            ),
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        base_url = (api_base or get_secret_str("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE).rstrip("/")
        if base_url.endswith("/images"):
            return base_url
        return f"{base_url}/images"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        resolved_api_key = api_key or litellm.api_key or get_secret_str("OPENROUTER_API_KEY")
        headers.update(
            {
                "Authorization": f"Bearer {resolved_api_key}",
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
        return {
            "model": model,
            "prompt": prompt,
            **{key: value for key, value in optional_params.items() if key not in NON_BODY_PARAMS},
        }

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
            parsed = OpenRouterImagesResponse.model_validate(raw_response.json())
        except (ValueError, ValidationError) as e:
            raise OpenRouterException(
                message=f"Error parsing OpenRouter image response: {str(e)}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )

        model_response.data = [
            ImageObject(
                b64_json=image.b64_json,
                url=image.url,
                revised_prompt=image.revised_prompt,
            )
            for image in parsed.data
        ]

        if parsed.created:
            model_response.created = parsed.created

        self._set_usage_and_cost(model_response=model_response, parsed=parsed, model=model)

        return model_response

    def _set_usage_and_cost(
        self,
        model_response: ImageResponse,
        parsed: OpenRouterImagesResponse,
        model: str,
    ) -> None:
        model_response._hidden_params["model"] = parsed.model or model

        usage = parsed.usage
        if usage is None:
            return

        image_tokens = usage.completion_tokens_details.image_tokens if usage.completion_tokens_details else None
        model_response.usage = ImageUsage(
            input_tokens=usage.prompt_tokens,
            input_tokens_details=ImageUsageInputTokensDetails(
                image_tokens=0,
                text_tokens=usage.prompt_tokens,
            ),
            output_tokens=image_tokens if image_tokens is not None else usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

        if usage.cost is not None:
            additional_headers = dict(model_response._hidden_params.get("additional_headers") or {})
            additional_headers["llm_provider-x-litellm-response-cost"] = usage.cost
            model_response._hidden_params["additional_headers"] = additional_headers

        if usage.cost_details:
            model_response._hidden_params["response_cost_details"] = {
                **(model_response._hidden_params.get("response_cost_details") or {}),
                **usage.cost_details,
            }

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

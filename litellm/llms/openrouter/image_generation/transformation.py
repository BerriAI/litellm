"""
OpenRouter image generation through POST {api_base}/images

Response shape:
{
    "created": 1790994420,
    "data": [{"b64_json": "...", "media_type": "image/png"}],
    "usage": {
        "prompt_tokens": 18,
        "completion_tokens": 272,
        "total_tokens": 290,
        "cost": 0.002212,
        "cost_details": {"upstream_inference_cost": 0.002212, ...},
        "completion_tokens_details": {"image_tokens": 272}
    }
}
"""

import re
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

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
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer
else:
    LiteLLMLoggingObj = Any

OPENROUTER_API_BASE: Final = "https://openrouter.ai/api/v1"
IMAGES_PATH: Final = "/images"
LEGACY_CHAT_COMPLETIONS_SUFFIX: Final = "/chat/completions"
QUALITY_ALIASES: Final = MappingProxyType({"standard": "low", "hd": "high"})
RESOLUTION_TIER_MODEL_AUTHOR: Final = "google/"
QUALITY_RESOLUTION_TIERS: Final = MappingProxyType({"auto": "1K", "low": "1K", "medium": "2K", "high": "4K"})
OPENAI_SIZE_ASPECT_RATIOS: Final = MappingProxyType(
    {
        "256x256": "1:1",
        "512x512": "1:1",
        "1024x1024": "1:1",
        "1536x1024": "3:2",
        "1024x1536": "2:3",
        "1792x1024": "16:9",
        "1024x1792": "9:16",
    }
)
LEGACY_IMAGE_CONFIG_FIELDS: Final = MappingProxyType({"aspect_ratio": "aspect_ratio", "image_size": "resolution"})
NON_BODY_PARAMS: Final = frozenset(
    {"model", "prompt", "messages", "modalities", "stream", "image_config", "extra_headers"}
)
PIXEL_SIZE: Final = re.compile(r"\d+x\d+")
SIZE_OVERRIDING_FIELDS: Final = frozenset({"aspect_ratio", "resolution"})


class OpenRouterImageGenerationConfig(BaseImageGenerationConfig):
    """
    OpenRouter image generation through the dedicated /images endpoint, which serves both
    image-only models (openai/gpt-image-*) and image+text models (google/gemini-*-image)
    """

    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        return [
            "size",
            "quality",
            "n",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        size and n pass through as is: /images takes explicit pixel sizes and normalizes them per
        provider. quality is native on /images, so only the dall-e-3 names are translated, except on
        Google's models, see _map_quality_to_resolution_tier
        """
        supported_params: Final = self.get_supported_openai_params(model)
        mapped_params: Final[dict[str, object]] = {
            key: QUALITY_ALIASES.get(value, value) if key == "quality" else value
            for key, value in non_default_params.items()
            if (key in supported_params or not drop_params) and (key, value) != ("size", "auto")
        }
        if (
            "quality" in mapped_params
            and "image_config" not in optional_params
            and model.removeprefix("openrouter/").startswith(RESOLUTION_TIER_MODEL_AUTHOR)
        ):
            return {**optional_params, **self._map_quality_to_resolution_tier(mapped_params)}
        return {**optional_params, **mapped_params}

    @staticmethod
    def _map_quality_to_resolution_tier(mapped_params: dict[str, object]) -> dict[str, object]:
        """
        Google's image models take a resolution tier on /images and ignore quality, so quality keeps the
        meaning it had on the chat-based path: image_config.image_size (1K, 2K or 4K), next to the aspect
        ratio of an OpenAI pixel size. A tier size or an image_config set by the caller wins
        """
        size: Final = str(mapped_params.get("size") or "")
        tier: Final = QUALITY_RESOLUTION_TIERS.get(str(mapped_params["quality"]))
        params: Final = {key: value for key, value in mapped_params.items() if key != "quality"}
        if tier is None or (size and PIXEL_SIZE.fullmatch(size) is None):
            return params
        aspect_ratio: Final = OPENAI_SIZE_ASPECT_RATIOS.get(size)
        image_config: Final = (
            {"image_size": tier} if aspect_ratio is None else {"aspect_ratio": aspect_ratio, "image_size": tier}
        )
        return {**params, "image_config": image_config}

    def _set_usage_and_cost(
        self,
        model_response: ImageResponse,
        response_json: dict,
        model: str,
    ) -> None:
        """
        Extract and set usage and cost information from OpenRouter response.

        Args:
            model_response: ImageResponse object to populate
            response_json: Parsed JSON response from OpenRouter
            model: The model name
        """
        usage_data: Final = response_json.get("usage", {})
        if usage_data:
            # The /images usage schema allows null for completion_tokens_details and image_tokens, and
            # per-image priced models report only completion_tokens
            prompt_tokens: Final = usage_data.get("prompt_tokens") or 0
            total_tokens: Final = usage_data.get("total_tokens") or 0

            completion_tokens_details: Final = usage_data.get("completion_tokens_details") or {}
            image_tokens: Final = completion_tokens_details.get("image_tokens")

            model_response.usage = ImageUsage(
                input_tokens=prompt_tokens,
                input_tokens_details=ImageUsageInputTokensDetails(
                    image_tokens=0,  # Input doesn't contain images for generation
                    text_tokens=prompt_tokens,
                ),
                output_tokens=image_tokens if image_tokens is not None else usage_data.get("completion_tokens") or 0,
                total_tokens=total_tokens,
            )

            cost: Final = usage_data.get("cost")
            if cost is not None:
                if not hasattr(model_response, "_hidden_params"):
                    model_response._hidden_params = {}
                if "additional_headers" not in model_response._hidden_params:
                    model_response._hidden_params["additional_headers"] = {}
                model_response._hidden_params["additional_headers"]["llm_provider-x-litellm-response-cost"] = float(
                    cost
                )

            cost_details: Final = usage_data.get("cost_details", {})
            if cost_details:
                if "response_cost_details" not in model_response._hidden_params:
                    model_response._hidden_params["response_cost_details"] = {}
                model_response._hidden_params["response_cost_details"].update(cost_details)

        model_response._hidden_params["model"] = response_json.get("model", model)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        base_url: Final = (api_base or OPENROUTER_API_BASE).rstrip("/")
        if base_url.endswith(IMAGES_PATH):
            return base_url
        return base_url.removesuffix(LEGACY_CHAT_COMPLETIONS_SUFFIX) + IMAGES_PATH

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
        """
        image_config is the request shape of the older chat-based path. Its fields map onto the
        /images names so existing configs keep working, and explicit top-level values win

        A configured aspect_ratio or resolution wins over an OpenAI pixel size, the way image_config
        won over size on the chat path, because /images answers that pair with a 400
        """
        legacy_image_config: Final = optional_params.get("image_config") or {}
        body: Final[dict[str, object]] = {
            "model": model,
            "prompt": prompt,
            **{
                LEGACY_IMAGE_CONFIG_FIELDS[key]: value
                for key, value in legacy_image_config.items()
                if key in LEGACY_IMAGE_CONFIG_FIELDS
            },
            **{key: value for key, value in optional_params.items() if key not in NON_BODY_PARAMS},
        }
        drop_pixel_size: Final = not SIZE_OVERRIDING_FIELDS.isdisjoint(body) and (
            PIXEL_SIZE.fullmatch(str(body.get("size", ""))) is not None
        )
        return {key: value for key, value in body.items() if not (drop_pixel_size and key == "size")}

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        try:
            response_json: Final = raw_response.json()
        except ValueError as e:
            raise OpenRouterException(
                message=f"Error parsing OpenRouter response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            ) from e

        image_response: Final = ImageResponse(
            created=response_json.get("created"),
            data=[ImageObject(b64_json=item.get("b64_json")) for item in response_json.get("data") or []],
        )
        self._set_usage_and_cost(image_response, response_json, model)
        return image_response

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        """Get the appropriate error class for OpenRouter errors."""
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )

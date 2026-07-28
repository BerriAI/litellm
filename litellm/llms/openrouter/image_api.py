"""
Shared plumbing for OpenRouter's unified image API, ``POST {api_base}/images``.

Both ``/v1/images/generations`` and ``/v1/images/edits`` map onto that one endpoint:
generation sends ``{model, prompt, ...}``, edit adds ``input_references`` carrying the
source image(s). Every image model is catalogued there, including the hybrid text+image
models that ``/chat/completions`` also serves, and models whose only output modality is
``image`` are reachable *only* there.

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
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import ValidationError

from litellm.exceptions import UnsupportedParamsError
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openrouter import OpenRouterImagesResponse
from litellm.types.utils import (
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)

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


def resolve_images_url(api_base: str | None) -> str:
    base_url = (api_base or get_secret_str("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE).rstrip("/")
    if base_url.endswith("/images"):
        return base_url
    return f"{base_url}/images"


def map_size_to_aspect_ratio(size: str) -> str | None:
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


def map_image_params(params: Mapping[str, object], model: str, drop_params: bool) -> dict[str, object]:
    """
    Translate OpenAI image params into OpenRouter's image API params.

    ``size`` becomes ``aspect_ratio``, which is what OpenRouter's image models take;
    none of them accept pixel dimensions. ``quality`` passes through untouched because
    OpenRouter reuses OpenAI's ``auto``/``low``/``medium``/``high`` enum for it.
    ``response_format`` is dropped because the endpoint always answers with base64.
    """
    response_format = params.get("response_format")
    if response_format is not None:
        _reject_unsupported_response_format(value=str(response_format), model=model, drop_params=drop_params)

    translated = (_translate_param(key=key, value=value) for key, value in params.items())
    return {key: value for key, value in translated if key is not None}


def _translate_param(key: str, value: object) -> tuple[str | None, object]:
    if key == "response_format":
        return None, None
    if key == "size":
        aspect_ratio = map_size_to_aspect_ratio(str(value))
        return ("aspect_ratio", aspect_ratio) if aspect_ratio is not None else (None, None)
    return key, value


def _reject_unsupported_response_format(value: str, model: str, drop_params: bool) -> None:
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


def parse_images_response(raw_response: httpx.Response) -> OpenRouterImagesResponse:
    try:
        return OpenRouterImagesResponse.model_validate(raw_response.json())
    except (ValueError, ValidationError) as e:
        raise OpenRouterException(
            message=f"Error parsing OpenRouter image response: {str(e)}",
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )


def apply_images_response(
    parsed: OpenRouterImagesResponse,
    model_response: ImageResponse,
    model: str,
) -> ImageResponse:
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

    _set_usage_and_cost(model_response=model_response, parsed=parsed, model=model)

    return model_response


def _set_usage_and_cost(
    model_response: ImageResponse,
    parsed: OpenRouterImagesResponse,
    model: str,
) -> None:
    model_response._hidden_params["model"] = parsed.model or model

    usage = parsed.usage
    if usage is None:
        return

    image_tokens = usage.completion_tokens_details.image_tokens if usage.completion_tokens_details else None
    input_image_tokens = (usage.prompt_tokens_details.image_tokens or 0) if usage.prompt_tokens_details else 0
    model_response.usage = ImageUsage(
        input_tokens=usage.prompt_tokens,
        input_tokens_details=ImageUsageInputTokensDetails(
            image_tokens=input_image_tokens,
            text_tokens=usage.prompt_tokens - input_image_tokens,
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

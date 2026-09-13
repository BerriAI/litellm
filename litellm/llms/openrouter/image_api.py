"""
Shared plumbing for OpenRouter's unified image API, ``POST {api_base}/images``.

Both image_generation and image_edit configs (see the sibling ``image_generation/`` and
``image_edit/`` packages) map their OpenAI-shaped requests onto this one endpoint:
generation sends ``{model, prompt, ...}``, edit adds ``input_references`` carrying the
source image(s). This module is the single place that owns the URL, the OpenAI-param ->
OpenRouter-param translation, and the response parsing, so the two transformation configs
stay thin adapters instead of duplicating that logic (as the pre-images-API version did,
via chat/completions, before this port of upstream BerriAI/litellm#34908).

Every OpenRouter image model is catalogued behind this endpoint, including models whose
only output modality is ``image`` (e.g. future `openai/gpt-image-2.5-flare` aliases) that
``/chat/completions`` cannot serve at all -- that gap is the reason this endpoint exists
as a distinct code path rather than a header-only URL swap on the chat transport.

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

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

# OpenRouter's image models only accept these aspect ratios; pixel sizes are mapped onto
# the closest one below rather than rejected, since most callers pass OpenAI-style sizes.
SUPPORTED_ASPECT_RATIOS: tuple[tuple[int, int], ...] = (
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

PIXEL_SIZE_PATTERN = re.compile(r"(\d+)x(\d+)")

# Params consumed by the transformation layer itself (model/prompt are pulled out
# explicitly when building the request body; extra_headers never belongs in the body).
NON_BODY_PARAMS = frozenset({"model", "prompt", "extra_headers"})

# Per OpenRouter's /api/v1/images schema (openrouter.ai/docs/features/multimodal/image-generation),
# these are the only scalar params typed as integer; everything else that survives translation
# (size/quality/aspect_ratio/background/output_format/response_format) is a string and passes
# through untouched. Callers may invoke image generation with a JSON body (values already
# typed) and image edit with multipart/form-data (every field arrives as `str`) against the
# same route, so without this coercion the multipart path can send OpenRouter e.g. `n: "1"`
# and OpenRouter's Zod validator 400s with "expected number, received string" -- generation
# looked fine only because JSON already carries real ints.
INT_PARAMS = frozenset({"n", "output_compression", "seed"})


def resolve_images_url(api_base: str | None) -> str:
    """
    Build the OpenRouter images endpoint URL from a configured api_base.

    Called by both OpenRouterImageGenerationConfig.get_complete_url and
    OpenRouterImageEditConfig.get_complete_url so the two request types always agree on
    where the unified image API lives, and so an operator override of api_base (e.g. in
    LiteLLM's DB-stored route config) doesn't need updating in two places.
    """
    base_url = (api_base or get_secret_str("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE).rstrip("/")
    if base_url.endswith("/images"):
        return base_url
    return f"{base_url}/images"


def map_size_to_aspect_ratio(size: str) -> str | None:
    """
    Map an OpenAI ``WxH`` size onto the closest aspect ratio OpenRouter accepts.

    Returns None for "auto" and for anything that isn't a recognizable pixel size, so the
    model's own default aspect ratio applies instead of a guessed one -- guessing a ratio
    for a value we can't parse would silently override a caller's intended default.
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

    Shared by image_generation and image_edit's map_openai_params so "size" ->
    "aspect_ratio" and the response_format rejection stay identical on both call paths --
    OpenRouter's image models don't accept pixel dimensions, and the endpoint always
    returns base64 regardless of what the caller asked for.
    """
    response_format = params.get("response_format")
    if response_format is not None:
        _reject_unsupported_response_format(value=str(response_format), model=model, drop_params=drop_params)

    mapped: dict[str, object] = {}
    for key, value in params.items():
        new_key, new_value = _translate_param(key=key, value=value, model=model)
        if new_key is not None:
            mapped[new_key] = new_value
    return mapped


def _translate_param(key: str, value: object, model: str) -> tuple[str | None, object]:
    if key == "response_format":
        return None, None
    if key == "size":
        aspect_ratio = map_size_to_aspect_ratio(str(value))
        return ("aspect_ratio", aspect_ratio) if aspect_ratio is not None else (None, None)
    if key in INT_PARAMS and value is not None:
        return key, _coerce_int_param(key=key, value=value, model=model)
    return key, value


def _coerce_int_param(key: str, value: object, model: str) -> int:
    """
    Normalize a scalar OpenRouter integer param regardless of caller transport.

    The JSON-bodied /v1/images/generations path already hands us a real int, but the
    multipart-form /v1/images/edits path (FastAPI form fields have no native type) hands
    us a `str` for every field including `n`. Coercing here -- once, for both call paths
    -- means a caller-transport quirk can never surface as a raw provider 400; a value
    that truly cannot be an int fails loudly and specifically instead of reaching
    OpenRouter and coming back as an opaque Zod validation error.
    """
    if isinstance(value, bool):
        # bool is an int subclass in Python; reject it explicitly rather than silently
        # sending 0/1 for a field where that would be a nonsensical image count/seed.
        raise UnsupportedParamsError(
            model=model,
            llm_provider="openrouter",
            message=f"OpenRouter's image API expects `{key}` to be an integer, got a boolean.",
        )
    if isinstance(value, int):
        return value
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        raise UnsupportedParamsError(
            model=model,
            llm_provider="openrouter",
            message=(
                f"OpenRouter's image API expects `{key}` to be an integer, got "
                f"{value!r}. Pass a numeric value (e.g. `{key}=1`)."
            ),
        )


def _reject_unsupported_response_format(value: str, model: str, drop_params: bool) -> None:
    # OpenRouter's image API has no url-mode; only b64_json is ever returned. Fail loudly
    # unless the caller opted into silently dropping unsupported params.
    if value == "b64_json" or drop_params:
        return
    raise UnsupportedParamsError(
        model=model,
        llm_provider="openrouter",
        message=(
            "OpenRouter's image API always returns base64 image data, so "
            f"response_format='{value}' is not supported. Request 'b64_json', or set "
            "`drop_params: true` to ignore it."
        ),
    )


def parse_images_response(raw_response: httpx.Response) -> OpenRouterImagesResponse:
    """Validate the raw HTTP body against the typed OpenRouter images schema."""
    try:
        return OpenRouterImagesResponse.model_validate(raw_response.json())
    except (ValueError, ValidationError) as e:
        raise OpenRouterException(
            message=f"Error parsing OpenRouter image response: {e!s}",
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )


def apply_images_response(
    parsed: OpenRouterImagesResponse,
    model_response: ImageResponse,
    model: str,
) -> ImageResponse:
    """
    Populate a litellm ImageResponse from a parsed OpenRouter images response.

    Called by both image_generation and image_edit's transform_*_response so usage/cost
    bookkeeping (spend-log accuracy) stays in one place instead of being copy-pasted per
    request type.
    """
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
    if not hasattr(model_response, "_hidden_params") or model_response._hidden_params is None:
        model_response._hidden_params = {}
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

    # Feed the upstream-reported cost straight into the spend-log path via the same
    # hidden-header convention the rest of the OpenRouter configs use, so settlement
    # prefers OpenRouter's real cost over litellm's static price table when both are
    # present.
    if usage.cost is not None:
        additional_headers = dict(model_response._hidden_params.get("additional_headers") or {})
        additional_headers["llm_provider-x-litellm-response-cost"] = usage.cost
        model_response._hidden_params["additional_headers"] = additional_headers

    if usage.cost_details:
        model_response._hidden_params["response_cost_details"] = {
            **(model_response._hidden_params.get("response_cost_details") or {}),
            **usage.cost_details,
        }

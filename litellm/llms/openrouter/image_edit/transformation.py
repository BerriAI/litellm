"""
OpenRouter image edit, served by ``POST {api_base}/images``.

The source image travels in ``input_references`` as a base64 data URL. Models whose only output
modality is ``image`` (``openai/gpt-image-*``, ``krea/*``, ...) are reachable only there.
"""

import base64
import re
from collections.abc import Mapping
from io import BufferedReader, BytesIO
from typing import TYPE_CHECKING, Any, Final, cast

import httpx
from httpx._types import RequestFiles
from pydantic import ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.openrouter import OpenRouterImagesResponse, OpenRouterImageUsage
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import (
    FileTypes,
    ImageObject,
    ImageResponse,
    ImageUsage,
    ImageUsageInputTokensDetails,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_DEFAULT_OPENROUTER_API_BASE: Final = "https://openrouter.ai/api/v1"

_SUPPORTED_ASPECT_RATIOS: Final[tuple[tuple[int, int], ...]] = (
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

_PIXEL_SIZE_PATTERN: Final = re.compile(r"(\d+)x(\d+)")

_NON_BODY_PARAMS: Final = frozenset({"model", "prompt", "extra_headers", "input_references"})


class OpenRouterImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return ["background", "n", "quality", "response_format", "size"]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        response_format: Final = image_edit_optional_params.get("response_format")
        if response_format is not None:
            _reject_unsupported_response_format(value=str(response_format), model=model, drop_params=drop_params)

        translated: Final = (
            _translate_param(key=key, value=value) for key, value in image_edit_optional_params.items()
        )
        return {key: value for key, value in translated if key is not None}

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_api_key: Final = api_key or litellm.api_key or get_secret_str("OPENROUTER_API_KEY")
        if not resolved_api_key:
            raise ValueError("OPENROUTER_API_KEY is not set")
        return {**headers, "Authorization": f"Bearer {resolved_api_key}"}

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        configured: Final = api_base or get_secret_str("OPENROUTER_API_BASE") or _DEFAULT_OPENROUTER_API_BASE
        base_url: Final = configured.rstrip("/").removesuffix("/chat/completions")
        if base_url.endswith("/images"):
            return base_url
        return f"{base_url}/images"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        candidates: Final = image if isinstance(image, list) else [image]
        images: Final = tuple(img for img in candidates if img is not None)
        if not images:
            raise ValueError("An image is required to edit; OpenRouter has no image-less edit mode.")

        passthrough: Final = {
            key: value for key, value in image_edit_optional_request_params.items() if key not in _NON_BODY_PARAMS
        }
        request_body: Final = {
            "model": model,
            **({"prompt": prompt} if prompt is not None else {}),
            **passthrough,
            "input_references": [_input_reference(img) for img in images],
        }
        return request_body, cast(RequestFiles, [])

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        parsed: Final = _parse_images_response(raw_response)
        return ImageResponse(
            created=parsed.created,
            data=[
                ImageObject(b64_json=image.b64_json, url=image.url, revised_prompt=image.revised_prompt)
                for image in parsed.data
            ],
            usage=_image_usage(parsed.usage) if parsed.usage is not None else None,
            hidden_params=_hidden_params(parsed=parsed, model=model),
        )

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return OpenRouterException(
            message=error_message,
            status_code=status_code,
            headers=headers,
        )


def _map_size_to_aspect_ratio(size: str) -> str | None:
    match: Final = _PIXEL_SIZE_PATTERN.fullmatch(size.strip())
    if match is None:
        return None

    width: Final = int(match.group(1))
    height: Final = int(match.group(2))
    if width == 0 or height == 0:
        return None

    target: Final = width / height
    closest: Final = min(_SUPPORTED_ASPECT_RATIOS, key=lambda ratio: abs(ratio[0] / ratio[1] - target))
    return f"{closest[0]}:{closest[1]}"


def _translate_param(key: str, value: object) -> tuple[str | None, object]:
    if key == "response_format":
        return None, None
    if key != "size":
        return key, value
    aspect_ratio: Final = _map_size_to_aspect_ratio(str(value))
    return ("aspect_ratio", aspect_ratio) if aspect_ratio is not None else (None, None)


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


def _input_reference(image: FileTypes) -> dict[str, object]:
    mime_type: Final = ImageEditRequestUtils.get_image_content_type(image)
    b64_data: Final = base64.b64encode(_read_image_bytes(image)).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{b64_data}"},
    }


def _read_image_bytes(image: FileTypes) -> bytes:
    if isinstance(image, bytes):
        return image
    if not isinstance(image, (BytesIO, BufferedReader)):
        raise ValueError("Unsupported image type for OpenRouter image edit.")
    current_pos: Final = image.tell()
    image.seek(0)
    data: Final = image.read()
    image.seek(current_pos)
    return data


def _parse_images_response(raw_response: httpx.Response) -> OpenRouterImagesResponse:
    try:
        return OpenRouterImagesResponse.model_validate(raw_response.json())
    except (ValueError, ValidationError) as e:
        raise OpenRouterException(
            message=f"Error parsing OpenRouter image response: {e}",
            status_code=raw_response.status_code if raw_response.status_code >= 400 else 502,
            headers=raw_response.headers,
        ) from e


def _image_usage(usage: OpenRouterImageUsage) -> ImageUsage:
    details: Final = usage.completion_tokens_details
    output_image_tokens: Final = details.image_tokens if details is not None else None
    prompt_details: Final = usage.prompt_tokens_details
    input_image_tokens: Final = (prompt_details.image_tokens or 0) if prompt_details is not None else 0
    return ImageUsage(
        input_tokens=usage.prompt_tokens,
        input_tokens_details=ImageUsageInputTokensDetails(
            image_tokens=input_image_tokens,
            text_tokens=usage.prompt_tokens - input_image_tokens,
        ),
        output_tokens=output_image_tokens if output_image_tokens is not None else usage.completion_tokens,
        total_tokens=usage.total_tokens,
    )


def _hidden_params(parsed: OpenRouterImagesResponse, model: str) -> dict[str, object]:
    usage: Final = parsed.usage
    cost: Final = usage.cost if usage is not None else None
    cost_details: Final[Mapping[str, float | None] | None] = usage.cost_details if usage is not None else None
    return {
        "model": parsed.model or model,
        **({"additional_headers": {"llm_provider-x-litellm-response-cost": cost}} if cost is not None else {}),
        **({"response_cost_details": dict(cost_details)} if cost_details else {}),
    }

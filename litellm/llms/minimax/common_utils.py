import re
from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import ContentPolicyViolationError, UnsupportedParamsError
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.utils import ImageObject, ImageResponse

_STREAM_FLAG: Final = TypeAdapter(bool)
_SAFETY_CHECK_CODE: Final = re.compile(r"\((?:1026|1027)\)")
_API_SURFACE_SUFFIX: Final = re.compile(r"/(?:anthropic(?:/v1(?:/messages)?)?|v[12])$")


class MinimaxImageError(BaseLLMException):
    pass


class _MinimaxImage(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None
    size: str | None = None
    error: Mapping[str, object] | None = None


class _MinimaxImages(BaseModel):
    model_config = ConfigDict(frozen=True)

    data: tuple[_MinimaxImage, ...] | None = None


def _asks_to_stream(value: object) -> bool:
    if value is None:
        return False
    try:
        return _STREAM_FLAG.validate_python(value)
    except ValidationError:
        return True


def _drops_params(drop_params: object) -> bool:
    return litellm.drop_params is True or drop_params is True


def minimax_images_url(api_base: str, endpoint: str) -> str:
    base_url: Final = api_base.rstrip("/")
    if base_url.endswith(f"/{endpoint}"):
        return base_url
    return join_api_base(_API_SURFACE_SUFFIX.sub("", base_url), f"v1/{endpoint}")


def image_params_without_stream(params: Mapping[str, object], drop_params: object) -> Mapping[str, object]:
    if _asks_to_stream(params.get("stream")) and not _drops_params(drop_params):
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                "LiteLLM does not support streaming MiniMax image responses. "
                "To drop `stream` from the call, set `drop_params: true`."
            ),
        )
    return {key: value for key, value in params.items() if key != "stream"}


def edit_params_without_json_images(params: Mapping[str, object], drop_params: object) -> Mapping[str, object]:
    if params.get("images") is not None and not _drops_params(drop_params):
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                "LiteLLM does not support JSON `images` on MiniMax image edits. Upload each image as an `image` "
                "file instead. To drop `images` from the call, set `drop_params: true`."
            ),
        )
    return {key: value for key, value in params.items() if key != "images"}


def _image_object(image: _MinimaxImage) -> ImageObject:
    return ImageObject(
        url=image.url,
        b64_json=image.b64_json,
        revised_prompt=image.revised_prompt,
        provider_specific_fields=None if image.error is None else {"error": dict(image.error)},
    )


def with_minimax_images(image_response: ImageResponse, raw_response: httpx.Response) -> ImageResponse:
    images: Final = _MinimaxImages.model_validate_json(raw_response.content).data or ()
    return image_response.model_copy(
        update={
            "data": [_image_object(image) for image in images],
            "size": next((image.size for image in images if image.size is not None), None),
            "quality": None,
        }
    )


def minimax_safety_error(model: object, message: str, original_exception: object) -> ContentPolicyViolationError | None:
    if not isinstance(original_exception, MinimaxImageError) or _SAFETY_CHECK_CODE.search(message) is None:
        return None
    return ContentPolicyViolationError(message=message, model=str(model), llm_provider="minimax")

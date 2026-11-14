from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.types.utils import ImageObject, ImageResponse

_STREAM_FLAG: Final = TypeAdapter(bool)


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


def image_params_without_stream(params: Mapping[str, object], drop_params: object) -> Mapping[str, object]:
    if _asks_to_stream(params.get("stream")) and litellm.drop_params is not True and drop_params is not True:
        raise UnsupportedParamsError(
            status_code=400,
            message=(
                "LiteLLM does not support streaming MiniMax image responses. "
                "To drop `stream` from the call, set `drop_params: true`."
            ),
        )
    return {key: value for key, value in params.items() if key != "stream"}


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

import base64
import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import httpx
from httpx._types import RequestFiles

from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.fal_ai.image_generation.gpt_image_2_transformation import (
    map_gpt_image_quality,
    map_gpt_image_size,
)
from litellm.llms.fal_ai.image_generation.transformation import fal_images_to_image_objects
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_BASE_URL: Final[str] = "https://fal.run"
EDIT_SUFFIX: Final[str] = "/edit"
SUPPORTED_OPENAI_PARAMS: Final[tuple[str, ...]] = ("background", "mask", "n", "quality", "size")
PARAM_TRANSLATION: Final[Mapping[str, str]] = MappingProxyType(
    {
        "background": "background",
        "n": "num_images",
        "quality": "quality",
        "size": "image_size",
    }
)


@runtime_checkable
class _SeekableBinaryReader(Protocol):
    def tell(self) -> int: ...

    def seek(self, offset: int) -> int: ...

    def read(self) -> bytes: ...


def _read_image_bytes(image: object) -> bytes:
    if isinstance(image, bytes):
        return image
    if isinstance(image, tuple):
        return _read_image_bytes(image[1])
    if isinstance(image, os.PathLike):
        return Path(image).read_bytes()
    if isinstance(image, _SeekableBinaryReader):
        position: Final = image.tell()
        image.seek(0)
        data: Final = image.read()
        image.seek(position)
        return data
    raise ValueError(f"Unsupported image type for Fal AI image edit: {type(image).__name__}")


def to_data_url(image: object) -> str:
    if isinstance(image, str):
        return image
    image_bytes: Final = _read_image_bytes(image)
    mime_type: Final = ImageEditRequestUtils.get_image_content_type(image_bytes)
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"


def _first(value: object) -> object:
    return value[0] if isinstance(value, list) and value else value


class FalAIImageEditConfig(BaseImageEditConfig):
    """
    Image edits served through Fal AI's ``/edit`` endpoints, e.g. openai/gpt-image-2.5/flare/edit.

    Fal expects a JSON body with ``image_urls`` (and an optional ``mask_url``) rather than multipart
    uploads, so local files are sent inline as base64 data URLs.
    """

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: base class contract returns a list
        return list(SUPPORTED_OPENAI_PARAMS)  # mutable-ok: base class contract returns a list

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        return {  # mutable-ok: base class contract returns a dict
            PARAM_TRANSLATION.get(key, key): self._translate_value(key, value, model)
            for key, value in image_edit_optional_params.items()
            if value is not None
        }

    def _translate_value(self, key: str, value: object, model: str) -> object:
        if key == "size":
            return map_gpt_image_size(value)
        if key == "quality":
            return map_gpt_image_quality(value, model)
        return value

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        final_api_key: Final = api_key or get_secret_str("FAL_AI_API_KEY")
        if not final_api_key:
            raise ValueError("FAL_AI_API_KEY is not set")
        return {**headers, "Authorization": f"Key {final_api_key}"}  # mutable-ok: base class contract returns a dict

    def use_multipart_form_data(self) -> bool:
        return False

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        base_url: Final = (api_base or get_secret_str("FAL_AI_API_BASE") or DEFAULT_BASE_URL).rstrip("/")
        endpoint: Final = model if model.endswith(EDIT_SUFFIX) else f"{model}{EDIT_SUFFIX}"
        return f"{base_url}/{endpoint}"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        images: Final = tuple(img for img in (image if isinstance(image, list) else (image,)) if img is not None)
        if not images:
            raise ValueError("Fal AI image edit requires at least one input image")
        mask: Final = _first(image_edit_optional_request_params.get("mask"))
        mask_field: Final[Mapping[str, str]] = (
            MappingProxyType({"mask_url": to_data_url(mask)}) if mask is not None else MappingProxyType({})
        )
        provider_params: Final[Mapping[str, object]] = MappingProxyType(
            {key: value for key, value in image_edit_optional_request_params.items() if key != "mask"}
        )
        request_body: Final[dict[str, object]] = {  # mutable-ok: base class contract returns a dict
            "prompt": prompt,
            "image_urls": tuple(to_data_url(img) for img in images),
            **mask_field,
            **provider_params,
        }
        return request_body, ()

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        try:
            response_json: Final = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error parsing Fal AI image edit response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        model_response: Final = ImageResponse()
        model_response.data = list(  # mutable-ok: ImageResponse.data is typed as a list
            fal_images_to_image_objects(response_json.get("images", ()))
        )
        return model_response

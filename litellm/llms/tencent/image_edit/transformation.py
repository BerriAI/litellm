from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles
from pydantic import TypeAdapter
from typing_extensions import override

from litellm.images.utils import image_to_data_url
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.tencent.image_generation.transformation import (
    TOKENHUB_IMAGE_PARAMS,
    tokenhub_image_headers,
    tokenhub_image_params,
    tokenhub_image_response,
    tokenhub_image_url,
)
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_IMAGES: Final = TypeAdapter(tuple[object, ...])
_FORM_BOOLEANS: Final = MappingProxyType({"true": True, "false": False})


def _form_field_value(key: str, value: object) -> object:
    match key, value:
        case "watermark", str() if value.lower() in _FORM_BOOLEANS:
            return _FORM_BOOLEANS[value.lower()]
        case "seed", str() if value.removeprefix("-").isdigit():
            return int(value)
        case _:
            return value


class TencentImageEditConfig(BaseImageEditConfig):
    @override
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: mirrors override contract
        return list(TOKENHUB_IMAGE_PARAMS)

    @override
    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        return dict(tokenhub_image_params(image_edit_optional_params, model, drop_params))

    @override
    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: mirrors override contract
        return dict(tokenhub_image_headers(headers, api_key, model))

    @override
    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        return tokenhub_image_url(api_base)

    @override
    def use_multipart_form_data(self) -> bool:
        return False

    @override
    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: object,
        image_edit_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles]:  # mutable-ok: mirrors override contract
        images: Final = _IMAGES.validate_python(image) if isinstance(image, list) else (image,)
        return {
            "model": model,
            "prompt": prompt,
            "images": tuple(image_to_data_url(img) for img in images if img is not None),
            **{key: _form_field_value(key, value) for key, value in image_edit_optional_request_params.items()},
        }, ()

    @override
    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        image_response: Final = tokenhub_image_response(raw_response)
        if image_response is None:
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
            )
        return image_response

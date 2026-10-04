from __future__ import annotations

import base64
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_ENDPOINT, get_native_api_url, validate_headers
from litellm.llms.alibaba_token_plan.image_generation.transformation import AlibabaTokenPlanImageGenerationConfig
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


@runtime_checkable
class _BinaryReader(Protocol):
    def tell(self) -> int: ...

    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def read(self, n: int = -1, /) -> bytes: ...


def _image_reference(image: FileTypes | str) -> str:
    if isinstance(image, str):
        if image.startswith(("https://", "http://", "data:image/")):
            return image
        raise ValueError("Image strings must be HTTP URLs or image data URLs")
    if isinstance(image, tuple):
        return _image_reference(image[1])
    if isinstance(image, PathLike):
        return _image_reference(Path(image).read_bytes())
    if isinstance(image, _BinaryReader):
        position: Final = image.tell()
        try:
            image.seek(0)
            return _image_reference(image.read())
        finally:
            image.seek(position)
    content_type: Final = ImageEditRequestUtils.get_image_content_type(image)
    return f"data:{content_type};base64,{base64.b64encode(image).decode('ascii')}"


class AlibabaTokenPlanImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: provider interface requires a mutable return value
        common: Final = ("n", "size", "response_format", "seed", "watermark")
        native: Final = (
            ("enable_sequential", "bbox_list", "color_palette")
            if model.startswith("wan")
            else ("negative_prompt", "prompt_extend", "prompt_extend_mode", "enable_thinking")
        )
        return list(common + native)

    def map_openai_params(
        self,
        image_edit_optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: provider interface requires a mutable return value
        unsupported: Final = tuple(
            key for key in image_edit_optional_params if key not in self.get_supported_openai_params(model)
        )
        unsupported_format: Final = image_edit_optional_params.get("response_format", "url") != "url"
        if (unsupported or unsupported_format) and not (drop_params or litellm.drop_params):
            raise litellm.UnsupportedParamsError(
                message=(
                    f"Alibaba Token Plan image editing does not support: {', '.join(unsupported)}"
                    if unsupported
                    else "Alibaba Token Plan image editing supports response_format='url'"
                ),
                model=model,
                llm_provider="alibaba_token_plan",
            )
        unsupported_agent_mode: Final = image_edit_optional_params.get("prompt_extend_mode") == "agent"
        if unsupported_agent_mode and not (drop_params or litellm.drop_params):
            raise litellm.UnsupportedParamsError(
                message="Qwen image editing supports prompt_extend_mode='direct'",
                model=model,
                llm_provider="alibaba_token_plan",
            )
        supported_params: Final = frozenset(self.get_supported_openai_params(model)) - {
            "response_format",
            *(("prompt_extend_mode",) if unsupported_agent_mode else ()),
        }
        return {
            key: value.replace("x", "*") if key == "size" and isinstance(value, str) else value
            for key, value in image_edit_optional_params.items()
            if key in supported_params and value is not None
        }

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: provider interface requires a mutable return value
        return validate_headers(headers, api_key)

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return get_native_api_url(api_base, IMAGE_ENDPOINT)

    def use_multipart_form_data(self) -> bool:
        return False

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | list[FileTypes | str] | str | None,  # mutable-ok: tuples represent single upload files
        image_edit_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles]:  # mutable-ok: provider interface requires a mutable return value
        if not prompt:
            raise self.get_error_class("Alibaba Token Plan image editing requires a prompt", 400, {})
        images: Final = image if isinstance(image, list) else (() if image is None else (image,))
        maximum_images: Final = 9 if model.startswith("wan") else 3
        if not 1 <= len(images) <= maximum_images:
            raise self.get_error_class(f"{model} image editing requires 1 to {maximum_images} input images", 400, {})
        try:
            content: Final = tuple({"image": _image_reference(item)} for item in images) + ({"text": prompt},)
        except ValueError as error:
            raise self.get_error_class(str(error), 400, {})
        return {
            "model": model,
            "input": {"messages": ({"role": "user", "content": content},)},
            "parameters": dict(image_edit_optional_request_params),
        }, ()

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        return AlibabaTokenPlanImageGenerationConfig().transform_image_generation_response(
            model=model,
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=dict(headers))

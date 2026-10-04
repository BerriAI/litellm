from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.llms.alibaba_token_plan.common_utils import IMAGE_PATH, get_api_url, image_reference, validate_headers
from litellm.llms.alibaba_token_plan.image_generation.transformation import AlibabaTokenPlanImageGenerationConfig
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_SUPPORTED_PARAMS: Final = ("n", "size", "response_format", "seed", "watermark", "negative_prompt", "prompt_extend")


class AlibabaTokenPlanImageEditConfig(BaseImageEditConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: provider interface requires a mutable return value
        return list(_SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        image_edit_optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: provider interface requires a mutable return value
        if image_edit_optional_params.get("response_format", "url") != "url" and not (
            drop_params or litellm.drop_params
        ):
            raise litellm.UnsupportedParamsError(
                message="Alibaba Token Plan image editing only returns image URLs; use response_format='url'",
                model=model,
                llm_provider="alibaba_token_plan",
            )
        return {
            key: value.replace("x", "*") if key == "size" and isinstance(value, str) else value
            for key, value in image_edit_optional_params.items()
            if key in _SUPPORTED_PARAMS and key != "response_format"
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
        return get_api_url(api_base, IMAGE_PATH)

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
        images: Final = image if isinstance(image, list) else (() if image is None else (image,))
        content: Final = (*({"image": image_reference(item)} for item in images), {"text": prompt or ""})
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

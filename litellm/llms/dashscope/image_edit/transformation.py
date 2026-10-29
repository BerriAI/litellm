from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Protocol

import httpx
from httpx._types import RequestFiles
from pydantic import TypeAdapter
from typing_extensions import override

import litellm
from litellm.images.utils import image_to_data_url
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.dashscope.image_generation.transformation import (
    OPENAI_TO_DASHSCOPE_SIZE,
    DashScopeImageGenerationConfig,
)
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.litellm_core_utils.tokenizer import Encoding as Tokenizer

MAX_INPUT_IMAGES: Final = 3
DASHSCOPE_IMAGE_EDIT_PARAMS: Final = ("n", "size", "negative_prompt", "prompt_extend", "watermark", "seed")

_IMAGES: Final = TypeAdapter(tuple[object, ...])


def _dashscope_size(size: object) -> object:
    if not isinstance(size, str):
        return size
    return OPENAI_TO_DASHSCOPE_SIZE.get(size, size.replace("x", "*"))


class DashScopeImageGeneration(Protocol):
    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: mirrors the image generation contract
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: mirrors the image generation contract
        optional_params: dict[str, object],  # mutable-ok: mirrors the image generation contract
        litellm_params: dict[str, object],  # mutable-ok: mirrors the image generation contract
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]: ...  # mutable-ok: mirrors the image generation contract

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict[str, object],  # mutable-ok: mirrors the image generation contract
        litellm_params: dict[str, object],  # mutable-ok: mirrors the image generation contract
        stream: bool | None = None,
    ) -> str: ...

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: "LiteLLMLoggingObj",
        request_data: dict[str, object],  # mutable-ok: mirrors the image generation contract
        optional_params: dict[str, object],  # mutable-ok: mirrors the image generation contract
        litellm_params: dict[str, object],  # mutable-ok: mirrors the image generation contract
        encoding: "Tokenizer | None",
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse: ...


class DashScopeImageEditConfig(BaseImageEditConfig):
    def __init__(self, generation_config: DashScopeImageGeneration | None = None) -> None:
        super().__init__()
        self._generation_config: Final[DashScopeImageGeneration] = generation_config or DashScopeImageGenerationConfig()

    @override
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: mirrors override contract
        return list(DASHSCOPE_IMAGE_EDIT_PARAMS)

    @override
    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: mirrors override contract
        return {
            key: _dashscope_size(value) if key == "size" else value
            for key, value in image_edit_optional_params.items()
            if key in DASHSCOPE_IMAGE_EDIT_PARAMS
        }

    @override
    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: mirrors override contract
        model: str,
        api_key: str | None = None,
        litellm_params: dict[str, object] | None = None,  # mutable-ok: mirrors override contract
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: mirrors override contract
        return self._generation_config.validate_environment(
            headers=dict(headers),
            model=model,
            messages=[],
            optional_params={},
            litellm_params=dict(litellm_params or {}),
            api_key=api_key,
            api_base=api_base,
        )

    @override
    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: mirrors override contract
    ) -> str:
        return self._generation_config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model=model,
            optional_params={},
            litellm_params=litellm_params,
        )

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
        candidates: Final = _IMAGES.validate_python(image) if isinstance(image, list) else (image,)
        images: Final = tuple(candidate for candidate in candidates if candidate is not None)
        if len(images) > MAX_INPUT_IMAGES:
            raise litellm.BadRequestError(
                message=f"DashScope image edit accepts at most {MAX_INPUT_IMAGES} input images, got {len(images)}",
                model=model,
                llm_provider="dashscope",
            )
        prompt_content: Final = ({"text": prompt},) if prompt is not None else ()
        return {
            "model": model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [*({"image": image_to_data_url(item)} for item in images), *prompt_content],
                    }
                ]
            },
            "parameters": dict(image_edit_optional_request_params),
        }, ()

    @override
    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        return self._generation_config.transform_image_generation_response(
            model=model,
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=logging_obj,
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

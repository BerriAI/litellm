from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles
from pydantic import TypeAdapter
from typing_extensions import override

import litellm
from litellm.images.utils import image_to_data_url
from litellm.llms.base_llm.image_edit.transformation import BaseImageEditConfig
from litellm.llms.dashscope.common_utils import (
    DASHSCOPE_CHAT_COMPATIBLE_PATH,
    DASHSCOPE_MULTIMODAL_GENERATION_PATH,
    dashscope_api_root,
    is_dashscope_native_endpoint_url,
    resolve_dashscope_endpoints,
)
from litellm.llms.dashscope.image_generation.transformation import (
    DEFAULT_API_BASE,
    OPENAI_TO_DASHSCOPE_SIZE,
    DashScopeImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

MAX_INPUT_IMAGES: Final = 3
DASHSCOPE_IMAGE_EDIT_PARAMS: Final = ("n", "size", "negative_prompt", "prompt_extend", "watermark", "seed")

_IMAGES: Final = TypeAdapter(tuple[object, ...])


def _dashscope_size(size: object) -> object:
    if not isinstance(size, str):
        return size
    return OPENAI_TO_DASHSCOPE_SIZE.get(size, size.replace("x", "*"))


def _deployment_edit_url(api_base: str) -> str:
    derives_from_root: Final = not is_dashscope_native_endpoint_url(api_base) and (
        resolve_dashscope_endpoints(api_base) is not None
        or api_base.rstrip("/").endswith(DASHSCOPE_CHAT_COMPATIBLE_PATH)
    )
    return f"{dashscope_api_root(api_base)}{DASHSCOPE_MULTIMODAL_GENERATION_PATH}" if derives_from_root else api_base


class DashScopeImageEditConfig(BaseImageEditConfig):
    def __init__(self, generation_config: DashScopeImageGenerationConfig | None = None) -> None:
        super().__init__()
        self._generation_config: Final = generation_config or DashScopeImageGenerationConfig()

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
        if api_base is not None:
            return _deployment_edit_url(api_base)
        env_api_base: Final = get_secret_str("DASHSCOPE_API_BASE")
        derived: Final = (
            f"{dashscope_api_root(env_api_base)}{DASHSCOPE_MULTIMODAL_GENERATION_PATH}" if env_api_base else None
        )
        return get_secret_str("DASHSCOPE_API_BASE_IMAGE") or derived or DEFAULT_API_BASE

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

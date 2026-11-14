from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles

from litellm.litellm_core_utils.prompt_templates.file_id_utils import extract_file_data
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.minimax.chat.transformation import MinimaxChatConfig
from litellm.llms.minimax.common_utils import image_params_without_stream, with_minimax_images
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.types.llms.openai import FileTypes
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_REQUEST_FIELDS: Final = frozenset({"model", "prompt", "image", "images"})


def _image_contents(image: FileTypes | list[FileTypes] | None) -> list[bytes] | None:  # pyright: ignore[reportUnknownParameterType]  # mutable-ok: the parent uploads each image of a list, FileTypes holds a bare PathLike
    if image is None:
        return None
    return [file["content"] for file in map(extract_file_data, image if isinstance(image, list) else (image,))]  # pyright: ignore[reportUnknownArgumentType]  # FileTypes holds a bare PathLike


class MinimaxImageEditConfig(OpenAIImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: mirrors override contract
        return [
            "n",
            "response_format",
            "size",
            "aspect_ratio",
            "area",
            "output_format",
            "output_compression",
        ]

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: mirrors override contract
        resolved_key: Final = MinimaxChatConfig.get_api_key(api_key=api_key)
        if not resolved_key:
            raise ValueError("Missing MiniMax API Key - set MINIMAX_API_KEY or pass api_key")
        return {**headers, "Authorization": f"Bearer {resolved_key}"}

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        base_url: Final = MinimaxChatConfig.get_api_base(api_base=api_base).rstrip("/")
        if base_url.endswith("/images/edits"):
            return base_url
        if base_url.endswith("/v1"):
            return join_api_base(base_url, "images/edits")
        return join_api_base(base_url, "v1/images/edits")

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | list[FileTypes] | None,  # pyright: ignore[reportUnknownParameterType]  # mutable-ok: mirrors image_edit's image param, FileTypes holds a bare PathLike
        image_edit_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles]:  # mutable-ok: mirrors override contract
        body_params: Final = image_params_without_stream(image_edit_optional_request_params, litellm_params.drop_params)
        return super().transform_image_edit_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # parent declares bare dict and PathLike types
            model=model,
            prompt=prompt,
            image=_image_contents(image),  # pyright: ignore[reportArgumentType]  # the parent uploads each image of a list
            image_edit_optional_request_params={
                key: value for key, value in body_params.items() if key not in _REQUEST_FIELDS
            },
            litellm_params=litellm_params,
            headers={**headers},
        )

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> ImageResponse:
        return with_minimax_images(
            super().transform_image_edit_response(model=model, raw_response=raw_response, logging_obj=logging_obj),
            raw_response,
        )

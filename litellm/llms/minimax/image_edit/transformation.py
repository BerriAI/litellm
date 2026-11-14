from collections.abc import Mapping
from typing import Final

from httpx._types import RequestFiles
from pydantic import TypeAdapter

from litellm.images.utils import image_to_data_url
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.minimax.chat.transformation import MinimaxChatConfig
from litellm.llms.minimax.common_utils import image_params_without_stream
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.types.router import GenericLiteLLMParams

_IMAGES: Final = TypeAdapter(tuple[object, ...])
_NUMERIC_FORM_PARAMS: Final = frozenset({"output_compression"})


def _form_number(key: str, value: object) -> object:
    return int(value) if key in _NUMERIC_FORM_PARAMS and isinstance(value, str) and value.isdigit() else value


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

    def use_multipart_form_data(self) -> bool:
        return False

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
        body_params: Final = image_params_without_stream(image_edit_optional_request_params, litellm_params.drop_params)
        return {
            "model": model,
            "images": [{"image_url": image_to_data_url(item)} for item in images if item is not None],
            **{key: _form_number(key, value) for key, value in body_params.items()},
            **({} if prompt is None else {"prompt": prompt}),
        }, []

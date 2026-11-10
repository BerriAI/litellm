from typing import Any, Final

from httpx._types import RequestFiles

from litellm.images.utils import image_to_data_url
from litellm.litellm_core_utils.url_utils import join_api_base
from litellm.llms.minimax.chat.transformation import MinimaxChatConfig
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.types.router import GenericLiteLLMParams


class MinimaxImageEditConfig(OpenAIImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return [
            "n",
            "response_format",
            "size",
            "aspect_ratio",
            "area",
        ]

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        resolved_key: Final = MinimaxChatConfig.get_api_key(api_key=api_key)
        if not resolved_key:
            raise ValueError(
                "Missing MiniMax API Key - set MINIMAX_API_KEY or pass api_key"
            )
        headers["Authorization"] = f"Bearer {resolved_key}"
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
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
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles]:
        image_list = image if isinstance(image, list) else [image]
        request: dict[str, Any] = {
            "model": model,
            "images": [
                {"image_url": image_to_data_url(item)}
                for item in image_list
                if item is not None
            ],
            **image_edit_optional_request_params,
        }
        if prompt is not None:
            request["prompt"] = prompt
        return request, []

import base64
from typing import Any, Final

from litellm.exceptions import AuthenticationError
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.openai import FileTypes
from litellm.types.router import GenericLiteLLMParams

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)


class ChatGPTImageEditConfig(OpenAIImageEditConfig):
    """Codex Images edit API configuration backed by ChatGPT OAuth."""

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "image",
            "prompt",
            "background",
            "model",
            "n",
            "quality",
            "size",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        ]

    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported: Final = self.get_supported_openai_params(model)
        return {
            key: value
            for key, value in image_edit_optional_params.items()
            if key in supported
        }

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: dict | None = None,
        api_base: str | None = None,
    ) -> dict:
        try:
            access_token: Final = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider="chatgpt",
                message=str(e),
            )

        account_id: Final = self.authenticator.get_account_id()
        turn_id: Final = ensure_chatgpt_session_id(litellm_params)
        default_headers: Final = get_chatgpt_default_headers(
            access_token, account_id, turn_id
        )
        default_headers["accept"] = "application/json"
        default_headers["x-codex-image-turn-id"] = turn_id
        return {**default_headers, **headers}

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        resolved_api_base: Final = (
            api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        )
        return f"{resolved_api_base.rstrip('/')}/images/edits"

    def transform_image_edit_request(
        self,
        model: str,
        prompt: str | None,
        image: FileTypes | None,
        image_edit_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, dict]:
        image_list: Final = image if isinstance(image, list) else [image]
        images: Final = [
            {"image_url": self._to_data_url(item)}
            for item in image_list
            if item is not None
        ]
        optional_params: Final = {
            key: value
            for key, value in image_edit_optional_request_params.items()
            if key in {"background", "n", "quality", "size"}
        }
        return {
            "images": images,
            "prompt": prompt or "",
            "model": model,
            **optional_params,
        }, {}

    def use_multipart_form_data(self) -> bool:
        return False

    @staticmethod
    def _to_data_url(image: Any) -> str:
        if isinstance(image, str) and (
            image.startswith("data:")
            or image.startswith("https://")
            or image.startswith("http://")
        ):
            return image

        content_type: Final = ImageEditRequestUtils.get_image_content_type(image)
        if isinstance(image, (bytes, bytearray)):
            image_bytes = bytes(image)
        elif hasattr(image, "read"):
            position = image.tell() if hasattr(image, "tell") else None
            image_bytes = image.read()
            if position is not None and hasattr(image, "seek"):
                image.seek(position)
        else:
            raise ValueError("ChatGPT image edits require image bytes or a URL")

        encoded: Final = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{content_type};base64,{encoded}"

from typing import Final

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.image_generation.gpt_transformation import (
    GPTImageGenerationConfig,
)
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)


class ChatGPTImageGenerationConfig(GPTImageGenerationConfig):
    """Codex Images API configuration backed by ChatGPT subscription OAuth."""

    def __init__(self) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def get_supported_openai_params(self, model: str) -> list:
        return ["background", "n", "quality", "size"]

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
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
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        resolved_api_base: Final = (
            api_base or self.authenticator.get_api_base() or CHATGPT_API_BASE
        )
        return f"{resolved_api_base.rstrip('/')}/images/generations"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {"prompt": prompt, "model": model, **optional_params}

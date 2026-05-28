from typing import Any, List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    normalize_chatgpt_access_token,
)
from .streaming_utils import ChatGPTToolCallNormalizer


class ChatGPTConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = api_base or self.authenticator.get_api_base()
        if api_key:
            return (
                dynamic_api_base,
                normalize_chatgpt_access_token(api_key),
                custom_llm_provider,
            )

        try:
            dynamic_api_key = self.authenticator.get_access_token(
                allow_device_auth=False
            )
        except GetAccessTokenError:
            dynamic_api_key = None
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        access_token = api_key
        if not access_token:
            try:
                access_token = self.authenticator.get_access_token()
            except GetAccessTokenError as e:
                raise AuthenticationError(
                    model=model,
                    llm_provider="chatgpt",
                    message=str(e),
                )
        access_token = normalize_chatgpt_access_token(access_token)

        validated_headers = super().validate_environment(
            headers,
            model,
            messages,
            optional_params,
            litellm_params,
            access_token,
            api_base,
        )

        account_id = self.authenticator.get_account_id_from_token(access_token)
        if not account_id:
            account_id = self.authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            access_token or "", account_id, session_id
        )
        return {**default_headers, **validated_headers}

    def post_stream_processing(self, stream: Any) -> Any:
        return ChatGPTToolCallNormalizer(stream)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        optional_params.setdefault("stream", False)
        return optional_params

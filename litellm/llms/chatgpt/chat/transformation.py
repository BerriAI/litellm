from typing import Any, List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    CHATGPT_API_BASE,
    GetAccessTokenError,
    RefreshAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
    get_chatgpt_static_headers,
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

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        if not api_key:
            return CHATGPT_API_BASE, None, custom_llm_provider
        # Pass the refresh token through without exchanging. ChatGPT models
        # use mode="responses", so /chat/completions gets bridged to the
        # responses API (via responses_api_bridge) which uses the token
        # directly. Exchanging here would fail because get_llm_provider()
        # runs before the bridge check.
        dynamic_api_base = api_base or CHATGPT_API_BASE
        return dynamic_api_base, api_key, custom_llm_provider

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
        validated_headers = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, api_key, api_base
        )

        if api_key:
            # api_key is the refresh token. Exchange for access token (JWT)
            # via OpenAI OAuth. Cached at module level by the authenticator.
            try:
                authenticator = Authenticator(refresh_token=api_key)
                access_token = authenticator.get_access_token()
                account_id = authenticator.get_account_id()
                session_id = ensure_chatgpt_session_id(litellm_params)
                # get_chatgpt_default_headers already includes static headers.
                default_headers = get_chatgpt_default_headers(
                    access_token, account_id, session_id
                )
                validated_headers = {**default_headers, **validated_headers}
            except (GetAccessTokenError, RefreshAccessTokenError) as e:
                raise AuthenticationError(
                    message=f"ChatGPT token exchange failed: {e}",
                    llm_provider="chatgpt",
                    model=model,
                )
        else:
            validated_headers = {**get_chatgpt_static_headers(), **validated_headers}

        return validated_headers

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

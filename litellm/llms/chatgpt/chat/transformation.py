from typing import Any, List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
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
        self.authenticator = Authenticator(api_base=api_base)

    @staticmethod
    def _get_authenticator_for_request(
        api_base: Optional[str],
        litellm_params: Optional[dict],
    ) -> Authenticator:
        auth_file_path: Optional[str] = None
        if litellm_params is not None:
            auth_file_path = litellm_params.get("chatgpt_auth_file_path")
        return Authenticator(auth_file_path=auth_file_path, api_base=api_base)

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = api_base or self.authenticator.get_api_base()
        return dynamic_api_base, None, custom_llm_provider

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
        request_api_base = api_base
        if request_api_base is None and litellm_params is not None:
            request_api_base = litellm_params.get("api_base")
        authenticator = self._get_authenticator_for_request(
            request_api_base, litellm_params
        )
        resolved_api_key = api_key
        if not resolved_api_key:
            try:
                resolved_api_key = authenticator.get_access_token()
            except GetAccessTokenError as e:
                raise AuthenticationError(
                    model=model,
                    llm_provider="chatgpt",
                    message=str(e),
                )
        validated_headers = super().validate_environment(
            headers,
            model,
            messages,
            optional_params,
            litellm_params,
            resolved_api_key,
            request_api_base or authenticator.get_api_base(),
        )

        account_id = authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            resolved_api_key or "", account_id, session_id
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

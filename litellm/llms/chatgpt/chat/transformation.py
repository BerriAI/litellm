from typing import Any, List, Mapping, Optional, Tuple, Union

from pydantic import BaseModel

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator, get_chatgpt_auth_file
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
        self.authenticator = Authenticator()

    def _resolve_authenticator(self, litellm_params: Union[Mapping[str, object], BaseModel, None]) -> Authenticator:
        auth_file = get_chatgpt_auth_file(litellm_params)
        if auth_file:
            return Authenticator(auth_file=auth_file)
        return self.authenticator

    @staticmethod
    def _get_access_token_or_raise(authenticator: Authenticator, model: str, llm_provider: str) -> str:
        try:
            return authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=llm_provider,
                message=str(e),
            )

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
        litellm_params: Union[Mapping[str, object], BaseModel, None] = None,
    ) -> Tuple[Optional[str], Optional[str], str]:
        authenticator = self._resolve_authenticator(litellm_params)
        dynamic_api_base = authenticator.get_api_base()
        dynamic_api_key = self._get_access_token_or_raise(authenticator, model, custom_llm_provider)
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
        authenticator = self._resolve_authenticator(litellm_params)
        resolved_api_key = (
            self._get_access_token_or_raise(authenticator, model, "chatgpt")
            if get_chatgpt_auth_file(litellm_params)
            else api_key
        )

        validated_headers = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, resolved_api_key, api_base
        )

        account_id = authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(resolved_api_key or "", account_id, session_id)
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
        optional_params = super().map_openai_params(non_default_params, optional_params, model, drop_params)
        optional_params.setdefault("stream", False)
        return optional_params

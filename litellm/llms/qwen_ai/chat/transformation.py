from typing import List, Optional, Tuple

from litellm.exceptions import AuthenticationError
from litellm.llms.dashscope.chat.transformation import DashScopeChatConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    GetAccessTokenError,
    QWEN_DEFAULT_API_BASE,
    get_qwen_user_agent,
    normalize_qwen_api_base,
)


class QwenAIConfig(DashScopeChatConfig):
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()
        self._uses_oauth = True
        self._auth_model: Optional[str] = None

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        dynamic_api_base = normalize_qwen_api_base(
            api_base
            or get_secret_str("QWEN_API_BASE")
            or self.authenticator.get_api_base()
            or QWEN_DEFAULT_API_BASE
        )
        dynamic_api_base = dynamic_api_base or QWEN_DEFAULT_API_BASE

        self._uses_oauth = True
        try:
            dynamic_api_key = self.authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=self._auth_model or "qwen_ai",
                llm_provider="qwen_ai",
                message=str(e),
            )
        return dynamic_api_base, dynamic_api_key

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
        headers = super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        user_agent = headers.get("User-Agent") or get_qwen_user_agent()
        headers.setdefault("User-Agent", user_agent)
        headers.setdefault("X-DashScope-UserAgent", user_agent)
        headers.setdefault("X-DashScope-CacheControl", "enable")
        if self._uses_oauth:
            headers.setdefault("X-DashScope-AuthType", "qwen-oauth")
        return headers

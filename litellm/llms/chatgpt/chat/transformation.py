from typing import Any, List, Optional, Tuple

from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import Authenticator
from ..common_utils import (
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)
from ..db_authenticator import resolve_authenticator
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
        # NOTE: we deliberately do NOT call ``get_access_token()`` here.
        # ``get_llm_provider`` is the resolution stage and runs at proxy
        # startup for every deployment (via ``add_deployment`` cycles).
        # Calling the filesystem ``Authenticator.get_access_token()`` with
        # no tokens on disk triggers ``_login_device_code()``, which
        # prints a device code to stdout and polls OpenAI for up to 15
        # minutes — blocking proxy startup and spamming the logs every
        # 30s once the background scheduler kicks in. Actual auth
        # resolution happens in ``validate_environment`` at request time,
        # which does the right thing via ``resolve_authenticator`` and
        # the DB-backed cache.
        authenticator = resolve_authenticator(api_key, None, self.authenticator)
        dynamic_api_base = api_base or authenticator.get_api_base()
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

        authenticator = resolve_authenticator(
            api_key, litellm_params, self.authenticator
        )
        account_id = authenticator.get_account_id()
        session_id = ensure_chatgpt_session_id(litellm_params)
        default_headers = get_chatgpt_default_headers(
            api_key or "", account_id, session_id
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

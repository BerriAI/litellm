"""
Support for OpenAI's `/v1/chat/completions` endpoint. 

Calls done in OpenAI/openai.py as DataRobot is openai-compatible.
"""

from typing import Optional, Tuple
from litellm.secret_managers.main import get_secret_str
from ...openai_like.chat.transformation import OpenAILikeChatConfig


class DataRobotConfig(OpenAILikeChatConfig):
    @staticmethod
    def _resolve_api_key(api_key: Optional[str] = None) -> str:
        """Attempt to ensure that the API key is set, preferring the user-provided key
        over the secret manager key (``DATAROBOT_API_TOKEN``).

        If both are None, a fake API key is returned for testing.
        """
        return api_key or get_secret_str("DATAROBOT_API_TOKEN") or "fake-api-key"

    @staticmethod
    def _resolve_api_base(api_base: Optional[str] = None) -> Optional[str]:
        """Attempt to ensure that the API base is set, preferring the user-provided key
        over the secret manager key (``DATAROBOT_ENDPOINT``).

        If both are None, a default Llamafile server URL is returned.
        See: https://github.com/Mozilla-Ocho/llamafile/blob/bd1bbe9aabb1ee12dbdcafa8936db443c571eb9d/README.md#L61
        """
        api_base = api_base or get_secret_str("DATAROBOT_ENDPOINT")

        if api_base is None:
            api_base = "https://app.datarobot.com"

        # If the api_base is a deployment URL, we do not append the chat completions path
        if "api/v2/deployments" not in api_base:
            # If the api_base is not a deployment URL, we need to append the chat completions path
            if "api/v2/genai/llmgw/chat/completions" not in api_base:
                api_base += "/api/v2/genai/llmgw/chat/completions"

        # Ensure the url ends with a trailing slash
        if not api_base.endswith("/"):
            api_base += "/"

        return api_base  # type: ignore

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        """Attempts to ensure that the API base and key are set, preferring user-provided values,
        before falling back to secret manager values (``DATAROBOT_ENDPOINT`` and ``DATAROBOT_API_TOKEN``
        respectively).

        If an API key cannot be resolved via either method, a fake key is returned.
        """
        api_base = DataRobotConfig._resolve_api_base(api_base)
        dynamic_api_key = DataRobotConfig._resolve_api_key(api_key)

        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete URL for the API call. Datarobot's API base is set to
        the complete value, so it does not need to be updated to additionally add
        chat completions.

        Returns:
            str: The complete URL for the API call.
        """
        return str(api_base)  # type: ignore

from typing import Any, Optional, Tuple, cast

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig

from ..authenticator import Authenticator
from ..common_utils import GetAPIKeyError


class GithubCopilotConfig(OpenAIConfig):
    GITHUB_COPILOT_API_BASE = "https://api.githubcopilot.com/"
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
        dynamic_api_base = self.authenticator.get_api_base() or self.GITHUB_COPILOT_API_BASE
        try:
            dynamic_api_key = self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=custom_llm_provider,
                message=str(e),
            )
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

    def _transform_messages(
        self,
        messages,
        model: str,
    ):
        import litellm
        disable_copilot_system_to_assistant = litellm.disable_copilot_system_to_assistant 
        if not disable_copilot_system_to_assistant:
            for message in messages:
                if "role" in message and message["role"] == "system":
                    cast(Any, message)["role"] = "assistant"
        return messages

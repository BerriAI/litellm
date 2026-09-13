from typing import Final

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class AIMLChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "aiml"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        # AIML is openai compatible, we just need to set the api_base
        api_base = (
            api_base or get_secret_str("AIML_API_BASE") or "https://api.aimlapi.com/v1"  # Default AIML API base URL
        )
        dynamic_api_key: Final = api_key or get_secret_str("AIML_API_KEY")
        return api_base, dynamic_api_key

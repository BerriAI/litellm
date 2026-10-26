from typing import Final

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

DEFAULT_API_BASE: Final = "http://127.0.0.1:17434/v1"
PLACEHOLDER_API_KEY: Final = "fake-api-key"


class LlmmanChatConfig(OpenAIGPTConfig):
    def get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        return (
            api_base or get_secret_str("LLMMAN_API_BASE") or DEFAULT_API_BASE,
            api_key or get_secret_str("LLMMAN_API_KEY") or PLACEHOLDER_API_KEY,
        )

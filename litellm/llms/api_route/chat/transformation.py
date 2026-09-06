from typing import Final

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str


class APIRouteChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> str | None:
        return "api_route"

    def _get_openai_compatible_provider_info(
        self, api_base: str | None, api_key: str | None
    ) -> tuple[str | None, str | None]:
        resolved_api_base: Final = api_base or get_secret_str("API_ROUTE_BASE_URL") or "https://global.api-route.com/v1"
        resolved_api_key: Final = api_key or get_secret_str("API_ROUTE_API_KEY")
        return resolved_api_base, resolved_api_key

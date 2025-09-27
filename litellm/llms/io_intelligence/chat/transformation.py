"""
Translate from OpenAI's `/v1/chat/completions` to IO Intelligence's `/v1/chat/completions`
"""

from typing import TYPE_CHECKING, Any, Optional, Tuple

from litellm.secret_managers.main import get_secret_str

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class IOIntelligenceChatConfig(OpenAIGPTConfig):
    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("IO_INTELLIGENCE_API_BASE") or "https://api.intelligence.io.solutions/api/v1"  # type: ignore
        dynamic_api_key = (
            api_key or get_secret_str("IO_INTELLIGENCE_API_KEY") or ""
        )  # io_intelligence does not require an api key
        return api_base, dynamic_api_key

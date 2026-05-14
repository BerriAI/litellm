from typing import Optional, Tuple

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import ProviderSpecificModelInfo


TENSORMESH_API_BASE = "https://serverless.tensormesh.ai/v1"


class TensormeshChatConfig(OpenAIGPTConfig):
    """
    Tensormesh serverless and on-demand inference are OpenAI-compatible.

    Serverless uses the default base URL. On-demand deployments pass a routed
    api_base and X-User-Id through LiteLLM's existing extra_headers parameter.
    """

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "tensormesh"

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        return api_key or get_secret_str("TENSORMESH_INFERENCE_API_KEY")

    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        return (
            api_base
            or get_secret_str("TENSORMESH_SERVERLESS_BASE_URL")
            or TENSORMESH_API_BASE
        )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        return self.get_api_base(api_base), self.get_api_key(api_key)

    def get_provider_info(self, model: str) -> ProviderSpecificModelInfo:
        return {
            "supports_function_calling": True,
            "supports_tool_choice": True,
            "supports_native_structured_output": True,
        }

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
        if "max_completion_tokens" in mapped_params:
            mapped_params["max_tokens"] = mapped_params.pop("max_completion_tokens")
        return mapped_params

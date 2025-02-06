"""
Transformation logic from OpenAI /v1/embeddings format to Jina AI's  `/v1/embeddings` format. 

Why separate file? Make it easy to see how transformation works

Docs - https://jina.ai/embeddings/
"""

import types
from typing import List, Optional, Tuple

from litellm import LlmProviders
from litellm.secret_managers.main import get_secret_str


class JinaAIEmbeddingConfig:
    """
    Reference: https://jina.ai/embeddings/
    """

    def __init__(
        self,
    ) -> None:
        locals_ = locals()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self) -> List[str]:
        return ["dimensions"]

    def map_openai_params(
        self, non_default_params: dict, optional_params: dict
    ) -> dict:
        if "dimensions" in non_default_params:
            optional_params["dimensions"] = non_default_params["dimensions"]
        return optional_params

    def _get_openai_compatible_provider_info(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Returns:
            Tuple[str, Optional[str], Optional[str]]:
                - custom_llm_provider: str
                - api_base: str
                - dynamic_api_key: str
        """
        api_base = (
            api_base or get_secret_str("JINA_AI_API_BASE") or "https://api.jina.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or (
            get_secret_str("JINA_AI_API_KEY")
            or get_secret_str("JINA_AI_API_KEY")
            or get_secret_str("JINA_AI_API_KEY")
            or get_secret_str("JINA_AI_TOKEN")
        )
        return LlmProviders.JINA_AI.value, api_base, dynamic_api_key

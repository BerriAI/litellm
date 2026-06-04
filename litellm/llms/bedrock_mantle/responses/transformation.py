"""
Amazon Bedrock Mantle - Responses API backend.

gpt-5.5 / gpt-5.4 on Mantle are exposed ONLY on the `/openai/v1/responses`
path (not the standard `/v1/responses`). Payloads and SSE follow the OpenAI
Responses spec, so this config inherits OpenAIResponsesAPIConfig and overrides
only the endpoint URL and Bearer authentication.

Auth: AWS Bedrock API key as Bearer token (BEDROCK_MANTLE_API_KEY or the
standard AWS_BEARER_TOKEN_BEDROCK), NOT SigV4.
"""

from typing import Optional

from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders

BEDROCK_MANTLE_DEFAULT_REGION = "us-east-1"

# Checked longest/most-specific first so a full endpoint URL collapses to host
# in one pass and the appended path never doubles.
_BASE_SUFFIXES_TO_STRIP = (
    "/openai/v1/responses",
    "/v1/responses",
    "/responses",
    "/openai/v1",
    "/v1",
)


class BedrockMantleResponsesAPIConfig(OpenAIResponsesAPIConfig):
    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK_MANTLE

    def get_complete_url(
        self,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        region = (
            get_secret_str("BEDROCK_MANTLE_REGION")
            or get_secret_str("AWS_REGION")
            or BEDROCK_MANTLE_DEFAULT_REGION
        )
        base = (
            api_base
            or get_secret_str("BEDROCK_MANTLE_API_BASE")
            or f"https://bedrock-mantle.{region}.api.aws"
        )
        base = base.rstrip("/")
        for suffix in _BASE_SUFFIXES_TO_STRIP:
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        return f"{base}/openai/v1/responses"

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = (
            litellm_params.api_key
            or get_secret_str("BEDROCK_MANTLE_API_KEY")
            or get_secret_str("AWS_BEARER_TOKEN_BEDROCK")
        )
        if not api_key:
            raise ValueError(
                "Bedrock Mantle API key is required. Set BEDROCK_MANTLE_API_KEY "
                "(or AWS_BEARER_TOKEN_BEDROCK) or pass api_key."
            )
        headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def supports_native_file_search(self) -> bool:
        return False

    def supports_native_websocket(self) -> bool:
        return False

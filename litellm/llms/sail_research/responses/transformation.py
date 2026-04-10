"""
Sail Responses API — OpenAI-compatible subset.

Sail is a responses-only provider. Key differences from vanilla OpenAI:
- No streaming (stream: true is rejected)
- No instructions, previous_response_id, conversation, prompt params
- No server-side tools (web_search, file_search, code_interpreter, etc.)
- No parallel_tool_calls
- No text.format.type "json_object" (use json_schema instead)
- background: true is supported

Ref: https://api.sailresearch.com
"""

from typing import Any, Dict, List, Optional, Union

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class SailResearchResponsesConfig(OpenAIResponsesAPIConfig):
    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "input",
            "temperature",
            "top_p",
            "max_output_tokens",
            "tools",
            "tool_choice",
            "text",
            "reasoning",
            "metadata",
            "background",
            "extra_headers",
            "extra_query",
            "extra_body",
            "timeout",
        ]

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.SAIL_RESEARCH

    def should_fake_stream(
        self,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        custom_llm_provider: Optional[str] = None,
    ) -> bool:
        """Sail does not support streaming — fake it client-side."""
        return True

    def validate_environment(
        self, headers: dict, model: str, litellm_params: Optional[GenericLiteLLMParams]
    ) -> dict:
        litellm_params = litellm_params or GenericLiteLLMParams()
        api_key = litellm_params.api_key or get_secret_str("SAIL_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def get_complete_url(self, api_base: Optional[str], litellm_params: dict) -> str:
        api_base = (
            api_base
            or get_secret_str("SAIL_API_BASE")
            or "https://api.sailresearch.com"
        )
        return f"{api_base.rstrip('/')}/v1/responses"

    def transform_responses_api_request(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        response_api_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        # Strip json_object format type — Sail only accepts json_schema
        text_config = response_api_optional_request_params.get("text")
        if isinstance(text_config, dict):
            fmt = text_config.get("format")
            if isinstance(fmt, dict) and fmt.get("type") == "json_object":
                text_config = {**text_config}
                del text_config["format"]
                response_api_optional_request_params = {
                    **response_api_optional_request_params,
                    "text": text_config,
                }

        # Never send stream: true to Sail
        response_api_optional_request_params.pop("stream", None)

        return super().transform_responses_api_request(
            model=model,
            input=input,
            response_api_optional_request_params=response_api_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def supports_native_websocket(self) -> bool:
        return False

"""
Snowflake Cortex REST API — Chat Transformation

Routes to the native OpenAI-compatible endpoint:
  POST /api/v2/cortex/v1/chat/completions

Previously used the legacy endpoint /api/v2/cortex/inference:complete which
required Snowflake-specific payload transformations and did not support the
full OpenAI parameter surface. This version uses the native endpoint which
accepts standard OpenAI chat completions format directly.

Ref: https://docs.snowflake.com/en/user-guide/snowflake-cortex/cortex-rest-api
"""

from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

from ...openai_like.chat.transformation import OpenAIGPTConfig
from ..utils import SnowflakeBaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SnowflakeConfig(SnowflakeBaseConfig, OpenAIGPTConfig):
    """
    Snowflake Cortex REST API — OpenAI-compatible endpoint.

    Endpoint: POST /api/v2/cortex/v1/chat/completions

    Supports all Snowflake Cortex models (Llama, Mistral, DeepSeek, Snowflake
    Arctic, and Claude series) via the OpenAI-compatible interface.

    For Claude-specific features (thinking, cache_control, extended context),
    use SnowflakeCortexAnthropicConfig which routes to /api/v2/cortex/v1/messages.

    Auth:
        PAT:  api_key="pat/<token>"  →  X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN
        JWT:  api_key="<jwt>"        →  X-Snowflake-Authorization-Token-Type: KEYPAIR_JWT
    """

    @classmethod
    def get_config(cls):
        return super().get_config()

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "temperature",
            "max_tokens",
            "max_completion_tokens",
            "top_p",
            "stream",
            "response_format",
            "tools",
            "tool_choice",
        ]

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
        Returns the native OpenAI-compatible Cortex REST API endpoint.

        _get_api_base normalizes api_base to:
            https://{account}.snowflakecomputing.com/api/v2

        We append:
            /cortex/v1/chat/completions

        Resulting in:
            https://{account}.snowflakecomputing.com/api/v2/cortex/v1/chat/completions
        """
        api_base = self._get_api_base(api_base, optional_params)
        return f"{api_base}/cortex/v1/chat/completions"

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform to OpenAI chat completions format.

        The native /chat/completions endpoint accepts standard OpenAI format
        directly — no Snowflake-specific tool_spec transformation required.
        """
        stream: bool = optional_params.pop("stream", False) or False
        extra_body = optional_params.pop("extra_body", {})

        max_tokens = optional_params.pop("max_tokens", None)
        max_completion_tokens = optional_params.pop("max_completion_tokens", None)
        resolved_max = max_completion_tokens or max_tokens

        body: dict = {
            "model": model,
            "messages": messages,
            "stream": stream,
            **optional_params,
            **extra_body,
        }

        if resolved_max is not None:
            body["max_tokens"] = resolved_max

        return body

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform from standard OpenAI chat completions response.

        The native endpoint returns standard OpenAI format — no content_list
        transformation required.
        """
        response_json = raw_response.json()

        logging_obj.post_call(
            input=messages,
            api_key="",
            original_response=response_json,
            additional_args={"complete_input_dict": request_data},
        )

        returned_response = ModelResponse(**response_json)
        returned_response.model = "snowflake/" + (returned_response.model or "")

        if model is not None:
            returned_response._hidden_params["model"] = model

        return returned_response

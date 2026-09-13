"""
Transformation for Bedrock Mantle (Claude Mythos Preview) - /messages endpoint

Inherits all Messages API request/response transformations from
AmazonAnthropicClaudeMessagesConfig. Overrides only the URL and model-prefix
stripping that are specific to the bedrock-mantle endpoint.
"""

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Final

import httpx

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.bedrock.common_utils import build_mantle_messages_url
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
    AnthropicUsage,
)
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonMantleMessagesConfig(AmazonAnthropicClaudeMessagesConfig):
    """
    Config for the bedrock-mantle /messages endpoint (Claude Mythos Preview).

    The mantle endpoint uses the Anthropic Messages API format and requires the
    model ID in the request body (unlike Bedrock Invoke which puts it in the URL).
    """

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        region: Final = self._get_aws_region_name(optional_params=optional_params, model=model)
        return build_mantle_messages_url(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.get("aws_bedrock_runtime_endpoint"),
            region=region,
        )

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: list[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:
        headers, api_base = super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        project_id: Final = litellm_params.get("aws_bedrock_project_id")
        if project_id:
            headers["anthropic-workspace"] = project_id
        return headers, api_base

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],
        anthropic_messages_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> dict:
        # Strip "mantle/" routing prefix to get the real model ID
        model_id: Final = model.replace("mantle/", "", 1)

        request: Final = super().transform_anthropic_messages_request(
            model=model_id,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Parent (AmazonAnthropicClaudeMessagesConfig) removes "model" and
        # "stream" from the body (Bedrock Invoke puts the model in the URL and
        # streams via a dedicated endpoint). The mantle endpoint (Messages API)
        # requires both in the request body.
        stream_fields: Final[dict[str, bool]] = (
            {"stream": True} if anthropic_messages_optional_request_params.get("stream") is True else {}
        )
        return {**request, "model": model_id, **stream_fields}

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        response: Final = super().transform_anthropic_messages_response(
            model=model,
            raw_response=raw_response,
            logging_obj=logging_obj,
        )
        existing_usage: Final[AnthropicUsage] = response.get("usage") or AnthropicUsage()
        normalized_usage: Final[AnthropicUsage] = {
            "input_tokens": 0,
            "output_tokens": 0,
            **existing_usage,
        }
        return {**response, "usage": normalized_usage}

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        return AnthropicMessagesConfig.get_async_streaming_response_iterator(
            self,
            model=model,
            httpx_response=httpx_response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
        )

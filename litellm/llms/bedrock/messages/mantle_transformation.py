"""
Transformation for Bedrock Mantle (Claude Mythos Preview) - /messages endpoint

Inherits all Messages API request/response transformations from
AmazonAnthropicClaudeMessagesConfig. Overrides only the URL and model-prefix
stripping that are specific to the bedrock-mantle endpoint.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from litellm.llms.bedrock.common_utils import build_mantle_messages_url
from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
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
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        region = self._get_aws_region_name(optional_params=optional_params, model=model)
        return build_mantle_messages_url(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.get(
                "aws_bedrock_runtime_endpoint"
            ),
            region=region,
        )

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        headers, api_base = super().validate_anthropic_messages_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        project_id = litellm_params.get("aws_bedrock_project_id")
        if project_id:
            headers["anthropic-workspace"] = project_id
        return headers, api_base

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        # Strip "mantle/" routing prefix to get the real model ID
        model_id = model.replace("mantle/", "", 1)

        request = super().transform_anthropic_messages_request(
            model=model_id,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        # Parent (AmazonAnthropicClaudeMessagesConfig) removes "model" from the
        # body (Bedrock Invoke puts model in the URL). The mantle endpoint
        # (Messages API) requires "model" in the request body.
        request["model"] = model_id
        return request

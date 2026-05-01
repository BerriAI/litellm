"""
Transformation for Bedrock Mantle (Claude Mythos Preview) - /messages endpoint

Inherits all Messages API request/response transformations from
AmazonAnthropicClaudeMessagesConfig. Overrides only the URL and model-prefix
stripping that are specific to the bedrock-mantle endpoint.
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm.llms.bedrock.messages.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeMessagesConfig,
)
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

MANTLE_ENDPOINT_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/v1/messages"

# AWS region names are documented as ``[a-z]{2}(-gov|-iso[a-z]?)?-[a-z]+-\d+``
# (e.g. ``us-east-1``, ``us-gov-east-1``, ``us-isob-east-1``). Lowercase
# alphanumerics + hyphens covers every known region, and refuses anything
# that could break out of the URL authority — including ``@`` (userinfo),
# ``/`` (path), ``:`` (port), ``%`` (percent-encoded delimiters), and ``.``
# (subdomain hop). VERIA-88.
_AWS_REGION_PATTERN = re.compile(r"^[a-z0-9-]+$")


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
        # ``aws_region_name`` originates from user-supplied ``optional_params``
        # and is interpolated directly into the URL authority. Without
        # validation, a value like ``us-east-1@attacker.example`` makes the
        # URL parser treat ``bedrock-mantle.us-east-1`` as basic-auth
        # userinfo and routes the SigV4-signed request (with the operator's
        # AWS access-key-ID in the headers) to the attacker. VERIA-88.
        if not _AWS_REGION_PATTERN.fullmatch(region or ""):
            raise ValueError(
                "Invalid AWS region for Bedrock Mantle: must match "
                "[a-z0-9-]+ (e.g. 'us-east-1')"
            )
        return MANTLE_ENDPOINT_TEMPLATE.format(region=region)

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

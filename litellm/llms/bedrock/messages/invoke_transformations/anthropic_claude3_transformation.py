from typing import TYPE_CHECKING, Any, Dict, List, Optional

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonAnthropicClaude3MessagesConfig(
    AnthropicMessagesConfig,
    AmazonInvokeConfig,
):
    """
    Call Claude model family in the /v1/messages API spec
    """

    DEFAULT_BEDROCK_ANTHROPIC_API_VERSION = "bedrock-2023-05-31"

    def __init__(self, **kwargs):
        BaseAnthropicMessagesConfig.__init__(self, **kwargs)
        AmazonInvokeConfig.__init__(self, **kwargs)

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> dict:
        return AmazonInvokeConfig.sign_request(
            self=self,
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        return headers

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return AmazonInvokeConfig.get_complete_url(
            self=self,
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        anthropic_messages_request = AnthropicMessagesConfig.transform_anthropic_messages_request(
            self=self,
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )

        #########################################################
        ############## BEDROCK Invoke SPECIFIC TRANSFORMATION ###
        #########################################################

        # 1. anthropic_version is required for all claude models
        if "anthropic_version" not in anthropic_messages_request:
            anthropic_messages_request[
                "anthropic_version"
            ] = self.DEFAULT_BEDROCK_ANTHROPIC_API_VERSION

        # 2. `stream` is not allowed in request body for bedrock invoke
        if "stream" in anthropic_messages_request:
            anthropic_messages_request.pop("stream", None)

        # 3. `model` is not allowed in request body for bedrock invoke
        if "model" in anthropic_messages_request:
            anthropic_messages_request.pop("model", None)
        return anthropic_messages_request

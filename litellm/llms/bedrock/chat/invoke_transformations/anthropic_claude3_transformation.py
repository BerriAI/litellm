from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import get_anthropic_beta_from_headers
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonAnthropicClaudeConfig(AmazonInvokeConfig, AnthropicConfig):
    """
    Reference:
        https://us-west-2.console.aws.amazon.com/bedrock/home?region=us-west-2#/providers?model=claude
        https://docs.anthropic.com/claude/docs/models-overview#model-comparison
        https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-request-response.html

    Supported Params for the Amazon / Anthropic Claude models (Claude 3, Claude 4, etc.):
    Supports anthropic_beta parameter for beta features like:
    - computer-use-2025-01-24 (Claude 3.7 Sonnet)
    - computer-use-2024-10-22 (Claude 3.5 Sonnet v2)
    - token-efficient-tools-2025-02-19 (Claude 3.7 Sonnet)
    - interleaved-thinking-2025-05-14 (Claude 4 models)
    - output-128k-2025-02-19 (Claude 3.7 Sonnet)
    - dev-full-thinking-2025-05-14 (Claude 4 models)
    - context-1m-2025-08-07 (Claude Sonnet 4)
    """

    anthropic_version: str = "bedrock-2023-05-31"

    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "bedrock"

    def get_supported_openai_params(self, model: str) -> List[str]:
        return AnthropicConfig.get_supported_openai_params(self, model)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return AnthropicConfig.map_openai_params(
            self,
            non_default_params,
            optional_params,
            model,
            drop_params,
        )


    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # Filter out AWS authentication parameters before passing to Anthropic transformation
        # AWS params should only be used for signing requests, not included in request body
        filtered_params = {
            k: v
            for k, v in optional_params.items()
            if k not in self.aws_authentication_params
        }
        
        _anthropic_request = AnthropicConfig.transform_request(
            self,
            model=model,
            messages=messages,
            optional_params=filtered_params, 
            litellm_params=litellm_params,
            headers=headers,
        )

        _anthropic_request.pop("model", None)
        _anthropic_request.pop("stream", None)
        if "anthropic_version" not in _anthropic_request:
            _anthropic_request["anthropic_version"] = self.anthropic_version

        # Handle anthropic_beta from user headers
        anthropic_beta_list = get_anthropic_beta_from_headers(headers)
        if anthropic_beta_list:
            _anthropic_request["anthropic_beta"] = anthropic_beta_list

        return _anthropic_request

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
        return AnthropicConfig.transform_response(
            self,
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

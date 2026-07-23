"""
Transformation for Bedrock Mantle (Claude Mythos Preview)

https://docs.aws.amazon.com/bedrock/latest/userguide/model-card-anthropic-claude-mythos-preview.html

The bedrock-mantle endpoint uses the Anthropic Messages API format but is served
at a different endpoint (bedrock-mantle.{region}.api.aws) with AWS SigV4 auth.
"""

from typing import TYPE_CHECKING, Any, AsyncIterator, Iterator, List, Optional

from litellm.llms.bedrock.chat.invoke_transformations.anthropic_claude3_transformation import (
    AmazonAnthropicClaudeConfig,
)
from litellm.llms.bedrock.common_utils import build_mantle_messages_url
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonMantleConfig(AmazonAnthropicClaudeConfig):
    """
    Config for the bedrock-mantle endpoint (Claude Mythos Preview).

    Uses the Anthropic Messages API format with AWS SigV4 auth, but at a
    different endpoint from bedrock-runtime. Model ID goes in the request body.

    Usage: model="bedrock/mantle/anthropic.claude-mythos-preview"
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
            aws_bedrock_runtime_endpoint=optional_params.get("aws_bedrock_runtime_endpoint"),
            region=region,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        headers = super().validate_environment(
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
            headers["anthropic-workspace-id"] = project_id
        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        # Strip the "mantle/" routing prefix to get the real model ID
        model_id = model.replace("mantle/", "", 1)

        request = self._build_bedrock_anthropic_request_base(
            model=model_id,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        # The parent strips "model" and "stream" from the body (Invoke API puts
        # the model in the URL and streams via a dedicated endpoint). The mantle
        # endpoint (Messages API) requires both in the body.
        return self._restore_mantle_body_fields(
            request=request,
            model_id=model_id,
            optional_params=optional_params,
        )

    async def async_transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        model_id = model.replace("mantle/", "", 1)

        request = self._build_bedrock_anthropic_request_base(
            model=model_id,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        await self._async_convert_document_url_sources_to_base64(request)
        return self._restore_mantle_body_fields(
            request=request,
            model_id=model_id,
            optional_params=optional_params,
        )

    @staticmethod
    def _restore_mantle_body_fields(request: dict, model_id: str, optional_params: dict) -> dict:
        stream_fields: dict = {"stream": True} if optional_params.get("stream") is True else {}
        return {**request, "model": model_id, **stream_fields}

    @property
    def has_custom_stream_wrapper(self) -> bool:
        return False

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | ModelResponse,
        sync_stream: bool,
        json_mode: Optional[bool] = False,
    ) -> Any:
        from litellm.llms.anthropic.chat.handler import ModelResponseIterator

        return ModelResponseIterator(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )

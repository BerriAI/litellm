"""
Translate from OpenAI's `/v1/chat/completions` to Sagemaker's `/invocations` API

Called if Sagemaker endpoint supports HF Messages API.

LiteLLM Docs: https://docs.litellm.ai/docs/providers/aws_sagemaker#sagemaker-messages-api
Huggingface Docs: https://huggingface.co/docs/text-generation-inference/en/messages_api
"""

from typing import TYPE_CHECKING, Any, List, Optional, Tuple, Union, cast

import httpx
from httpx._models import Headers

from litellm.litellm_core_utils.logging_utils import track_llm_api_timing
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import LlmProviders

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import AWSEventStreamDecoder, SagemakerError

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class SagemakerChatConfig(OpenAIGPTConfig, BaseAWSLLM):
    def __init__(self, **kwargs):
        OpenAIGPTConfig.__init__(self, **kwargs)
        BaseAWSLLM.__init__(self, **kwargs)

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return SagemakerError(
            status_code=status_code, message=error_message, headers=headers
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
        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params,
            model=model,
            model_id=None,
        )
        if stream is True:
            api_base = f"https://runtime.sagemaker.{aws_region_name}.amazonaws.com/endpoints/{model}/invocations-response-stream"
        else:
            api_base = f"https://runtime.sagemaker.{aws_region_name}.amazonaws.com/endpoints/{model}/invocations"

        sagemaker_base_url = cast(
            Optional[str], optional_params.get("sagemaker_base_url")
        )
        if sagemaker_base_url is not None:
            api_base = sagemaker_base_url

        return api_base

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        fake_stream: Optional[bool] = None,
    ) -> Tuple[dict, Optional[bytes]]:
        return self._sign_request(
            service_name="sagemaker",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    @property
    def has_custom_stream_wrapper(self) -> bool:
        return True

    @property
    def supports_stream_param_in_request_body(self) -> bool:
        return False

    @track_llm_api_timing()
    def get_sync_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> CustomStreamWrapper:
        if client is None or isinstance(client, AsyncHTTPHandler):
            client = _get_httpx_client(params={})

        try:
            response = client.post(
                api_base,
                headers=headers,
                data=signed_json_body if signed_json_body is not None else data,
                stream=True,
                logging_obj=logging_obj,
            )
        except httpx.HTTPStatusError as e:
            raise SagemakerError(
                status_code=e.response.status_code, message=e.response.text
            )

        if response.status_code != 200:
            raise SagemakerError(
                status_code=response.status_code, message=response.text
            )

        custom_stream_decoder = AWSEventStreamDecoder(model="", is_messages_api=True)
        completion_stream = custom_stream_decoder.iter_bytes(
            response.iter_bytes(chunk_size=1024)
        )

        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="sagemaker_chat",
            logging_obj=logging_obj,
        )
        return streaming_response

    @track_llm_api_timing()
    async def get_async_custom_stream_wrapper(
        self,
        model: str,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        api_base: str,
        headers: dict,
        data: dict,
        messages: list,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        json_mode: Optional[bool] = None,
        signed_json_body: Optional[bytes] = None,
    ) -> CustomStreamWrapper:
        if client is None or isinstance(client, HTTPHandler):
            client = get_async_httpx_client(
                llm_provider=LlmProviders.SAGEMAKER_CHAT, params={}
            )

        try:
            response = await client.post(
                api_base,
                headers=headers,
                data=signed_json_body if signed_json_body is not None else data,
                stream=True,
                logging_obj=logging_obj,
            )
        except httpx.HTTPStatusError as e:
            raise SagemakerError(
                status_code=e.response.status_code, message=e.response.text
            )

        if response.status_code != 200:
            raise SagemakerError(
                status_code=response.status_code, message=response.text
            )

        custom_stream_decoder = AWSEventStreamDecoder(model="", is_messages_api=True)
        completion_stream = custom_stream_decoder.aiter_bytes(
            response.aiter_bytes(chunk_size=1024)
        )

        streaming_response = CustomStreamWrapper(
            completion_stream=completion_stream,
            model=model,
            custom_llm_provider="sagemaker_chat",
            logging_obj=logging_obj,
        )
        return streaming_response

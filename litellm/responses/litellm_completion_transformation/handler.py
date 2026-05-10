"""
Handler for transforming responses api requests to litellm.completion requests
"""

from typing import Any, Coroutine, Dict, Optional, Union

import litellm
from litellm.responses.litellm_completion_transformation.streaming_iterator import (
    LiteLLMCompletionStreamingIterator,
)
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ResponsesAPIOptionalRequestParams,
    ResponsesAPIResponse,
)
from litellm.types.utils import ModelResponse


def _is_domestic_model_endpoint(api_base: Optional[str]) -> bool:
    """
    判断是否是国内模型 provider endpoint。
    国内模型不支持某些 Codex CLI 特有的参数和工具类型。

    Args:
        api_base: API endpoint URL

    Returns:
        bool: True 表示是国内模型 endpoint，需要做兼容处理
    """
    if not api_base:
        return False

    domestic_endpoints = [
        "dashscope.aliyuncs.com",  # 阿里云 DashScope
        "ark.cn-beijing.volces.com",  # 火山引擎 Volcengine
        "api.minimaxi.com",  # MiniMax 官方
        "xiaomimimo.com",  # 小米 MiMo
        "api.deepseek.com",  # DeepSeek 官方
    ]

    return any(endpoint in str(api_base) for endpoint in domestic_endpoints)


class LiteLLMCompletionTransformationHandler:
    def response_api_handler(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        _is_async: bool = False,
        stream: Optional[bool] = None,
        extra_headers: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Union[
        ResponsesAPIResponse,
        BaseResponsesAPIStreamingIterator,
        Coroutine[
            Any, Any, Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]
        ],
    ]:
        litellm_completion_request: dict = (
            LiteLLMCompletionResponsesConfig.transform_responses_api_request_to_chat_completion_request(
                model=model,
                input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                stream=stream,
                extra_headers=extra_headers,
                **kwargs,
            )
        )

        if _is_async:
            return self.async_response_api_handler(
                litellm_completion_request=litellm_completion_request,
                request_input=input,
                responses_api_request=responses_api_request,
                **kwargs,
            )

        completion_args = {}
        completion_args.update(kwargs)

        # 只对国内模型 endpoint 过滤 client_metadata (Codex CLI 0.130.0 兼容修复)
        # 国内模型不支持 Codex CLI 特有的 client_metadata 参数
        api_base = kwargs.get("api_base") or litellm_completion_request.get("api_base")
        if _is_domestic_model_endpoint(api_base):
            completion_args.pop("client_metadata", None)

        completion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = litellm.completion(
            **completion_args,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=model,
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=input,
                responses_api_request=responses_api_request,
                custom_llm_provider=custom_llm_provider,
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(
            f"Unexpected response type: {type(litellm_completion_response)}"
        )

    async def async_response_api_handler(
        self,
        litellm_completion_request: dict,
        request_input: Union[str, ResponseInputParam],
        responses_api_request: ResponsesAPIOptionalRequestParams,
        **kwargs,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
        previous_response_id: Optional[str] = responses_api_request.get(
            "previous_response_id"
        )
        if previous_response_id:
            litellm_completion_request = await LiteLLMCompletionResponsesConfig.async_responses_api_session_handler(
                previous_response_id=previous_response_id,
                litellm_completion_request=litellm_completion_request,
            )

        acompletion_args = {}
        acompletion_args.update(kwargs)

        # 只对国内模型 endpoint 过滤 client_metadata (Codex CLI 0.130.0 兼容修复)
        # 国内模型不支持 Codex CLI 特有的 client_metadata 参数
        api_base = kwargs.get("api_base") or litellm_completion_request.get("api_base")
        if _is_domestic_model_endpoint(api_base):
            acompletion_args.pop("client_metadata", None)

        acompletion_args.update(litellm_completion_request)

        litellm_completion_response: Union[
            ModelResponse, litellm.CustomStreamWrapper
        ] = await litellm.acompletion(
            **acompletion_args,
        )

        if isinstance(litellm_completion_response, ModelResponse):
            responses_api_response: ResponsesAPIResponse = (
                LiteLLMCompletionResponsesConfig.transform_chat_completion_response_to_responses_api_response(
                    chat_completion_response=litellm_completion_response,
                    request_input=request_input,
                    responses_api_request=responses_api_request,
                )
            )

            return responses_api_response

        elif isinstance(litellm_completion_response, litellm.CustomStreamWrapper):
            return LiteLLMCompletionStreamingIterator(
                model=litellm_completion_request.get("model") or "",
                litellm_custom_stream_wrapper=litellm_completion_response,
                request_input=request_input,
                responses_api_request=responses_api_request,
                custom_llm_provider=litellm_completion_request.get(
                    "custom_llm_provider"
                ),
                litellm_metadata=kwargs.get("litellm_metadata", {}),
            )
        raise ValueError(
            f"Unexpected response type: {type(litellm_completion_response)}"
        )

"""
Handler for transforming responses api requests to litellm.completion requests
"""

from typing import Any, Coroutine, Dict, Optional, Union

import litellm
from litellm.llms.domestic.domestic_utils import is_domestic_model_or_endpoint
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
        completion_args.update(litellm_completion_request)

        # 国内模型兼容过滤 (Codex CLI 0.130.0 + 国内模型扩展参数)
        # 国内模型不支持 Codex CLI / OpenAI 特有的参数，需要过滤
        # 注意：必须在合并 kwargs 和 litellm_completion_request 之后过滤
        # 获取实际模型名（可能来自 litellm_params.model 或 completion_args.model）
        actual_model = completion_args.get("model", model)
        api_base = completion_args.get("api_base", "")
        # 从 litellm_params 提取实际模型名（如果是 openai/qwen3.6-plus 格式）
        litellm_params_model = completion_args.get("litellm_params", {}).get(
            "model", ""
        )
        if litellm_params_model and litellm_params_model.startswith("openai/"):
            actual_model = litellm_params_model

        if is_domestic_model_or_endpoint(actual_model, api_base):
            # Codex CLI 特有参数
            completion_args.pop("client_metadata", None)
            # OpenAI 扩展参数（国内模型不支持）
            completion_args.pop("reasoning_effort", None)  # reasoning 模式
            completion_args.pop("reasoning", None)  # reasoning 参数 (Codex 0.130.0+)
            completion_args.pop("coding_plan", None)  # Codex coding plan
            completion_args.pop("parallel_tool_calls", None)  # 并行工具调用
            # 其他国内模型可能不支持的参数
            completion_args.pop("stream_options", None)  # stream 选项
            completion_args.pop("modalities", None)  # 多模态参数
            completion_args.pop("prediction", None)  # 预测参数
            completion_args.pop("audio", None)  # 音频参数
            completion_args.pop("store", None)  # 存储参数
            completion_args.pop("include", None)  # 包含参数
            completion_args.pop("prompt_cache_key", None)  # 缓存键
            # LiteLLM 内部参数（不应发送给 API）
            completion_args.pop("litellm_metadata", None)
            completion_args.pop("litellm_logging_obj", None)
            completion_args.pop("litellm_trace_id", None)
            completion_args.pop("litellm_call_id", None)
            completion_args.pop("proxy_server_request", None)
            completion_args.pop("shared_session", None)
            completion_args.pop("model_info", None)
            completion_args.pop("secret_fields", None)
            completion_args.pop("use_in_pass_through", None)
            completion_args.pop("use_litellm_proxy", None)
            completion_args.pop("merge_reasoning_content_in_choices", None)
            completion_args.pop("supports_function_calling", None)
            completion_args.pop("caching", None)
            completion_args.pop("extra_body", None)
            completion_args.pop("custom_llm_provider", None)
            completion_args.pop("max_retries", None)

        # DEBUG: 国内模型参数过滤日志
        import sys

        print(
            f"[DEBUG handler.py sync] actual_model={actual_model}, api_base={api_base}",
            file=sys.stderr,
        )
        print(
            f"[DEBUG handler.py sync] is_domestic={is_domestic_model_or_endpoint(actual_model, api_base)}",
            file=sys.stderr,
        )
        print(
            f"[DEBUG handler.py sync] completion_args keys: {list(completion_args.keys())}",
            file=sys.stderr,
        )

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
        acompletion_args.update(litellm_completion_request)

        # 国内模型兼容过滤 (Codex CLI 0.130.0 + 国内模型扩展参数)
        # 国内模型不支持 Codex CLI / OpenAI 特有的参数，需要过滤
        # 注意：必须在合并 kwargs 和 litellm_completion_request 之后过滤
        # 获取实际模型名（可能来自 litellm_params.model 或 completion_args.model）
        actual_model = acompletion_args.get("model", "")
        api_base = acompletion_args.get("api_base", "")
        # 从 litellm_params 提取实际模型名（如果是 openai/qwen3.6-plus 格式）
        litellm_params_model = acompletion_args.get("litellm_params", {}).get(
            "model", ""
        )
        if litellm_params_model and litellm_params_model.startswith("openai/"):
            actual_model = litellm_params_model

        if is_domestic_model_or_endpoint(actual_model, api_base):
            # Codex CLI 特有参数
            acompletion_args.pop("client_metadata", None)
            # OpenAI 扩展参数（国内模型不支持）
            acompletion_args.pop("reasoning_effort", None)  # reasoning 模式
            acompletion_args.pop("reasoning", None)  # reasoning 参数 (Codex 0.130.0+)
            acompletion_args.pop("coding_plan", None)  # Codex coding plan
            acompletion_args.pop("parallel_tool_calls", None)  # 并行工具调用
            # 其他国内模型可能不支持的参数
            acompletion_args.pop("stream_options", None)  # stream 选项
            acompletion_args.pop("modalities", None)  # 多模态参数
            acompletion_args.pop("prediction", None)  # 预测参数
            acompletion_args.pop("audio", None)  # 音频参数
            acompletion_args.pop("store", None)  # 存储参数
            acompletion_args.pop("include", None)  # 包含参数
            acompletion_args.pop("prompt_cache_key", None)  # 缓存键
            # LiteLLM 内部参数（不应发送给 API）
            acompletion_args.pop("litellm_metadata", None)
            acompletion_args.pop("litellm_logging_obj", None)
            acompletion_args.pop("litellm_trace_id", None)
            acompletion_args.pop("litellm_call_id", None)
            acompletion_args.pop("proxy_server_request", None)
            acompletion_args.pop("shared_session", None)
            acompletion_args.pop("model_info", None)
            acompletion_args.pop("secret_fields", None)
            acompletion_args.pop("use_in_pass_through", None)
            acompletion_args.pop("use_litellm_proxy", None)
            acompletion_args.pop("merge_reasoning_content_in_choices", None)
            acompletion_args.pop("supports_function_calling", None)
            acompletion_args.pop("caching", None)
            acompletion_args.pop("extra_body", None)
            acompletion_args.pop("custom_llm_provider", None)
            acompletion_args.pop("max_retries", None)

        # DEBUG: 国内模型参数过滤日志
        import sys

        print(
            f"[DEBUG handler.py async] actual_model={actual_model}, api_base={api_base}",
            file=sys.stderr,
        )
        print(
            f"[DEBUG handler.py async] is_domestic={is_domestic_model_or_endpoint(actual_model, api_base)}",
            file=sys.stderr,
        )
        print(
            f"[DEBUG handler.py async] acompletion_args keys: {list(acompletion_args.keys())}",
            file=sys.stderr,
        )

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

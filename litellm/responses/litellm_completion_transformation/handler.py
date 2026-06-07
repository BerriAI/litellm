"""
Handler for transforming responses api requests to litellm.completion requests
"""

import copy
import json
import logging
from typing import Any, Coroutine, Dict, List, Optional, Tuple, Union

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

_domestic_logger = logging.getLogger("LiteLLM.DomesticFilter")


class LiteLLMCompletionTransformationHandler:
    @staticmethod
    def _ensure_all_tool_calls_have_valid_json_arguments(
        messages: List[Any],
    ) -> List[Any]:
        """
        Ensure all tool_calls in messages have valid JSON arguments.

        国内模型（火山引擎、阿里云）要求 function.arguments 必须是严格的 JSON 格式。
        如果 arguments 不是有效的 JSON，将其替换为 "{}"。

        Args:
            messages: List of chat completion messages

        Returns:
            List of messages with corrected tool_calls arguments
        """
        fixed_count = 0
        result_messages = []

        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role")
                tool_calls = msg.get("tool_calls")
            else:
                role = getattr(msg, "role", None)
                tool_calls = getattr(msg, "tool_calls", None)

            if (
                role != "assistant"
                or not tool_calls
                or not isinstance(tool_calls, list)
            ):
                result_messages.append(msg)
                continue

            needs_rebuild = False
            new_tool_calls = []

            for tc in tool_calls:
                if isinstance(tc, dict):
                    tc_id = tc.get("id")
                    tc_type = tc.get("type", "function")
                    func = tc.get("function")
                else:
                    tc_id = getattr(tc, "id", None)
                    tc_type = getattr(tc, "type", "function")
                    func = getattr(tc, "function", None)

                if func is not None:
                    if isinstance(func, dict):
                        func_name = func.get("name", "")
                        args = func.get("arguments")
                    else:
                        func_name = getattr(func, "name", "")
                        args = getattr(func, "arguments", None)
                else:
                    func_name = ""
                    args = None

                fixed_args = None

                if args is None:
                    fixed_args = "{}"
                    needs_rebuild = True
                    fixed_count += 1
                elif isinstance(args, str):
                    args_stripped = args.strip()
                    if not args_stripped:
                        fixed_args = "{}"
                        needs_rebuild = True
                        fixed_count += 1
                    else:
                        try:
                            json.loads(args_stripped)
                            fixed_args = args_stripped
                        except (json.JSONDecodeError, ValueError):
                            fixed_args = "{}"
                            needs_rebuild = True
                            fixed_count += 1
                elif isinstance(args, dict):
                    fixed_args = json.dumps(args)
                    needs_rebuild = True
                else:
                    try:
                        args_str = str(args)
                        json.loads(args_str)
                        fixed_args = args_str
                    except (json.JSONDecodeError, ValueError):
                        fixed_args = "{}"
                        needs_rebuild = True
                        fixed_count += 1

                new_tool_calls.append(
                    {
                        "id": tc_id,
                        "type": tc_type,
                        "function": {
                            "name": func_name,
                            "arguments": fixed_args if fixed_args is not None else "{}",
                        },
                    }
                )

            if needs_rebuild:
                if isinstance(msg, dict):
                    new_msg = copy.deepcopy(msg)
                    new_msg["tool_calls"] = new_tool_calls
                else:
                    new_msg = {
                        "role": role,
                        "tool_calls": new_tool_calls,
                    }
                    for field in ["content", "name", "audio"]:
                        val = (
                            getattr(msg, field, None)
                            if not isinstance(msg, dict)
                            else msg.get(field)
                        )
                        if val is not None:
                            new_msg[field] = val
                result_messages.append(new_msg)
            else:
                result_messages.append(msg)

        if fixed_count > 0:
            _domestic_logger.info(
                "[DomesticFilter] Fixed %d invalid tool_calls arguments", fixed_count
            )

        return result_messages

    @staticmethod
    def _resolve_model_and_api_base(
        args: dict, fallback_model: str = ""
    ) -> Tuple[str, str]:
        """Extract actual model name and api_base from completion args."""
        actual_model = args.get("model", fallback_model)
        api_base = args.get("api_base", "")
        litellm_params_model = args.get("litellm_params", {}).get("model", "")
        if litellm_params_model and litellm_params_model.startswith("openai/"):
            actual_model = litellm_params_model
        return actual_model, api_base

    @staticmethod
    def _filter_domestic_params(args: dict) -> dict:
        """
        Filter unsupported parameters for domestic (Chinese) models.

        This filters parameters that domestic model providers don't support,
        while preserving LiteLLM internal bookkeeping parameters needed for
        proxy spend tracking and logging.
        """
        actual_model, api_base = (
            LiteLLMCompletionTransformationHandler._resolve_model_and_api_base(args)
        )

        if not is_domestic_model_or_endpoint(actual_model, api_base):
            return args

        # ========== Responses API 专用参数（已转换，不应传递）==========
        args.pop("text", None)
        args.pop("reasoning", None)
        args.pop("instructions", None)
        args.pop("background", None)
        args.pop("truncation", None)
        args.pop("max_output_tokens", None)

        # ========== Codex CLI 特有参数 ==========
        args.pop("client_metadata", None)
        args.pop("coding_plan", None)

        # ========== 旧格式参数 ==========
        args.pop("functions", None)
        args.pop("function_call", None)

        # ========== 厂商不支持的参数 ==========
        args.pop("stream_options", None)
        args.pop("modalities", None)
        args.pop("prediction", None)
        args.pop("audio", None)
        args.pop("store", None)
        args.pop("include", None)
        args.pop("prompt_cache_key", None)
        args.pop("caching", None)
        args.pop("extra_body", None)
        args.pop("parallel_tool_calls", None)

        # ========== 注意：以下参数不能 pop ==========
        # litellm_metadata - proxy 计费需要
        # litellm_call_id - 日志追踪需要
        # litellm_logging_obj - 日志对象需要
        # proxy_server_request - proxy 身份识别需要
        # model_info - 模型信息需要
        # secret_fields - 密钥管理需要
        # shared_session - 会话共享需要

        # ========== tool_choice 兼容 ==========
        if args.get("tool_choice") == "required":
            args["tool_choice"] = "auto"
            _domestic_logger.info(
                "[DomesticFilter] tool_choice converted: required -> auto"
            )

        # ========== 确保 tool_calls arguments 是有效 JSON ==========
        messages = args.get("messages")
        if messages:
            args["messages"] = (
                LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
                    messages
                )
            )

        return args

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

        # 国内模型参数过滤（统一逻辑）
        LiteLLMCompletionTransformationHandler._filter_domestic_params(completion_args)

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

        # 国内模型参数过滤（统一逻辑）
        LiteLLMCompletionTransformationHandler._filter_domestic_params(acompletion_args)

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

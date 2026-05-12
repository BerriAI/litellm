"""
Handler for transforming responses api requests to litellm.completion requests
"""

import json
import logging
from typing import Any, Coroutine, Dict, List, Optional, Union

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
        import copy
        import logging

        logger = logging.getLogger("LiteLLM.DomesticFilter")

        fixed_messages = []
        fixed_count = 0
        for msg in messages:
            # Deep copy to avoid modifying original
            fixed_msg = copy.deepcopy(msg) if isinstance(msg, dict) else msg

            # Check for tool_calls in assistant messages
            if isinstance(fixed_msg, dict):
                role = fixed_msg.get("role")
                if role == "assistant":
                    tool_calls = fixed_msg.get("tool_calls")
                    if tool_calls and isinstance(tool_calls, list):
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                func = tc.get("function")
                                if isinstance(func, dict):
                                    args = func.get("arguments")
                                    original_args = args
                                    # Ensure arguments is valid JSON
                                    if args is None:
                                        func["arguments"] = "{}"
                                        fixed_count += 1
                                    elif isinstance(args, str):
                                        if not args.strip():
                                            func["arguments"] = "{}"
                                            fixed_count += 1
                                        else:
                                            try:
                                                json.loads(args)
                                            except (json.JSONDecodeError, ValueError):
                                                func["arguments"] = "{}"
                                                fixed_count += 1
                                                # 显示原始 arguments 的前50个字符（截断）
                                                args_preview = (
                                                    str(original_args)[:50]
                                                    if original_args
                                                    else "None"
                                                )
                                                logger.warning(
                                                    f"[DomesticFilter] Fixed invalid JSON arguments, original: {args_preview}"
                                                )
                                    elif isinstance(args, dict):
                                        func["arguments"] = json.dumps(args)
                                    else:
                                        # Other types, try to convert
                                        try:
                                            args_str = str(args)
                                            json.loads(args_str)
                                            func["arguments"] = args_str
                                        except (json.JSONDecodeError, ValueError):
                                            func["arguments"] = "{}"
                                            fixed_count += 1

            fixed_messages.append(fixed_msg)

        if fixed_count > 0:
            logger.info(
                f"[DomesticFilter] Fixed {fixed_count} invalid tool_calls arguments"
            )

        return fixed_messages

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
            # DEBUG logger（先定义，后面使用）
            logger = logging.getLogger("LiteLLM.DomesticFilter")

            # Codex CLI 特有参数
            completion_args.pop("client_metadata", None)
            # OpenAI 扩展参数（国内模型不支持）
            completion_args.pop("reasoning_effort", None)  # reasoning 模式
            completion_args.pop("reasoning", None)  # reasoning 参数 (Codex 0.130.0+)
            completion_args.pop("coding_plan", None)  # Codex coding plan
            completion_args.pop("parallel_tool_calls", None)  # 并行工具调用
            # 旧格式参数（国内模型只支持 tools/tool_choice，不支持 functions/function_call）
            completion_args.pop("functions", None)  # OpenAI 旧格式
            completion_args.pop("function_call", None)  # OpenAI 旧格式
            # 其他国内模型可能不支持的参数
            completion_args.pop("stream_options", None)  # stream 选项
            completion_args.pop("modalities", None)  # 多模态参数
            completion_args.pop("prediction", None)  # 预测参数
            completion_args.pop("audio", None)  # 音频参数
            completion_args.pop("store", None)  # 存储参数
            completion_args.pop("include", None)  # 包含参数
            completion_args.pop("prompt_cache_key", None)  # 缓存键
            completion_args.pop("text", None)  # text format 参数 (Codex 0.130.0+)
            # OpenAI 高级参数（国内模型可能不支持）
            completion_args.pop("frequency_penalty", None)  # 频率惩罚
            completion_args.pop("presence_penalty", None)  # 存在惩罚
            completion_args.pop("logprobs", None)  # logprobs
            completion_args.pop("top_logprobs", None)  # top_logprobs
            completion_args.pop("response_format", None)  # 响应格式（如 json_object）
            completion_args.pop("seed", None)  # 随机种子
            completion_args.pop("logit_bias", None)  # logit bias
            completion_args.pop("n", None)  # 返回多个结果
            completion_args.pop("service_tier", None)  # 服务等级
            # LiteLLM 内部参数（不应发送给 API，但 custom_llm_provider 必须保留）
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
            # 注意：custom_llm_provider 必须保留，LiteLLM 需要它识别提供商
            completion_args.pop("max_retries", None)
            # tool_choice 兼容：国内模型不支持 "required"，改成 "auto"
            tool_choice = completion_args.get("tool_choice")
            logger.info(f"[DomesticFilter] tool_choice before fix: {tool_choice}")
            if tool_choice == "required":
                completion_args["tool_choice"] = "auto"
                logger.info("[DomesticFilter] tool_choice converted: required -> auto")

            # DEBUG: 国内模型参数过滤验证
            logger.info(
                f"[DomesticFilter] actual_model={actual_model}, api_base={api_base}, "
                f"is_domestic={is_domestic_model_or_endpoint(actual_model, api_base)}"
            )
            logger.info(
                f"[DomesticFilter] completion_args keys after filter: "
                f"{list(completion_args.keys())}"
            )

            # 确保所有历史消息中的 tool_calls arguments 是有效 JSON 格式
            # 国内模型要求 function.arguments 必须是严格 JSON
            messages = completion_args.get("messages")
            if messages:
                completion_args["messages"] = (
                    LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
                        messages
                    )
                )
                logger.info(
                    "[DomesticFilter] Validated tool_calls arguments in messages"
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
        # 先获取 model 和 api_base，用于国内模型判断
        # 这些参数需要在 session handler 之前获取，确保 orphan 过滤逻辑正确
        model_name = litellm_completion_request.get("model", "")
        api_base = litellm_completion_request.get("api_base", "") or kwargs.get(
            "api_base", ""
        )
        # 从 litellm_params 提取实际模型名（如果是 openai/qwen3.6-plus 格式）
        litellm_params_model = kwargs.get("litellm_params", {}).get("model", "")
        if litellm_params_model and litellm_params_model.startswith("openai/"):
            model_name = litellm_params_model

        previous_response_id: Optional[str] = responses_api_request.get(
            "previous_response_id"
        )
        if previous_response_id:
            # 传入正确的 model_name 和 api_base 用于 orphan 过滤
            litellm_completion_request["model"] = model_name
            litellm_completion_request["api_base"] = api_base
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
            # DEBUG logger（先定义，后面使用）
            logger = logging.getLogger("LiteLLM.DomesticFilter")

            # Codex CLI 特有参数
            acompletion_args.pop("client_metadata", None)
            # OpenAI 扩展参数（国内模型不支持）
            acompletion_args.pop("reasoning_effort", None)  # reasoning 模式
            acompletion_args.pop("reasoning", None)  # reasoning 参数 (Codex 0.130.0+)
            acompletion_args.pop("coding_plan", None)  # Codex coding plan
            acompletion_args.pop("parallel_tool_calls", None)  # 并行工具调用
            # 旧格式参数（国内模型只支持 tools/tool_choice，不支持 functions/function_call）
            acompletion_args.pop("functions", None)  # OpenAI 旧格式
            acompletion_args.pop("function_call", None)  # OpenAI 旧格式
            # 其他国内模型可能不支持的参数
            acompletion_args.pop("stream_options", None)  # stream 选项
            acompletion_args.pop("modalities", None)  # 多模态参数
            acompletion_args.pop("prediction", None)  # 预测参数
            acompletion_args.pop("audio", None)  # 音频参数
            acompletion_args.pop("store", None)  # 存储参数
            acompletion_args.pop("include", None)  # 包含参数
            acompletion_args.pop("prompt_cache_key", None)  # 缓存键
            acompletion_args.pop("text", None)  # text format 参数 (Codex 0.130.0+)
            # OpenAI 高级参数（国内模型可能不支持）
            acompletion_args.pop("frequency_penalty", None)  # 频率惩罚
            acompletion_args.pop("presence_penalty", None)  # 存在惩罚
            acompletion_args.pop("logprobs", None)  # logprobs
            acompletion_args.pop("top_logprobs", None)  # top_logprobs
            acompletion_args.pop("response_format", None)  # 响应格式（如 json_object）
            acompletion_args.pop("seed", None)  # 随机种子
            acompletion_args.pop("logit_bias", None)  # logit bias
            acompletion_args.pop("n", None)  # 返回多个结果
            acompletion_args.pop("service_tier", None)  # 服务等级
            # LiteLLM 内部参数（不应发送给 API，但 custom_llm_provider 必须保留）
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
            # 注意：custom_llm_provider 必须保留，LiteLLM 需要它识别提供商
            acompletion_args.pop("max_retries", None)
            # tool_choice 兼容：国内模型不支持 "required"，改成 "auto"
            tool_choice = acompletion_args.get("tool_choice")
            logger.info(f"[DomesticFilter] tool_choice before fix: {tool_choice}")
            if tool_choice == "required":
                acompletion_args["tool_choice"] = "auto"
                logger.info("[DomesticFilter] tool_choice converted: required -> auto")

            # DEBUG: 国内模型参数过滤验证
            logger.info(
                f"[DomesticFilter] actual_model={actual_model}, api_base={api_base}, "
                f"is_domestic={is_domestic_model_or_endpoint(actual_model, api_base)}"
            )
            logger.info(
                f"[DomesticFilter] acompletion_args keys after filter: "
                f"{list(acompletion_args.keys())}"
            )

            # 确保所有历史消息中的 tool_calls arguments 是有效 JSON 格式
            # 国内模型要求 function.arguments 必须是严格 JSON
            messages = acompletion_args.get("messages")
            if messages:
                acompletion_args["messages"] = (
                    LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
                        messages
                    )
                )
                logger.info(
                    "[DomesticFilter] Validated tool_calls arguments in messages"
                )

                # 打印 tools 和 messages 用于调试
                tools_debug = acompletion_args.get("tools")
                messages_debug = acompletion_args.get("messages")
                logger.info(
                    f"[DomesticFilter] tools count: {len(tools_debug) if tools_debug else 0}"
                )
                if tools_debug:
                    for i, tool in enumerate(tools_debug[:3]):  # 只打印前3个
                        tool_name = (
                            tool.get("function", {}).get("name", "unknown")
                            if isinstance(tool, dict)
                            else "unknown"
                        )
                        logger.info(f"[DomesticFilter] tool[{i}] name: {tool_name}")
                logger.info(
                    f"[DomesticFilter] messages count: {len(messages_debug) if messages_debug else 0}"
                )
                if messages_debug:
                    for i, msg in enumerate(messages_debug[:5]):  # 只打印前5条
                        role = (
                            msg.get("role", "unknown")
                            if isinstance(msg, dict)
                            else "unknown"
                        )
                        has_tools = (
                            "tool_calls" in msg if isinstance(msg, dict) else False
                        )
                        logger.info(
                            f"[DomesticFilter] msg[{i}] role={role}, has_tool_calls={has_tools}"
                        )

                # 打印第一个有 tool_calls 的消息详情
                for i, msg in enumerate(messages_debug):
                    if isinstance(msg, dict) and msg.get("tool_calls"):
                        tc = msg.get("tool_calls", [])
                        if tc and isinstance(tc, list) and len(tc) > 0:
                            first_tc = tc[0]
                            func_name = (
                                first_tc.get("function", {}).get("name", "unknown")
                                if isinstance(first_tc, dict)
                                else "unknown"
                            )
                            args_preview = str(
                                first_tc.get("function", {}).get("arguments", "")
                            )[:100]
                            logger.info(
                                f"[DomesticFilter] First tool_call msg[{i}] function={func_name}, args_preview={args_preview}"
                            )
                            break

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

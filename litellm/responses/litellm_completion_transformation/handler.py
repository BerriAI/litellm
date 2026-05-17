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

        fixed_count = 0
        result_messages = []

        for msg in messages:
            # 获取 role
            if isinstance(msg, dict):
                role = msg.get("role")
                tool_calls = msg.get("tool_calls")
            else:
                role = getattr(msg, "role", None)
                tool_calls = getattr(msg, "tool_calls", None)

            # 只处理 assistant 消息的 tool_calls
            if (
                role != "assistant"
                or not tool_calls
                or not isinstance(tool_calls, list)
            ):
                result_messages.append(msg)
                continue

            # 需要重建 tool_calls，因为 Pydantic 模型属性可能不可修改
            needs_rebuild = False
            new_tool_calls = []

            for tc in tool_calls:
                # 获取 id, type, function
                if isinstance(tc, dict):
                    tc_id = tc.get("id")
                    tc_type = tc.get("type", "function")
                    func = tc.get("function")
                else:
                    tc_id = getattr(tc, "id", None)
                    tc_type = getattr(tc, "type", "function")
                    func = getattr(tc, "function", None)

                # 获取 function 的 name 和 arguments
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

                # 检查并修复 arguments
                original_args = args
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
                            # Valid JSON, keep as-is
                            fixed_args = args_stripped
                        except (json.JSONDecodeError, ValueError):
                            fixed_args = "{}"
                            needs_rebuild = True
                            fixed_count += 1
                            # 安全：不记录 original_args 内容，可能包含敏感信息
                elif isinstance(args, dict):
                    # dict 类型需要转成 JSON string
                    fixed_args = json.dumps(args)
                    needs_rebuild = True
                else:
                    # 其他类型，尝试转换
                    try:
                        args_str = str(args)
                        json.loads(args_str)
                        fixed_args = args_str
                    except (json.JSONDecodeError, ValueError):
                        fixed_args = "{}"
                        needs_rebuild = True
                        fixed_count += 1

                # 重建 tool_call dict（确保 arguments 是有效 JSON string）
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
                # 重建整个消息为 dict（确保所有字段被正确复制）
                if isinstance(msg, dict):
                    new_msg = copy.deepcopy(msg)
                    new_msg["tool_calls"] = new_tool_calls
                else:
                    # 对于非 dict 对象，创建一个新的 dict 消息
                    # 提取所有可能的字段
                    new_msg = {
                        "role": role,
                        "tool_calls": new_tool_calls,
                    }
                    # 复制其他可能存在的字段
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
                # 不需要修复，保留原消息
                result_messages.append(msg)

        if fixed_count > 0:
            logger.info(
                f"[DomesticFilter] Fixed {fixed_count} invalid tool_calls arguments, rebuilt {fixed_count} messages"
            )

        return result_messages

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

        # Responses API 专用参数清理（不应传递给 chat completion endpoint）
        # 这些参数在 transformation.py 中已转换为对应的 chat completion 参数
        # 例如：text → response_format, reasoning → reasoning_effort
        completion_args.pop("text", None)  # 已转换为 response_format
        completion_args.pop("reasoning", None)  # 已转换为 reasoning_effort
        completion_args.pop("instructions", None)  # Responses API 专用
        completion_args.pop("background", None)  # Responses API 专用
        completion_args.pop("truncation", None)  # Responses API 专用
        completion_args.pop("max_output_tokens", None)  # 已转换为 max_tokens

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
            # DEBUG logger
            logger = logging.getLogger("LiteLLM.DomesticFilter")

            # ========== 必须过滤的参数 ==========

            # Codex CLI 特有参数 - LiteLLM 内部报错
            completion_args.pop("client_metadata", None)

            # Responses API 专用参数（国内厂商 Chat API 不支持）
            completion_args.pop("coding_plan", None)  # Codex coding plan

            # 旧格式参数（国内模型只支持 tools/tool_choice）
            completion_args.pop("functions", None)
            completion_args.pop("function_call", None)

            # LiteLLM 内部参数（不应发送给上游）
            completion_args.pop("shared_session", None)
            completion_args.pop("model_info", None)
            completion_args.pop("secret_fields", None)
            completion_args.pop("use_in_pass_through", None)
            completion_args.pop("use_litellm_proxy", None)
            completion_args.pop("merge_reasoning_content_in_choices", None)
            completion_args.pop("supports_function_calling", None)
            completion_args.pop("max_retries", None)

            # ========== 厂商忽略的参数（保留过滤，避免无意义传输）==========
            # 这些参数厂商 Chat API 不支持，发送也不会报错但无意义
            completion_args.pop("stream_options", None)
            completion_args.pop("modalities", None)
            completion_args.pop("prediction", None)
            completion_args.pop("audio", None)
            completion_args.pop("store", None)
            completion_args.pop("include", None)
            completion_args.pop("prompt_cache_key", None)
            completion_args.pop("caching", None)
            completion_args.pop("extra_body", None)

            # ========== 不过滤的参数（厂商支持或忽略）==========
            # reasoning_effort - DeepSeek/Xiaomi 支持，其他厂商忽略
            # parallel_tool_calls - 厂商支持/忽略
            # frequency_penalty, presence_penalty - 厂商支持
            # logprobs, top_logprobs - 厂商支持/忽略
            # response_format - 厂商支持（json_object 等）
            # seed - 厂商支持/忽略
            # logit_bias, n, service_tier - 厂商支持/忽略

            # tool_choice 兼容：国内模型不支持 "required"，改成 "auto"
            tool_choice = completion_args.get("tool_choice")
            if tool_choice == "required":
                completion_args["tool_choice"] = "auto"
                logger.info("[DomesticFilter] tool_choice converted: required -> auto")

            # 确保 tool_calls arguments 是有效 JSON
            messages = completion_args.get("messages")
            if messages:
                completion_args["messages"] = (
                    LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
                        messages
                    )
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

        # Responses API 专用参数清理（不应传递给 chat completion endpoint）
        # 这些参数在 transformation.py 中已转换为对应的 chat completion 参数
        # 例如：text → response_format, reasoning → reasoning_effort
        acompletion_args.pop("text", None)  # 已转换为 response_format
        acompletion_args.pop("reasoning", None)  # 已转换为 reasoning_effort
        acompletion_args.pop("instructions", None)  # Responses API 专用
        acompletion_args.pop("background", None)  # Responses API 专用
        acompletion_args.pop("truncation", None)  # Responses API 专用
        acompletion_args.pop("max_output_tokens", None)  # 已转换为 max_tokens

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
            # DEBUG logger
            logger = logging.getLogger("LiteLLM.DomesticFilter")

            # ========== 必须过滤的参数 ==========

            # Codex CLI 特有参数 - LiteLLM 内部报错
            acompletion_args.pop("client_metadata", None)

            # Responses API 专用参数（国内厂商 Chat API 不支持）
            acompletion_args.pop("coding_plan", None)  # Codex coding plan

            # 旧格式参数（国内模型只支持 tools/tool_choice）
            acompletion_args.pop("functions", None)
            acompletion_args.pop("function_call", None)

            # LiteLLM 内部参数（不应发送给上游）
            acompletion_args.pop("shared_session", None)
            acompletion_args.pop("model_info", None)
            acompletion_args.pop("secret_fields", None)
            acompletion_args.pop("use_in_pass_through", None)
            acompletion_args.pop("use_litellm_proxy", None)
            acompletion_args.pop("merge_reasoning_content_in_choices", None)
            acompletion_args.pop("supports_function_calling", None)
            acompletion_args.pop("max_retries", None)

            # ========== 厂商忽略的参数（保留过滤，避免无意义传输）==========
            acompletion_args.pop("stream_options", None)
            acompletion_args.pop("modalities", None)
            acompletion_args.pop("prediction", None)
            acompletion_args.pop("audio", None)
            acompletion_args.pop("store", None)
            acompletion_args.pop("include", None)
            acompletion_args.pop("prompt_cache_key", None)
            acompletion_args.pop("caching", None)
            acompletion_args.pop("extra_body", None)

            # ========== 不过滤的参数（厂商支持或忽略）==========
            # reasoning_effort - DeepSeek/Xiaomi 支持，其他厂商忽略
            # parallel_tool_calls - 厂商支持/忽略
            # frequency_penalty, presence_penalty - 厂商支持
            # response_format, seed, logprobs 等 - 厂商支持/忽略

            # tool_choice 兼容：国内模型不支持 "required"，改成 "auto"
            tool_choice = acompletion_args.get("tool_choice")
            if tool_choice == "required":
                acompletion_args["tool_choice"] = "auto"
                logger.info("[DomesticFilter] tool_choice converted: required -> auto")

            # 确保 tool_calls arguments 是有效 JSON
            messages = acompletion_args.get("messages")
            if messages:
                acompletion_args["messages"] = (
                    LiteLLMCompletionTransformationHandler._ensure_all_tool_calls_have_valid_json_arguments(
                        messages
                    )
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

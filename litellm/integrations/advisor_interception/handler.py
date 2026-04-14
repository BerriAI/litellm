"""
Advisor interception for chat completions.

This mirrors the websearch interception pattern:
- Convert advisor tools to provider-compatible tool schema pre-request.
- Detect advisor tool calls in chat completion responses.
- Execute advisor sub-calls and continue the agentic loop server-side.
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import litellm
from litellm._logging import verbose_logger
from litellm.constants import ADVISOR_MAX_USES, ADVISOR_NATIVE_PROVIDERS
from litellm.integrations.advisor_interception.tools import (
    LITELLM_ADVISOR_TOOL_NAME,
    get_litellm_advisor_tool,
    get_litellm_advisor_tool_openai,
    is_advisor_tool,
    is_advisor_tool_chat_completion,
)
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import CallTypes, LlmProviders


class AdvisorInterceptionLogger(CustomLogger):
    """
    Intercept advisor tool calls in chat completions and orchestrate sub-calls.
    """

    def __init__(
        self,
        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None,
        default_advisor_model: str = "claude-opus-4-6",
    ):
        super().__init__()
        if enabled_providers is None:
            self.enabled_providers = None
        else:
            self.enabled_providers = [
                p.value if isinstance(p, LlmProviders) else p for p in enabled_providers
            ]
        self.default_advisor_model = default_advisor_model
        self._advisor_config_by_call_id: Dict[str, Dict[str, Any]] = {}
        self._skip_post_hook_call_ids: set[str] = set()

    async def async_pre_request_hook(
        self, model: str, messages: List[Dict], kwargs: Dict
    ) -> Optional[Dict]:
        """
        Convert advisor tools into provider-compatible form before request.
        """
        custom_llm_provider = kwargs.get("litellm_params", {}).get(
            "custom_llm_provider", ""
        )
        return self._convert_tools_for_provider(
            kwargs=kwargs, custom_llm_provider=custom_llm_provider
        )

    async def async_pre_call_deployment_hook(
        self, kwargs: Dict[str, Any], call_type: Optional[Any]
    ) -> Optional[dict]:
        """
        Pre-call hook used by completion/chat-completions paths.
        """
        if kwargs.pop("_advisor_interception_skip_post_hook", False):
            call_id = kwargs.get("litellm_call_id")
            if isinstance(call_id, str):
                self._skip_post_hook_call_ids.add(call_id)

        custom_llm_provider = kwargs.get("custom_llm_provider", "") or kwargs.get(
            "litellm_params", {}
        ).get("custom_llm_provider", "")
        if not custom_llm_provider:
            try:
                _, custom_llm_provider, _, _ = litellm.get_llm_provider(
                    model=kwargs.get("model", "")
                )
            except Exception:
                custom_llm_provider = ""
        return self._convert_tools_for_provider(
            kwargs=kwargs, custom_llm_provider=custom_llm_provider
        )

    async def async_post_call_success_deployment_hook(
        self,
        request_data: Dict[str, Any],
        response: Any,
        call_type: Optional[CallTypes],
    ) -> Optional[Any]:
        """
        Fallback advisor interception for providers that do not call
        async_should_run_chat_completion_agentic_loop internally.
        """
        if call_type not in {CallTypes.completion, CallTypes.acompletion}:
            return None

        call_id = request_data.get("litellm_call_id")
        if isinstance(call_id, str) and call_id in self._skip_post_hook_call_ids:
            self._skip_post_hook_call_ids.remove(call_id)
            return None

        model = request_data.get("model")
        messages = request_data.get("messages")
        if not isinstance(model, str) or not isinstance(messages, list):
            return None

        custom_llm_provider = request_data.get("custom_llm_provider", "") or request_data.get(
            "litellm_params", {}
        ).get("custom_llm_provider", "")
        if not custom_llm_provider:
            try:
                _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
            except Exception:
                custom_llm_provider = ""

        tools = request_data.get("tools")
        stream = bool(request_data.get("stream", False))

        should_run, tools_dict = await self.async_should_run_chat_completion_agentic_loop(
            response=response,
            model=model,
            messages=messages,
            tools=tools if isinstance(tools, list) else None,
            stream=stream,
            custom_llm_provider=custom_llm_provider,
            kwargs=request_data,
        )
        if not should_run:
            if isinstance(call_id, str):
                self._advisor_config_by_call_id.pop(call_id, None)
            return None

        optional_params = self._build_optional_params_from_request_data(request_data)
        return await self.async_run_chat_completion_agentic_loop(
            tools=tools_dict,
            model=model,
            messages=messages,
            response=response,
            optional_params=optional_params,
            logging_obj=request_data.get("litellm_logging_obj"),
            stream=stream,
            kwargs=request_data,
        )

    async def async_should_run_chat_completion_agentic_loop(
        self,
        response: Any,
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        stream: bool,
        custom_llm_provider: str,
        kwargs: Dict,
    ) -> Tuple[bool, Dict]:
        """
        Determine whether advisor agentic loop should run for chat completions.
        """
        if (
            self.enabled_providers is not None
            and custom_llm_provider not in self.enabled_providers
        ):
            return False, {}
        if custom_llm_provider in ADVISOR_NATIVE_PROVIDERS:
            return False, {}

        call_id = kwargs.get("litellm_call_id")
        has_advisor_config = isinstance(call_id, str) and (
            call_id in self._advisor_config_by_call_id
        )
        tools_list: List[Dict] = tools or []
        has_advisor_tool = has_advisor_config or (
            bool(tools_list)
            and any(is_advisor_tool_chat_completion(t) for t in tools_list)
        )
        if not has_advisor_tool:
            return False, {}

        advisor_calls, raw_tool_calls = self._extract_advisor_tool_calls(response)
        if not advisor_calls:
            if isinstance(call_id, str):
                self._advisor_config_by_call_id.pop(call_id, None)
            return False, {}

        # If there are mixed tool calls, do not hijack the request.
        if len(advisor_calls) != len(raw_tool_calls):
            verbose_logger.debug(
                "AdvisorInterception: Mixed tool calls detected, skipping advisor interception"
            )
            if isinstance(call_id, str):
                self._advisor_config_by_call_id.pop(call_id, None)
            return False, {}

        advisor_config = {}
        if isinstance(call_id, str):
            advisor_config = self._advisor_config_by_call_id.get(call_id, {})
        return True, {
            "advisor_calls": advisor_calls,
            "raw_tool_calls": raw_tool_calls,
            "advisor_config": advisor_config,
            "response_format": "openai",
        }

    async def async_run_chat_completion_agentic_loop(
        self,
        tools: Dict,
        model: str,
        messages: List[Dict],
        response: Any,
        optional_params: Dict,
        logging_obj: Any,
        stream: bool,
        kwargs: Dict,
    ) -> Any:
        """
        Execute advisor sub-calls and continue chat-completion loop.
        """
        advisor_config = tools.get("advisor_config", {}) or {}
        max_uses = int(advisor_config.get("max_uses", ADVISOR_MAX_USES))
        advisor_model = advisor_config.get("advisor_model") or self.default_advisor_model
        advisor_api_key = advisor_config.get("api_key")
        advisor_api_base = advisor_config.get("api_base")
        call_id = kwargs.get("litellm_call_id")

        current_messages: List[Dict] = list(messages)
        current_response = response
        advisor_uses = 0
        total_response_cost = self._safe_get_response_cost(current_response)

        try:
            while True:
                advisor_calls, raw_tool_calls = self._extract_advisor_tool_calls(
                    current_response
                )
                if not advisor_calls:
                    self._set_response_cost_if_possible(
                        response=current_response, response_cost=total_response_cost
                    )
                    return current_response
                if len(advisor_calls) != len(raw_tool_calls):
                    verbose_logger.debug(
                        "AdvisorInterception: Mixed tool calls detected inside advisor loop, stopping interception"
                    )
                    return current_response

                assistant_content = self._extract_message_content(current_response)
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "tool_calls": raw_tool_calls,
                }
                if assistant_content is not None:
                    assistant_message["content"] = assistant_content

                tool_messages: List[Dict[str, Any]] = []
                for advisor_call in advisor_calls:
                    advisor_uses += 1
                    if advisor_uses > max_uses:
                        raise ValueError(
                            "Advisor orchestration loop exceeded max_uses={}. "
                            "Increase max_uses in advisor tool config.".format(max_uses)
                        )

                    question = advisor_call.get(
                        "question", "Please provide guidance on the current task."
                    )
                    advisor_messages = self._build_advisor_context(
                        messages=current_messages,
                        assistant_content=assistant_content,
                        question=question,
                    )
                    advisor_response = await litellm.acompletion(
                        model=advisor_model,
                        messages=advisor_messages,
                        tools=None,
                        max_tokens=optional_params.get("max_tokens", 1024),
                        stream=False,
                        api_key=advisor_api_key,
                        api_base=advisor_api_base,
                        _advisor_interception_skip_post_hook=True,
                    )
                    total_response_cost += self._safe_get_response_cost(advisor_response)
                    advisor_text = self._extract_text_content(advisor_response)
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": advisor_call["id"],
                            "content": advisor_text,
                        }
                    )

                current_messages = current_messages + [assistant_message] + tool_messages

                optional_params_clean = {
                    k: v
                    for k, v in optional_params.items()
                    if k
                    not in {
                        "tools",
                        "extra_body",
                        "model_alias_map",
                        "stream_response",
                        "custom_prompt_dict",
                    }
                }
                kwargs_for_followup = self._prepare_followup_kwargs(kwargs)
                current_response = await litellm.acompletion(
                    model=self._get_full_model_name(model=model, kwargs=kwargs),
                    messages=current_messages,
                    tools=optional_params.get("tools"),
                    _advisor_interception_skip_post_hook=True,
                    **optional_params_clean,
                    **kwargs_for_followup,
                )
                total_response_cost += self._safe_get_response_cost(current_response)
        finally:
            if isinstance(call_id, str):
                self._advisor_config_by_call_id.pop(call_id, None)

    def _convert_tools_for_provider(
        self, kwargs: Dict[str, Any], custom_llm_provider: str
    ) -> Optional[Dict[str, Any]]:
        if (
            self.enabled_providers is not None
            and custom_llm_provider not in self.enabled_providers
        ):
            return None

        tools = kwargs.get("tools")
        if not tools:
            return None
        if not any(is_advisor_tool(t) for t in tools):
            return None

        advisor_cfg = self._extract_advisor_config(tools)
        advisor_model = advisor_cfg.get("advisor_model") or self.default_advisor_model
        if not advisor_model:
            raise ValueError(
                "Advisor tool requires a 'model'. Either pass native advisor tool "
                "with `model`, or set default_advisor_model on AdvisorInterceptionLogger."
            )
        max_uses = advisor_cfg.get("max_uses")
        if max_uses is None:
            max_uses = ADVISOR_MAX_USES
        api_key = advisor_cfg.get("api_key")
        api_base = advisor_cfg.get("api_base")

        converted_tools: List[Dict] = []
        if custom_llm_provider in ADVISOR_NATIVE_PROVIDERS:
            for tool in tools:
                if is_advisor_tool(tool):
                    converted_tools.append(
                        get_litellm_advisor_tool(
                            model=advisor_model,
                            max_uses=int(max_uses),
                            api_key=api_key,
                            api_base=api_base,
                        )
                    )
                else:
                    converted_tools.append(tool)
            kwargs["tools"] = converted_tools
            return kwargs

        for tool in tools:
            if is_advisor_tool(tool):
                converted_tools.append(get_litellm_advisor_tool_openai())
            else:
                converted_tools.append(tool)
        kwargs["tools"] = converted_tools
        call_id = kwargs.get("litellm_call_id")
        if isinstance(call_id, str):
            self._advisor_config_by_call_id[call_id] = {
            "advisor_model": advisor_model,
            "max_uses": int(max_uses),
            "api_key": api_key,
            "api_base": api_base,
            }
        if kwargs.get("stream"):
            kwargs["stream"] = False
            kwargs["_advisor_interception_converted_stream"] = True
        return kwargs

    def _extract_advisor_config(self, tools: List[Dict]) -> Dict[str, Any]:
        advisor_model = None
        max_uses = None
        api_key = None
        api_base = None

        for tool in tools:
            if not is_advisor_tool(tool):
                continue

            if tool.get("type") == "advisor_20260301":
                advisor_model = tool.get("model", advisor_model)
                max_uses = tool.get("max_uses", max_uses)
                api_key = tool.get("api_key", api_key)
                api_base = tool.get("api_base", api_base)

        return {
            "advisor_model": advisor_model,
            "max_uses": max_uses,
            "api_key": api_key,
            "api_base": api_base,
        }

    def _extract_advisor_tool_calls(self, response: Any) -> Tuple[List[Dict], List[Dict]]:
        message = self._extract_first_choice_message(response)
        if not message:
            return [], []

        tool_calls = message.get("tool_calls", [])
        function_call = message.get("function_call")
        advisor_calls: List[Dict] = []
        raw_tool_calls: List[Dict] = []

        for tool_call in tool_calls:
            raw_tool_calls.append(tool_call)
            function = tool_call.get("function", {})
            function_name = function.get("name")
            if function_name not in {"advisor", LITELLM_ADVISOR_TOOL_NAME}:
                continue

            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    parsed_args = json.loads(arguments)
                except json.JSONDecodeError:
                    parsed_args = {}
            elif isinstance(arguments, dict):
                parsed_args = arguments
            else:
                parsed_args = {}

            advisor_calls.append(
                {
                    "id": tool_call.get("id"),
                    "question": parsed_args.get("question"),
                }
            )

        # Some providers (e.g. Gemini in certain modes) can return legacy function_call.
        if not advisor_calls and function_call is not None:
            if isinstance(function_call, dict):
                function_name = function_call.get("name")
                function_arguments = function_call.get("arguments", {})
            else:
                function_name = getattr(function_call, "name", None)
                function_arguments = getattr(function_call, "arguments", {})
            if function_name in {"advisor", LITELLM_ADVISOR_TOOL_NAME}:
                if isinstance(function_arguments, str):
                    try:
                        parsed_args = json.loads(function_arguments)
                    except json.JSONDecodeError:
                        parsed_args = {}
                elif isinstance(function_arguments, dict):
                    parsed_args = function_arguments
                else:
                    parsed_args = {}

                synthetic_id = "call_litellm_advisor_0"
                advisor_calls.append(
                    {
                        "id": synthetic_id,
                        "question": parsed_args.get("question"),
                    }
                )
                raw_tool_calls.append(
                    {
                        "id": synthetic_id,
                        "type": "function",
                        "function": {
                            "name": function_name,
                            "arguments": json.dumps(parsed_args),
                        },
                    }
                )
        return advisor_calls, raw_tool_calls

    @staticmethod
    def _extract_first_choice_message(response: Any) -> Optional[Dict]:
        if isinstance(response, dict):
            choices = response.get("choices", [])
        else:
            choices = getattr(response, "choices", None) or []
        if not choices:
            return None

        first_choice = choices[0]
        if isinstance(first_choice, dict):
            message = first_choice.get("message")
        else:
            message = getattr(first_choice, "message", None)

        if message is None:
            return None
        if isinstance(message, dict):
            return message

        # pydantic object -> dict
        tool_calls = getattr(message, "tool_calls", None) or []
        normalized_tool_calls = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                normalized_tool_calls.append(tc)
            else:
                function = getattr(tc, "function", None)
                normalized_tool_calls.append(
                    {
                        "id": getattr(tc, "id", None),
                        "type": getattr(tc, "type", None),
                        "function": {
                            "name": getattr(function, "name", None) if function else None,
                            "arguments": getattr(function, "arguments", None)
                            if function
                            else None,
                        },
                    }
                )
        return {
            "role": getattr(message, "role", "assistant"),
            "content": getattr(message, "content", None),
            "tool_calls": normalized_tool_calls,
            "function_call": getattr(message, "function_call", None),
        }

    @staticmethod
    def _extract_message_content(response: Any) -> Optional[str]:
        message = AdvisorInterceptionLogger._extract_first_choice_message(response)
        if not message:
            return None
        return message.get("content")

    @staticmethod
    def _extract_text_content(response: Any) -> str:
        message = AdvisorInterceptionLogger._extract_first_choice_message(response)
        if not message:
            return ""
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _safe_get_response_cost(response: Any) -> float:
        if response is None:
            return 0.0

        if isinstance(response, dict):
            hidden_params = response.get("_hidden_params", {})
        else:
            hidden_params = getattr(response, "_hidden_params", {})

        if isinstance(hidden_params, dict):
            hidden_response_cost = hidden_params.get("response_cost")
            if isinstance(hidden_response_cost, (int, float)):
                return float(hidden_response_cost)

        try:
            response_cost = litellm.completion_cost(completion_response=response)
            if isinstance(response_cost, (int, float)):
                return float(response_cost)
        except Exception:
            return 0.0
        return 0.0

    @staticmethod
    def _set_response_cost_if_possible(response: Any, response_cost: float) -> None:
        if response is None:
            return
        if response_cost <= 0:
            return

        if isinstance(response, dict):
            hidden_params = response.setdefault("_hidden_params", {})
            if isinstance(hidden_params, dict):
                hidden_params["response_cost"] = response_cost
            return

        hidden_params = getattr(response, "_hidden_params", None)
        if isinstance(hidden_params, dict):
            hidden_params["response_cost"] = response_cost

    @staticmethod
    def _build_advisor_context(
        messages: List[Dict],
        assistant_content: Optional[str],
        question: str,
    ) -> List[Dict]:
        advisor_messages = list(messages)
        if assistant_content:
            advisor_messages.append({"role": "assistant", "content": assistant_content})
        advisor_messages.append({"role": "user", "content": question})
        return advisor_messages

    @staticmethod
    def _prepare_followup_kwargs(kwargs: Dict) -> Dict:
        internal_params = {
            "_advisor_interception",
            "_advisor_interception_converted_stream",
            "acompletion",
            "litellm_logging_obj",
            "custom_llm_provider",
            "model_alias_map",
            "stream_response",
            "custom_prompt_dict",
            "model",
            "messages",
            "tools",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "seed",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning_effort",
            "verbosity",
            "extra_headers",
            "api_version",
            "metadata",
            "web_search_options",
            "safety_identifier",
            "service_tier",
            "stream",
            "litellm_call_id",
        }
        return {
            k: v
            for k, v in kwargs.items()
            if not k.startswith("_advisor_interception") and k not in internal_params
        }

    @staticmethod
    def _get_full_model_name(model: str, kwargs: Dict) -> str:
        full_model_name = model
        custom_llm_provider = kwargs.get("custom_llm_provider")
        if custom_llm_provider and "/" not in model:
            full_model_name = "{}/{}".format(custom_llm_provider, model)
        return full_model_name

    @staticmethod
    def _build_optional_params_from_request_data(
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        allowed_keys = {
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "n",
            "stop",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "user",
            "response_format",
            "seed",
            "tools",
            "tool_choice",
            "parallel_tool_calls",
            "reasoning_effort",
            "verbosity",
            "extra_headers",
            "api_version",
            "metadata",
            "web_search_options",
            "safety_identifier",
            "service_tier",
        }
        return {k: v for k, v in request_data.items() if k in allowed_keys}

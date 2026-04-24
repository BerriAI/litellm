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
from litellm.types.integrations.advisor_interception import AdvisorInterceptionConfig
from litellm.types.utils import CallTypes, LlmProviders
from litellm.utils import (
    resolve_proxy_model_alias_to_litellm_model,
    supports_native_advisor_tool,
)


class AdvisorInterceptionLogger(CustomLogger):
    """
    Intercept advisor tool calls in chat completions and orchestrate sub-calls.
    """

    def __init__(
        self,
        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None,
        default_advisor_model: Optional[str] = None,
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
        self._converted_stream_call_ids: set[str] = set()

    @classmethod
    def from_config_yaml(
        cls, config: AdvisorInterceptionConfig
    ) -> "AdvisorInterceptionLogger":
        """
        Initialize AdvisorInterceptionLogger from proxy config.yaml parameters.

        Args:
            config: Configuration dictionary from litellm_settings.advisor_interception_params

        Example:
            From proxy_config.yaml:
                litellm_settings:
                  advisor_interception_params:
                    default_advisor_model: "advisor-model"
                    enabled_providers: ["openai", "vertex_ai"]
        """
        enabled_providers_str = config.get("enabled_providers", None)
        default_advisor_model = config.get("default_advisor_model", None)

        enabled_providers: Optional[List[Union[LlmProviders, str]]] = None
        if enabled_providers_str is not None:
            enabled_providers = []
            for provider in enabled_providers_str:
                try:
                    provider_enum = LlmProviders(provider)
                    enabled_providers.append(provider_enum)
                except ValueError:
                    enabled_providers.append(provider)

        return cls(
            enabled_providers=enabled_providers,
            default_advisor_model=default_advisor_model,
        )

    @staticmethod
    def initialize_from_proxy_config(
        litellm_settings: Dict[str, Any],
        callback_specific_params: Dict[str, Any],
    ) -> "AdvisorInterceptionLogger":
        """
        Static method to initialize AdvisorInterceptionLogger from proxy config.

        Used in callback_utils.py to simplify initialization logic.
        """
        advisor_params: AdvisorInterceptionConfig = {}
        if "advisor_interception_params" in litellm_settings:
            advisor_params = litellm_settings["advisor_interception_params"]
        elif "advisor_interception" in callback_specific_params:
            advisor_params = callback_specific_params["advisor_interception"]

        return AdvisorInterceptionLogger.from_config_yaml(advisor_params)

    async def async_pre_request_hook(
        self, model: str, messages: List[Dict], kwargs: Dict
    ) -> Optional[Dict]:
        """
        Convert advisor tools into provider-compatible form before request.

        Skips conversion for anthropic_messages call type because the Messages
        API path has its own AdvisorOrchestrationHandler interceptor.
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

        Skips conversion for anthropic_messages call type because the Messages
        API path has its own AdvisorOrchestrationHandler interceptor that
        expects the raw advisor_20260301 tool definition.
        """
        if call_type == CallTypes.anthropic_messages:
            return None

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
        converted_stream = (
            isinstance(call_id, str) and call_id in self._converted_stream_call_ids
        )
        if converted_stream and isinstance(call_id, str):
            self._converted_stream_call_ids.discard(call_id)

        if isinstance(call_id, str) and call_id in self._skip_post_hook_call_ids:
            self._skip_post_hook_call_ids.remove(call_id)
            if converted_stream:
                return self._wrap_as_streaming_if_needed(response)
            return None

        model = request_data.get("model")
        messages = request_data.get("messages")
        if not isinstance(model, str) or not isinstance(messages, list):
            if converted_stream:
                return self._wrap_as_streaming_if_needed(response)
            return None

        custom_llm_provider = request_data.get(
            "custom_llm_provider", ""
        ) or request_data.get("litellm_params", {}).get("custom_llm_provider", "")
        if not custom_llm_provider:
            try:
                _, custom_llm_provider, _, _ = litellm.get_llm_provider(model=model)
            except Exception:
                custom_llm_provider = ""

        tools = request_data.get("tools")
        stream = bool(request_data.get("stream", False))

        should_run, tools_dict = (
            await self.async_should_run_chat_completion_agentic_loop(
                response=response,
                model=model,
                messages=messages,
                tools=tools if isinstance(tools, list) else None,
                stream=stream,
                custom_llm_provider=custom_llm_provider,
                kwargs=request_data,
            )
        )
        if not should_run:
            if isinstance(call_id, str):
                self._advisor_config_by_call_id.pop(call_id, None)
            if converted_stream:
                return self._wrap_as_streaming_if_needed(response)
            return None

        optional_params = self._build_optional_params_from_request_data(request_data)
        result = await self.async_run_chat_completion_agentic_loop(
            tools=tools_dict,
            model=model,
            messages=messages,
            response=response,
            optional_params=optional_params,
            logging_obj=request_data.get("litellm_logging_obj"),
            stream=stream,
            kwargs=request_data,
        )
        if converted_stream:
            return self._wrap_as_streaming_if_needed(result)
        return result

    async def async_log_failure_event(
        self, kwargs, response_obj, start_time, end_time
    ) -> None:
        """
        Cleanup per-call advisor state on failure paths.
        """
        call_id = kwargs.get("litellm_call_id")
        if isinstance(call_id, str):
            self._advisor_config_by_call_id.pop(call_id, None)
            self._skip_post_hook_call_ids.discard(call_id)
            self._converted_stream_call_ids.discard(call_id)

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

        # Only skip the orchestration loop for native providers when the advisor
        # model is actually supported natively by the provider.
        # For Anthropic executors with a non-native advisor model, fall through
        # to the orchestration loop below.
        if custom_llm_provider in ADVISOR_NATIVE_PROVIDERS:
            call_id_check = kwargs.get("litellm_call_id")
            advisor_cfg = (
                self._advisor_config_by_call_id.get(call_id_check, {})
                if isinstance(call_id_check, str)
                else {}
            )
            advisor_model_check = (
                advisor_cfg.get("advisor_model") or self.default_advisor_model or ""
            )
            if self._is_native_anthropic_advisor_model(advisor_model_check):
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

    async def async_run_chat_completion_agentic_loop(  # noqa: PLR0915
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
        advisor_model = (
            advisor_config.get("advisor_model") or self.default_advisor_model
        )
        if not advisor_model:
            raise ValueError(
                "No advisor model configured. Either:\n"
                "  1. Set 'default_advisor_model' in advisor_interception_params in your proxy config YAML, or\n"
                "  2. Pass 'model' in the native advisor_20260301 tool definition.\n"
                "The advisor model should be a model_name from your model_list for correct credential resolution."
            )
        advisor_api_key = advisor_config.get("api_key")
        advisor_api_base = advisor_config.get("api_base")
        call_id = kwargs.get("litellm_call_id")

        llm_router = self._get_llm_router()

        current_messages: List[Dict] = list(messages)
        current_response = response
        advisor_uses = 0
        total_response_cost = self._safe_get_response_cost(current_response)
        advisor_subcall_cost: float = 0.0
        iterations: List[Dict[str, Any]] = [
            self._build_iteration_entry(
                response=current_response, iteration_type="message"
            )
        ]
        advisor_interactions: List[Dict[str, str]] = []

        try:
            while True:
                advisor_calls, raw_tool_calls = self._extract_advisor_tool_calls(
                    current_response
                )
                if not advisor_calls:
                    final_executor_cost = self._safe_get_response_cost(current_response)
                    advisor_first_call_cost = max(
                        total_response_cost
                        - advisor_subcall_cost
                        - final_executor_cost,
                        0.0,
                    )
                    self._set_response_cost_if_possible(
                        response=current_response, response_cost=total_response_cost
                    )
                    self._store_advisor_cost_breakdown(
                        logging_obj=kwargs.get("litellm_logging_obj") or logging_obj,
                        final_executor_cost=final_executor_cost,
                        advisor_first_call_cost=advisor_first_call_cost,
                        advisor_subcall_cost=advisor_subcall_cost,
                        total_response_cost=total_response_cost,
                    )
                    self._inject_advisor_results_into_response(
                        current_response, advisor_interactions
                    )
                    self._inject_advisor_iterations_into_response(
                        current_response, iterations
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
                    advisor_response = await self._call_advisor_model(
                        llm_router=llm_router,
                        advisor_model=advisor_model,
                        messages=advisor_messages,
                        max_tokens=optional_params.get("max_tokens", 1024),
                        api_key=advisor_api_key,
                        api_base=advisor_api_base,
                    )
                    advisor_call_cost = self._safe_get_response_cost(advisor_response)
                    total_response_cost += advisor_call_cost
                    advisor_subcall_cost += advisor_call_cost
                    iterations.append(
                        self._build_iteration_entry(
                            response=advisor_response,
                            iteration_type="advisor_message",
                            model=advisor_model,
                        )
                    )
                    advisor_text = self._extract_text_content(advisor_response)
                    advisor_interactions.append(
                        {
                            "tool_use_id": advisor_call["id"],
                            "advisor_text": advisor_text,
                        }
                    )
                    tool_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": advisor_call["id"],
                            "content": advisor_text,
                        }
                    )

                current_messages = (
                    current_messages + [assistant_message] + tool_messages
                )

                optional_params_clean = {
                    k: v
                    for k, v in optional_params.items()
                    if k
                    not in {
                        "tools",
                        "tool_choice",  # never force tool use on follow-up turns
                        "extra_body",
                        "model_alias_map",
                        "stream_response",
                        "custom_prompt_dict",
                    }
                }
                kwargs_for_followup = self._prepare_followup_kwargs(kwargs)
                executor_model = self._get_full_model_name(model=model, kwargs=kwargs)
                current_response = await self._call_executor_model(
                    llm_router=llm_router,
                    model=executor_model,
                    messages=current_messages,
                    tools=optional_params.get("tools"),
                    optional_params_clean=optional_params_clean,
                    kwargs_for_followup=kwargs_for_followup,
                )
                total_response_cost += self._safe_get_response_cost(current_response)
                iterations.append(
                    self._build_iteration_entry(
                        response=current_response, iteration_type="message"
                    )
                )
        finally:
            if isinstance(call_id, str):
                self._advisor_config_by_call_id.pop(call_id, None)
                self._skip_post_hook_call_ids.discard(call_id)
                self._converted_stream_call_ids.discard(call_id)

    @staticmethod
    def _wrap_as_streaming_if_needed(response: Any) -> Any:
        """
        Wrap a ModelResponse in a MockResponseIterator so the proxy can
        async-iterate it when the original request was stream=True but the
        advisor hook converted it to stream=False for the agentic loop.
        """
        from litellm.types.utils import ModelResponse as _ModelResponse

        if isinstance(response, _ModelResponse):
            from litellm.llms.base_llm.base_model_iterator import (
                MockResponseIterator,
            )

            return MockResponseIterator(response)
        return response

    @staticmethod
    def _get_llm_router() -> Optional[Any]:
        """Import the proxy router at runtime. Returns None in SDK-only usage."""
        try:
            from litellm.proxy.proxy_server import llm_router
        except ImportError:
            verbose_logger.debug(
                "AdvisorInterception: Could not import llm_router from proxy_server, "
                "falling back to direct litellm.acompletion()"
            )
            llm_router = None
        return llm_router

    @staticmethod
    def _is_native_anthropic_advisor_model(advisor_model: str) -> bool:
        """
        Return True when the advisor model supports Anthropic native advisor.

        Handles bare model names, litellm provider-prefixed names
        (e.g. ``anthropic/claude-opus-4-6``) and proxy model aliases.
        """
        resolved_model = resolve_proxy_model_alias_to_litellm_model(advisor_model)
        model_to_check = resolved_model or advisor_model
        return supports_native_advisor_tool(
            model=model_to_check, custom_llm_provider="anthropic"
        )

    @staticmethod
    async def _call_advisor_model(
        llm_router: Optional[Any],
        advisor_model: str,
        messages: List[Dict],
        max_tokens: int,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Any:
        """
        Call the advisor model, routing through the proxy router when available
        so that deployed credentials and load-balancing are used.
        """
        if llm_router is not None:
            try:
                return await llm_router.acompletion(
                    model=advisor_model,
                    messages=messages,
                    tools=None,
                    max_tokens=max_tokens,
                    stream=False,
                    _advisor_interception_skip_post_hook=True,
                )
            except Exception:
                verbose_logger.debug(
                    "AdvisorInterception: Router call for advisor model '%s' failed, "
                    "falling back to direct litellm.acompletion()",
                    advisor_model,
                )

        kwargs: Dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if api_base is not None:
            kwargs["api_base"] = api_base
        return await litellm.acompletion(
            model=advisor_model,
            messages=messages,
            tools=None,
            max_tokens=max_tokens,
            stream=False,
            _advisor_interception_skip_post_hook=True,
            **kwargs,
        )

    @staticmethod
    async def _call_executor_model(
        llm_router: Optional[Any],
        model: str,
        messages: List[Dict],
        tools: Optional[List[Dict]],
        optional_params_clean: Dict[str, Any],
        kwargs_for_followup: Dict[str, Any],
    ) -> Any:
        """
        Call the executor model for the follow-up turn, routing through the
        proxy router when available.
        """
        if llm_router is not None:
            try:
                return await llm_router.acompletion(
                    model=model,
                    messages=messages,
                    tools=tools,
                    _advisor_interception_skip_post_hook=True,
                    **optional_params_clean,
                    **kwargs_for_followup,
                )
            except Exception:
                verbose_logger.debug(
                    "AdvisorInterception: Router call for executor model '%s' failed, "
                    "falling back to direct litellm.acompletion()",
                    model,
                )

        return await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            _advisor_interception_skip_post_hook=True,
            **optional_params_clean,
            **kwargs_for_followup,
        )

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
                "No advisor model configured. Either:\n"
                "  1. Set 'default_advisor_model' in advisor_interception_params in your proxy config YAML, or\n"
                "  2. Pass 'model' in the native advisor_20260301 tool definition.\n"
                "The advisor model should be a model_name from your model_list for correct credential resolution."
            )
        max_uses = advisor_cfg.get("max_uses")
        if max_uses is None:
            max_uses = ADVISOR_MAX_USES
        api_key = advisor_cfg.get("api_key")
        api_base = advisor_cfg.get("api_base")

        converted_tools: List[Dict] = []
        use_native = (
            custom_llm_provider in ADVISOR_NATIVE_PROVIDERS
            and self._is_native_anthropic_advisor_model(advisor_model)
        )
        if use_native:
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
            call_id_for_stream = kwargs.get("litellm_call_id")
            if isinstance(call_id_for_stream, str):
                self._converted_stream_call_ids.add(call_id_for_stream)
            litellm_params = kwargs.get("litellm_params")
            if isinstance(litellm_params, dict):
                litellm_params["_advisor_interception_converted_stream"] = True
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

    def _extract_advisor_tool_calls(
        self, response: Any
    ) -> Tuple[List[Dict], List[Dict]]:
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
    def _inject_advisor_results_into_response(
        response: Any, advisor_interactions: List[Dict[str, str]]
    ) -> None:
        """
        Add ``advisor_tool_result`` blocks to ``provider_specific_fields``
        of the final chat-completion response message.

        This gives callers the same advisor visibility as the Anthropic
        native ``/v1/messages`` path.
        """
        if not advisor_interactions:
            return

        advisor_results: List[Dict] = []
        for interaction in advisor_interactions:
            tool_use_id = interaction["tool_use_id"]
            advisor_text = interaction["advisor_text"]
            advisor_results.append(
                {
                    "type": "server_tool_use",
                    "id": tool_use_id,
                    "name": "advisor",
                }
            )
            advisor_results.append(
                {
                    "type": "advisor_tool_result",
                    "tool_use_id": tool_use_id,
                    "content": {
                        "type": "advisor_result",
                        "text": advisor_text,
                    },
                }
            )

        message = AdvisorInterceptionLogger._extract_first_choice_message_obj(response)
        if message is None:
            return

        existing_psf = getattr(message, "provider_specific_fields", None) or {}
        existing_psf["advisor_tool_results"] = advisor_results
        try:
            message.provider_specific_fields = existing_psf
        except Exception:
            try:
                setattr(message, "provider_specific_fields", existing_psf)
            except Exception:
                pass

    @staticmethod
    def _extract_first_choice_message_obj(response: Any) -> Any:
        """Return the raw message object (not dict-normalised) from the first choice."""
        if isinstance(response, dict):
            choices = response.get("choices", [])
        else:
            choices = getattr(response, "choices", None) or []
        if not choices:
            return None
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            return first_choice.get("message")
        return getattr(first_choice, "message", None)

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
                            "name": (
                                getattr(function, "name", None) if function else None
                            ),
                            "arguments": (
                                getattr(function, "arguments", None)
                                if function
                                else None
                            ),
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
    def _build_iteration_entry(
        response: Any,
        iteration_type: str,
        model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build one entry for ``iterations[]`` on advisor-orchestrated responses.

        Mirrors the Anthropic usage shape: ``input_tokens``, ``output_tokens``,
        ``cache_read_input_tokens``, ``cache_creation_input_tokens``. For advisor
        sub-calls, also includes the advisor model name.
        """
        input_tokens = 0
        output_tokens = 0
        cache_read_input_tokens = 0
        cache_creation_input_tokens = 0

        usage: Any = None
        if isinstance(response, dict):
            usage = response.get("usage")
        else:
            usage = getattr(response, "usage", None)

        if usage is not None:
            input_tokens = (
                AdvisorInterceptionLogger._get_usage_value(usage, "prompt_tokens") or 0
            )
            output_tokens = (
                AdvisorInterceptionLogger._get_usage_value(usage, "completion_tokens")
                or 0
            )
            cache_read_input_tokens = (
                AdvisorInterceptionLogger._get_usage_value(
                    usage, "cache_read_input_tokens"
                )
                or 0
            )
            cache_creation_input_tokens = (
                AdvisorInterceptionLogger._get_usage_value(
                    usage, "cache_creation_input_tokens"
                )
                or 0
            )
            if not cache_read_input_tokens:
                prompt_tokens_details = AdvisorInterceptionLogger._get_usage_value(
                    usage, "prompt_tokens_details"
                )
                if prompt_tokens_details is not None:
                    cache_read_input_tokens = (
                        AdvisorInterceptionLogger._get_usage_value(
                            prompt_tokens_details, "cached_tokens"
                        )
                        or 0
                    )

        entry: Dict[str, Any] = {
            "type": iteration_type,
            "input_tokens": int(input_tokens or 0),
            "cache_read_input_tokens": int(cache_read_input_tokens or 0),
            "cache_creation_input_tokens": int(cache_creation_input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
        }
        if iteration_type == "advisor_message" and model is not None:
            entry["model"] = model
        return entry

    @staticmethod
    def _get_usage_value(usage_obj: Any, key: str) -> Any:
        if usage_obj is None:
            return None
        if isinstance(usage_obj, dict):
            return usage_obj.get(key)
        return getattr(usage_obj, key, None)

    @staticmethod
    def _store_advisor_cost_breakdown(
        logging_obj: Any,
        final_executor_cost: float,
        advisor_first_call_cost: float,
        advisor_subcall_cost: float,
        total_response_cost: float,
    ) -> None:
        """
        Populate ``cost_breakdown.additional_costs`` on the logging object so
        the proxy UI can display the advisor cost split (mirrors Azure Router).
        """
        if logging_obj is None:
            return
        if not hasattr(logging_obj, "set_cost_breakdown"):
            return

        additional_costs: Dict[str, float] = {}
        if advisor_first_call_cost > 0:
            additional_costs["Main Model (initial)"] = advisor_first_call_cost
        if advisor_subcall_cost > 0:
            additional_costs["Advisor Model"] = advisor_subcall_cost

        try:
            logging_obj.set_cost_breakdown(
                input_cost=final_executor_cost,
                output_cost=0.0,
                total_cost=total_response_cost,
                cost_for_built_in_tools_cost_usd_dollar=0.0,
                additional_costs=additional_costs or None,
            )
        except Exception as breakdown_error:
            verbose_logger.debug(
                "AdvisorInterception: failed to store cost breakdown: %s",
                str(breakdown_error),
            )

    @staticmethod
    def _inject_advisor_iterations_into_response(
        response: Any, iterations: List[Dict[str, Any]]
    ) -> None:
        """
        Attach the per-iteration usage breakdown under
        ``message.provider_specific_fields["advisor_iterations"]`` so callers
        can inspect the full orchestration without breaking the OpenAI-compatible
        ``usage`` shape.
        """
        if not iterations:
            return

        message = AdvisorInterceptionLogger._extract_first_choice_message_obj(response)
        if message is None:
            return

        existing_psf = getattr(message, "provider_specific_fields", None) or {}
        existing_psf["advisor_iterations"] = iterations
        try:
            message.provider_specific_fields = existing_psf
        except Exception:
            try:
                setattr(message, "provider_specific_fields", existing_psf)
            except Exception:
                pass

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

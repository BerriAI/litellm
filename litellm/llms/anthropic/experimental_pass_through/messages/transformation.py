from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

from litellm.constants import (
    DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET,
    DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.litellm_logging import verbose_logger
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic import (
    ANTHROPIC_ADVISOR_TOOL_TYPE,
    ANTHROPIC_BETA_HEADER_VALUES,
    AnthropicMessagesRequest,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.llms.anthropic_tool_search import get_tool_search_beta_header
from litellm.types.router import GenericLiteLLMParams

from ...common_utils import (
    AnthropicError,
    AnthropicModelInfo,
    optionally_handle_anthropic_oauth,
    strip_advisor_blocks_from_messages,
)

DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"

DROP_UNSUPPORTED_ADAPTIVE_EFFORT_WARNING = (
    "Dropping adaptive `thinking`/`output_config.effort` for model=%s: the model "
    "does not support extended thinking, or max_tokens is too small to fit the "
    "minimum thinking budget."
)


class AnthropicMessagesConfig(BaseAnthropicMessagesConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "anthropic"

    @property
    def _resolved_provider(self) -> str:
        return self.custom_llm_provider or "anthropic"

    def get_supported_anthropic_messages_params(self, model: str) -> list:
        return [
            "messages",
            "model",
            "system",
            "max_tokens",
            "stop_sequences",
            "temperature",
            "top_p",
            "top_k",
            "tools",
            "tool_choice",
            "thinking",
            "context_management",
            "output_format",
            "inference_geo",
            "speed",
            "output_config",
            "reasoning_effort",
            # TODO: Add Anthropic `metadata` support
            # "metadata",
        ]

    def _remove_scope_from_cache_control(self, anthropic_messages_request: Dict) -> None:
        """
        Remove `scope` field from cache_control blocks.

        Some providers (Vertex AI, Azure AI Foundry) do not support the `scope`
        field in cache_control (e.g. "global" for cross-request caching).
        Processes both `system` and `messages` content blocks.
        """

        def _sanitize(cache_control: Any) -> None:
            if isinstance(cache_control, dict):
                cache_control.pop("scope", None)

        def _process_content_list(content: list) -> None:
            for item in content:
                if isinstance(item, dict) and "cache_control" in item:
                    _sanitize(item["cache_control"])

        if "system" in anthropic_messages_request:
            system = anthropic_messages_request["system"]
            if isinstance(system, list):
                _process_content_list(system)

        if "messages" in anthropic_messages_request:
            for message in anthropic_messages_request["messages"]:
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if isinstance(content, list):
                        _process_content_list(content)

    def should_strip_billing_metadata(self) -> bool:
        """
        Whether to drop x-anthropic-billing-header system blocks before sending upstream.

        The first-party Anthropic API uses these blocks for Claude Code attribution, so the
        base config keeps them. Providers that reject them override this to True.
        """
        return False

    @staticmethod
    def _filter_billing_headers_from_system(system_param):
        """
        Filter out x-anthropic-billing-header metadata from system parameter.

        Args:
            system_param: Can be a string or a list of system message content blocks

        Returns:
            Filtered system parameter (string or list), or None if all content was filtered
        """
        if isinstance(system_param, str):
            # If it's a string and starts with billing header, filter it out
            if system_param.startswith("x-anthropic-billing-header:"):
                return None
            return system_param
        elif isinstance(system_param, list):
            # Filter list of system content blocks
            filtered_list = []
            for content_block in system_param:
                if isinstance(content_block, dict):
                    text = content_block.get("text", "")
                    content_type = content_block.get("type", "")
                    # Skip text blocks that start with billing header
                    if content_type == "text" and text.startswith("x-anthropic-billing-header:"):
                        continue
                    filtered_list.append(content_block)
                else:
                    # Keep non-dict items as-is
                    filtered_list.append(content_block)
            return filtered_list if len(filtered_list) > 0 else None
        else:
            return system_param

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        api_base = AnthropicModelInfo.get_api_base(api_base) or "https://api.anthropic.com"
        if not api_base.endswith("/v1/messages"):
            api_base = f"{api_base}/v1/messages"
        return api_base

    def validate_anthropic_messages_environment(
        self,
        headers: dict,
        model: str,
        messages: List[Any],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> Tuple[dict, Optional[str]]:
        # Check for Anthropic OAuth token in Authorization header
        headers, api_key = optionally_handle_anthropic_oauth(headers=headers, api_key=api_key)

        if api_base is None and isinstance(litellm_params, dict):
            api_base = litellm_params.get("api_base")
        use_bearer_for_custom_base = bool(
            isinstance(litellm_params, dict) and litellm_params.get("use_bearer_for_custom_base", False)
        )
        if "x-api-key" not in headers and "authorization" not in headers:
            auth_header = AnthropicModelInfo.get_auth_header(api_key, api_base, use_bearer_for_custom_base)
            if auth_header is not None:
                headers.update(auth_header)
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_API_VERSION
        if "content-type" not in headers:
            headers["content-type"] = "application/json"

        headers = self._update_headers_with_anthropic_beta(
            headers=headers,
            optional_params=optional_params,
        )

        return headers, api_base

    @staticmethod
    def _translate_reasoning_effort_to_anthropic(model: str, optional_params: Dict, custom_llm_provider: str) -> None:
        """Map OpenAI-style ``reasoning_effort`` to native Anthropic params.

        Caller-supplied ``thinking`` / ``output_config`` win over the alias.
        ``effort='none'`` clears both. Invalid efforts raise a 400.
        """
        from litellm.exceptions import BadRequestError as _BadRequestError
        from litellm.llms.anthropic.chat.transformation import (
            REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT,
            AnthropicConfig,
        )

        reasoning_effort = optional_params.pop("reasoning_effort", None)
        if not isinstance(reasoning_effort, str):
            return

        try:
            mapped_thinking = AnthropicConfig._map_reasoning_effort(
                reasoning_effort=reasoning_effort,
                model=model,
                custom_llm_provider=custom_llm_provider,
            )
        except _BadRequestError as e:
            raise AnthropicError(message=str(e.message), status_code=400)

        if mapped_thinking is None:
            optional_params.pop("thinking", None)
            optional_params.pop("output_config", None)
            return

        optional_params.setdefault("thinking", mapped_thinking)
        if AnthropicModelInfo._is_adaptive_thinking_model(model, custom_llm_provider):
            mapped_effort = REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT.get(reasoning_effort)
            if mapped_effort is None:
                raise AnthropicError(
                    message=(
                        f"Invalid reasoning_effort: {reasoning_effort!r}. "
                        f"Must be one of: 'minimal', 'low', 'medium', 'high', "
                        f"'xhigh', 'max', 'none'"
                    ),
                    status_code=400,
                )
            gate_error = AnthropicConfig._validate_effort_for_model(model, mapped_effort, custom_llm_provider)
            if gate_error is not None:
                raise AnthropicError(message=gate_error, status_code=400)
            existing_output_config = optional_params.get("output_config")
            if not isinstance(existing_output_config, dict):
                existing_output_config = {}
            existing_output_config.setdefault("effort", mapped_effort)
            optional_params["output_config"] = existing_output_config

    @staticmethod
    def _translate_legacy_thinking_for_adaptive_model(
        model: str, optional_params: Dict, custom_llm_provider: str
    ) -> None:
        """Translate legacy ``thinking.type=enabled`` to adaptive for 4.6/4.7.
        Caller-provided ``output_config.effort`` is never overridden.
        """
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        if not AnthropicModelInfo._is_adaptive_thinking_model(model, custom_llm_provider):
            return
        thinking = optional_params.get("thinking")
        if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
            return

        budget = int(thinking.get("budget_tokens") or 0)
        if budget >= DEFAULT_REASONING_EFFORT_XHIGH_THINKING_BUDGET and (
            AnthropicConfig._supports_effort_level(model, "xhigh", custom_llm_provider)
        ):
            effort = "xhigh"
        elif budget >= DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET:
            effort = "high"
        elif budget >= DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET:
            effort = "medium"
        else:
            effort = "low"

        optional_params["thinking"] = {"type": "adaptive"}
        existing_output_config = optional_params.get("output_config")
        if not isinstance(existing_output_config, dict):
            existing_output_config = {}
        existing_output_config.setdefault("effort", effort)
        optional_params["output_config"] = existing_output_config

    @staticmethod
    def _translate_adaptive_effort_for_non_adaptive_model(
        model: str, optional_params: Dict, max_tokens: Optional[int], custom_llm_provider: str
    ) -> None:
        """Translate the 4.6+ adaptive-thinking interface (``thinking.type=adaptive``
        and/or ``output_config.effort``) down to what an older Anthropic model
        supports. Clients like Claude Code send this interface unconditionally, so
        without translation it reaches a pre-4.6 model and Anthropic rejects it with
        "This model does not support the effort parameter".

        The reshape is silent, matching how the messages path already strips
        unsupported ``output_config`` for older models (bedrock invoke, issue
        #22797): the goal is to keep the request working, not to fail it.

        ``thinking.type=adaptive`` and ``output_config.effort`` are independent
        capabilities. Adaptive thinking needs ``supports_adaptive_thinking`` (4.6+);
        ``output_config.effort`` needs ``supports_output_config``, which some
        non-adaptive models (e.g. Claude Opus 4.5) advertise on its own. So the two
        are handled separately:

        - Adaptive-thinking models (4.6+): both are native, left untouched.
        - ``supports_output_config`` but non-adaptive (Opus 4.5): keep
          ``output_config.effort`` (native), only drop the unsupported adaptive
          ``thinking`` block. When adaptive thinking is being dropped and the
          effort level itself isn't supported by the model (e.g. ``xhigh``/``max``
          on Opus 4.5, which only accepts low/medium/high, while ``xhigh`` is
          Claude Code's default), fall through to the legacy translation below
          instead of forwarding a level Anthropic would reject. Effort-only
          requests are always left untouched: provider subclasses own their level
          normalization (bedrock clamps ``xhigh`` to the model's ceiling after
          this base transform runs).
        - Thinking-capable but neither (``supports_reasoning``, e.g. Haiku/Sonnet
          4.5): map effort to legacy ``thinking={type: enabled, budget_tokens}`` via
          ``AnthropicConfig._map_reasoning_effort``, capped below ``max_tokens``
          (Anthropic requires ``max_tokens > budget_tokens``) and dropped when
          ``max_tokens`` can't fit even the minimum budget.
        - No reasoning support: ``thinking`` is dropped.

        For the last two, only the consumed ``effort`` key is removed from
        ``output_config``; any residual (e.g. ``format``) is left for provider
        subclasses to handle.
        """
        from litellm.exceptions import BadRequestError as _BadRequestError
        from litellm.llms.anthropic.chat.transformation import AnthropicConfig

        if AnthropicConfig._is_adaptive_thinking_model(model, custom_llm_provider):
            return

        output_config = optional_params.get("output_config")
        thinking = optional_params.get("thinking")
        effort = output_config.get("effort") if isinstance(output_config, dict) else None
        adaptive_thinking = isinstance(thinking, dict) and thinking.get("type") == "adaptive"
        if effort is None and not adaptive_thinking:
            return

        # Models that natively accept `output_config.effort` but are not adaptive (Claude Opus 4.5).
        # Keep the native effort and only drop the adaptive `thinking` block, which these models
        # reject. Effort-only requests pass through so provider subclasses (bedrock/vertex) keep
        # owning level clamping; an adaptive request only stays here when its effort level is one
        # the model supports, otherwise it falls through to the legacy budget translation below.
        if AnthropicConfig._model_supports_effort_param(model, custom_llm_provider) and (
            not adaptive_thinking
            or AnthropicConfig._validate_effort_for_model(model, effort, custom_llm_provider) is None
        ):
            if adaptive_thinking:
                optional_params.pop("thinking", None)
            return

        supports_thinking = AnthropicModelInfo._supports_model_capability(
            model, "supports_reasoning", custom_llm_provider
        )
        try:
            legacy_thinking = (
                AnthropicConfig._map_reasoning_effort(
                    reasoning_effort=effort or "medium",
                    model=model,
                    custom_llm_provider=custom_llm_provider,
                )
                if supports_thinking
                else None
            )
        except _BadRequestError as e:
            raise AnthropicError(message=str(e.message), status_code=400)
        capped_thinking = (
            AnthropicConfig._cap_thinking_budget_to_max_tokens(legacy_thinking, max_tokens)
            if legacy_thinking is not None
            else None
        )

        if capped_thinking is not None:
            optional_params["thinking"] = capped_thinking
        else:
            verbose_logger.warning(DROP_UNSUPPORTED_ADAPTIVE_EFFORT_WARNING, model)
            optional_params.pop("thinking", None)

        if isinstance(output_config, dict) and "effort" in output_config:
            residual = {k: v for k, v in output_config.items() if k != "effort"}
            if residual:
                optional_params["output_config"] = residual
            else:
                optional_params.pop("output_config", None)

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        """
        No transformation is needed for Anthropic messages


        This takes in a request in the Anthropic /v1/messages API spec -> transforms it to /v1/messages API spec (i.e) no transformation is needed
        """
        max_tokens = anthropic_messages_optional_request_params.pop("max_tokens", None)
        if max_tokens is None:
            raise AnthropicError(
                message="max_tokens is required for Anthropic /v1/messages API",
                status_code=400,
            )

        self._translate_reasoning_effort_to_anthropic(
            model=model,
            optional_params=anthropic_messages_optional_request_params,
            custom_llm_provider=self._resolved_provider,
        )

        self._translate_legacy_thinking_for_adaptive_model(
            model=model,
            optional_params=anthropic_messages_optional_request_params,
            custom_llm_provider=self._resolved_provider,
        )

        self._translate_adaptive_effort_for_non_adaptive_model(
            model=model,
            optional_params=anthropic_messages_optional_request_params,
            max_tokens=max_tokens,
            custom_llm_provider=self._resolved_provider,
        )

        system_param = anthropic_messages_optional_request_params.get("system")
        if self.should_strip_billing_metadata() and system_param is not None:
            filtered_system = self._filter_billing_headers_from_system(system_param)
            if filtered_system is not None and len(filtered_system) > 0:
                anthropic_messages_optional_request_params["system"] = filtered_system
            else:
                anthropic_messages_optional_request_params.pop("system", None)

        # Transform context_management from OpenAI format to Anthropic format if needed
        context_management_param = anthropic_messages_optional_request_params.get("context_management")
        if context_management_param is not None:
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig

            transformed_context_management = AnthropicConfig.map_openai_context_management_to_anthropic(
                context_management_param
            )
            if transformed_context_management is not None:
                anthropic_messages_optional_request_params["context_management"] = transformed_context_management

        ####### get required params for all anthropic messages requests ######
        # Lazy %s: the f-string previously stringified the entire messages
        # payload on every request regardless of log level (a full scan of the
        # request body on the hot path). Defer it to when DEBUG is enabled.
        verbose_logger.debug("TRANSFORMATION DEBUG - Messages: %s", messages)

        # Auto-strip advisor blocks from history if advisor tool is absent.
        # Prevents Anthropic 400: advisor_tool_result in history requires advisor tool.
        _tools = anthropic_messages_optional_request_params.get("tools") or []
        _has_advisor = any(isinstance(t, dict) and t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE for t in _tools)
        if not _has_advisor:
            messages = strip_advisor_blocks_from_messages(messages)  # type: ignore[assignment]

        anthropic_messages_request: AnthropicMessagesRequest = AnthropicMessagesRequest(
            messages=messages,
            max_tokens=max_tokens,
            model=model,
            **anthropic_messages_optional_request_params,
        )
        return dict(anthropic_messages_request)

    def transform_anthropic_messages_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> AnthropicMessagesResponse:
        """
        No transformation is needed for Anthropic messages, since we want the response in the Anthropic /v1/messages API spec
        """
        try:
            raw_response_json = raw_response.json()
        except Exception:
            raise AnthropicError(message=raw_response.text, status_code=raw_response.status_code)
        return AnthropicMessagesResponse(**raw_response_json)

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        """Helper function to handle Anthropic streaming responses using the existing logging handlers"""
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            BaseAnthropicMessagesStreamingIterator,
        )

        # Use the shared streaming handler for Anthropic
        handler = BaseAnthropicMessagesStreamingIterator(
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )
        return handler.get_async_streaming_response_iterator(
            httpx_response=httpx_response,
            request_body=request_body,
            litellm_logging_obj=litellm_logging_obj,
        )

    @staticmethod
    def _update_headers_with_anthropic_beta(
        headers: dict,
        optional_params: dict,
        custom_llm_provider: str = "anthropic",
    ) -> dict:
        """
        Auto-inject anthropic-beta headers based on features used.

        Handles:
        - context_management: adds 'context-management-2025-06-27'
        - tool_search: adds provider-specific tool search header
        - output_format: adds 'structured-outputs-2025-11-13'
        - speed: adds 'fast-mode-2026-02-01'

        Args:
            headers: Request headers dict
            optional_params: Optional parameters including tools, context_management, output_format, speed
            custom_llm_provider: Provider name for looking up correct tool search header
        """
        beta_values: set = set()

        # Get existing beta headers if any
        existing_beta = headers.get("anthropic-beta")
        if existing_beta:
            beta_values.update(b.strip() for b in existing_beta.split(","))

        # Check for context management
        context_management_param = optional_params.get("context_management")
        if context_management_param is not None:
            # Check edits array for compact_20260112 type
            edits = context_management_param.get("edits", [])
            has_compact = False
            has_other = False

            for edit in edits:
                edit_type = edit.get("type", "")
                if edit_type == "compact_20260112":
                    has_compact = True
                else:
                    has_other = True

            # Add compact header if any compact edits exist
            if has_compact:
                beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.COMPACT_2026_01_12.value)

            # Add context management header if any other edits exist
            if has_other:
                beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value)

        # Check for structured outputs. Anthropic's newer request shape nests
        # the schema under output_config.format; the older top-level
        # output_format remains supported for backwards compatibility.
        output_config = optional_params.get("output_config")
        if optional_params.get("output_format") is not None or (
            isinstance(output_config, dict) and output_config.get("format") is not None
        ):
            beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.STRUCTURED_OUTPUT_2025_09_25.value)

        # Check for fast mode
        if optional_params.get("speed") == "fast":
            beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.FAST_MODE_2026_02_01.value)

        # Check for advisor tool
        tools = optional_params.get("tools")
        if tools:
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE:
                    beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.ADVISOR_TOOL_2026_03_01.value)
                    break

        # Check for tool search tools
        tools = optional_params.get("tools")
        if tools:
            anthropic_model_info = AnthropicModelInfo()
            if anthropic_model_info.is_tool_search_used(tools):
                # Use provider-specific tool search header
                tool_search_header = get_tool_search_beta_header(custom_llm_provider)
                beta_values.add(tool_search_header)

        if beta_values:
            headers["anthropic-beta"] = ",".join(sorted(beta_values))

        return headers

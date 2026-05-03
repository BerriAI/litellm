from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

import httpx

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


class AnthropicMessagesConfig(BaseAnthropicMessagesConfig):
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
            # OpenAI-style tier knob — translated to native ``thinking`` +
            # ``output_config`` in ``transform_anthropic_messages_request``
            # and popped before the request is forwarded.
            "reasoning_effort",
            # TODO: Add Anthropic `metadata` support
            # "metadata",
        ]

    def _remove_scope_from_cache_control(
        self, anthropic_messages_request: Dict
    ) -> None:
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
                    if content_type == "text" and text.startswith(
                        "x-anthropic-billing-header:"
                    ):
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
        api_base = (
            AnthropicModelInfo.get_api_base(api_base) or "https://api.anthropic.com"
        )
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
        headers, api_key = optionally_handle_anthropic_oauth(
            headers=headers, api_key=api_key
        )

        if "x-api-key" not in headers and "authorization" not in headers:
            auth_header = AnthropicModelInfo.get_auth_header(api_key)
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
    def _translate_reasoning_effort_to_anthropic(
        model: str, optional_params: Dict
    ) -> None:
        """Map OpenAI-style ``reasoning_effort`` to native Anthropic params.

        The /v1/messages spec doesn't include ``reasoning_effort`` — without
        this translation it gets silently dropped, leaving every adaptive
        tier collapsed to the same behavior on Bedrock Invoke /v1/messages
        (and on Anthropic / Azure AI / Vertex AI when callers pass it on
        the messages route). Mirrors ``AnthropicConfig.map_openai_params``
        on the chat completion path so the two routes can't drift.

        - Pops ``reasoning_effort`` from ``optional_params`` so it never
          reaches the wire.
        - Caller-supplied ``thinking`` / ``output_config`` always win — we
          don't override an explicit native value.
        - Effort=``none`` clears thinking + output_config so callers can
          opt out per request.
        - Invalid efforts raise ``BadRequestError`` (clean 400) instead of
          surfacing as 500s downstream.
        """
        from litellm.llms.anthropic.chat.transformation import (
            REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT,
            AnthropicConfig,
        )

        reasoning_effort = optional_params.pop("reasoning_effort", None)
        if not isinstance(reasoning_effort, str):
            return

        # ``_map_reasoning_effort`` raises ``BadRequestError`` (400) directly
        # on unmapped efforts. The /v1/messages pass-through surfaces errors
        # as ``AnthropicError``; convert here so callers see a provider-shaped
        # 400 rather than the LiteLLM-shaped one.
        from litellm.exceptions import BadRequestError as _BadRequestError

        try:
            mapped_thinking = AnthropicConfig._map_reasoning_effort(
                reasoning_effort=reasoning_effort, model=model
            )
        except _BadRequestError as e:
            raise AnthropicError(message=str(e.message), status_code=400)

        if mapped_thinking is None:
            optional_params.pop("thinking", None)
            optional_params.pop("output_config", None)
            return

        optional_params.setdefault("thinking", mapped_thinking)
        if AnthropicModelInfo._is_adaptive_thinking_model(model):
            mapped_effort = REASONING_EFFORT_TO_OUTPUT_CONFIG_EFFORT.get(
                reasoning_effort
            )
            # ``_map_reasoning_effort`` returns ``type=adaptive`` for any
            # string on adaptive models without checking the value. The
            # chat completion path validates the resolved effort downstream
            # via ``_apply_output_config``; /v1/messages has no equivalent
            # downstream check, so reject unmapped values here so callers
            # see a clean 400 instead of a 500 from the provider.
            if mapped_effort is None:
                raise AnthropicError(
                    message=(
                        f"Invalid reasoning_effort: {reasoning_effort!r}. "
                        f"Must be one of: 'minimal', 'low', 'medium', 'high', "
                        f"'xhigh', 'max', 'none'"
                    ),
                    status_code=400,
                )
            # Per-model gating: ``xhigh`` and ``max`` are only valid on
            # specific tiers (Opus 4.6/4.7 for max; data-driven for xhigh).
            # The chat completion path enforces this via
            # ``_apply_output_config``; mirror it here so /v1/messages
            # callers see a clean 400 instead of a provider-side error.
            # ``max`` is supported on Claude 4.6 (Opus + Sonnet) and Claude
            # 4.7 adaptive-thinking models. Prefer the data-driven
            # ``supports_max_reasoning_effort`` flag in
            # ``model_prices_and_context_window.json``; family-level checks
            # are a fallback for variants whose entries don't yet carry the
            # flag.
            if mapped_effort == "max" and not (
                AnthropicConfig._is_claude_4_6_model(model)
                or AnthropicConfig._is_claude_4_7_model(model)
                or AnthropicConfig._supports_effort_level(model, "max")
            ):
                raise AnthropicError(
                    message=(
                        f"effort='max' is not supported by this model. "
                        f"Got model: {model}"
                    ),
                    status_code=400,
                )
            if mapped_effort == "xhigh" and not AnthropicConfig._supports_effort_level(
                model, "xhigh"
            ):
                raise AnthropicError(
                    message=(
                        f"effort='xhigh' is not supported by this model. "
                        f"Got model: {model}"
                    ),
                    status_code=400,
                )
            existing_output_config = optional_params.get("output_config")
            if not isinstance(existing_output_config, dict):
                existing_output_config = {}
            existing_output_config.setdefault("effort", mapped_effort)
            optional_params["output_config"] = existing_output_config

    @staticmethod
    def _translate_legacy_thinking_for_adaptive_model(
        model: str, optional_params: Dict
    ) -> None:
        """Translate legacy ``thinking.type=enabled`` to adaptive for 4.6/4.7.
        Caller-provided ``output_config.effort`` is never overridden.
        """
        if not AnthropicModelInfo._is_adaptive_thinking_model(model):
            return
        thinking = optional_params.get("thinking")
        if not isinstance(thinking, dict) or thinking.get("type") != "enabled":
            return

        budget = int(thinking.get("budget_tokens") or 0)
        if budget >= 24000:
            effort = "xhigh"
        elif budget >= 10000:
            effort = "high"
        elif budget >= 5000:
            effort = "medium"
        else:
            effort = "low"

        optional_params["thinking"] = {"type": "adaptive"}
        existing_output_config = optional_params.get("output_config")
        if not isinstance(existing_output_config, dict):
            existing_output_config = {}
        existing_output_config.setdefault("effort", effort)
        optional_params["output_config"] = existing_output_config

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
        )

        self._translate_legacy_thinking_for_adaptive_model(
            model=model,
            optional_params=anthropic_messages_optional_request_params,
        )

        # Filter out x-anthropic-billing-header from system messages
        system_param = anthropic_messages_optional_request_params.get("system")
        if system_param is not None:
            filtered_system = self._filter_billing_headers_from_system(system_param)
            if filtered_system is not None and len(filtered_system) > 0:
                anthropic_messages_optional_request_params["system"] = filtered_system
            else:
                # Remove system parameter if all content was filtered out
                anthropic_messages_optional_request_params.pop("system", None)

        # Transform context_management from OpenAI format to Anthropic format if needed
        context_management_param = anthropic_messages_optional_request_params.get(
            "context_management"
        )
        if context_management_param is not None:
            from litellm.llms.anthropic.chat.transformation import AnthropicConfig

            transformed_context_management = (
                AnthropicConfig.map_openai_context_management_to_anthropic(
                    context_management_param
                )
            )
            if transformed_context_management is not None:
                anthropic_messages_optional_request_params["context_management"] = (
                    transformed_context_management
                )

        ####### get required params for all anthropic messages requests ######
        verbose_logger.debug(f"TRANSFORMATION DEBUG - Messages: {messages}")

        # Auto-strip advisor blocks from history if advisor tool is absent.
        # Prevents Anthropic 400: advisor_tool_result in history requires advisor tool.
        _tools = anthropic_messages_optional_request_params.get("tools") or []
        _has_advisor = any(
            isinstance(t, dict) and t.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE
            for t in _tools
        )
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
            raise AnthropicError(
                message=raw_response.text, status_code=raw_response.status_code
            )
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
                beta_values.add(
                    ANTHROPIC_BETA_HEADER_VALUES.CONTEXT_MANAGEMENT_2025_06_27.value
                )

        # Check for structured outputs
        if optional_params.get("output_format") is not None:
            beta_values.add(
                ANTHROPIC_BETA_HEADER_VALUES.STRUCTURED_OUTPUT_2025_09_25.value
            )

        # Check for fast mode
        if optional_params.get("speed") == "fast":
            beta_values.add(ANTHROPIC_BETA_HEADER_VALUES.FAST_MODE_2026_02_01.value)

        # Check for advisor tool
        tools = optional_params.get("tools")
        if tools:
            for tool in tools:
                if (
                    isinstance(tool, dict)
                    and tool.get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE
                ):
                    beta_values.add(
                        ANTHROPIC_BETA_HEADER_VALUES.ADVISOR_TOOL_2026_03_01.value
                    )
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

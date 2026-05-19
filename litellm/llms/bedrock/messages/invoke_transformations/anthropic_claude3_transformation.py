from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import httpx

import litellm
from litellm.anthropic_beta_headers_manager import filter_and_transform_beta_headers
from litellm.constants import BEDROCK_MIN_THINKING_BUDGET_TOKENS
from litellm.litellm_core_utils.litellm_logging import verbose_logger
from litellm.llms.anthropic.chat.transformation import (
    DROP_UNSUPPORTED_OUTPUT_CONFIG_WARNING,
    AnthropicConfig,
)
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.llms.bedrock.chat.invoke_handler import AWSEventStreamDecoder
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
)
from litellm.llms.bedrock.common_utils import (
    ensure_bedrock_anthropic_messages_tool_names,
    get_anthropic_beta_from_headers,
    is_claude_4_5_on_bedrock,
    normalize_tool_input_schema_types_for_bedrock_invoke,
    remove_custom_field_from_tools,
)
from litellm.types.llms.anthropic import ANTHROPIC_TOOL_SEARCH_BETA_HEADER
from litellm.types.llms.bedrock import BedrockInvokeAnthropicMessagesRequest
from litellm.types.llms.openai import AllMessageValues
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import GenericStreamingChunk
from litellm.types.utils import GenericStreamingChunk as GChunk
from litellm.types.utils import ModelResponseStream

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class AmazonAnthropicClaudeMessagesConfig(
    AnthropicMessagesConfig,
    AmazonInvokeConfig,
):
    """
    Call Claude model family in the /v1/messages API spec
    Supports anthropic_beta parameter for beta features.
    """

    DEFAULT_BEDROCK_ANTHROPIC_API_VERSION = "bedrock-2023-05-31"

    BEDROCK_INVOKE_ALLOWED_TOP_LEVEL_FIELDS = frozenset(
        BedrockInvokeAnthropicMessagesRequest.__annotations__.keys()
    )

    def __init__(self, **kwargs):
        BaseAnthropicMessagesConfig.__init__(self, **kwargs)
        AmazonInvokeConfig.__init__(self, **kwargs)

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
        return headers, api_base

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
        return AmazonInvokeConfig.sign_request(
            self=self,
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        return AmazonInvokeConfig.get_complete_url(
            self=self,
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params,
            stream=stream,
        )

    def _remove_ttl_from_cache_control(
        self, anthropic_messages_request: Dict, model: Optional[str] = None
    ) -> None:
        """
        Remove unsupported fields from cache_control for Bedrock.

        Bedrock only supports `type` and `ttl` in cache_control. It does NOT support:
        - `scope` (e.g., "global") - always removed
        - `ttl` - removed for older models; Claude 4.5+ supports "5m" and "1h"

        Processes `tools`, `system`, and `messages` content blocks.

        Args:
            anthropic_messages_request: The request dictionary to modify in-place
            model: The model name to check if it supports ttl
        """
        is_claude_4_5 = False
        if model:
            is_claude_4_5 = self._is_claude_4_5_on_bedrock(model)

        def _sanitize_cache_control(cache_control: dict) -> None:
            if not isinstance(cache_control, dict):
                return
            # Bedrock doesn't support scope (e.g., "global" for cross-request caching)
            cache_control.pop("scope", None)
            # Remove ttl for models that don't support it
            if "ttl" in cache_control:
                ttl = cache_control["ttl"]
                if is_claude_4_5 and ttl in ["5m", "1h"]:
                    return
                cache_control.pop("ttl", None)

        def _process_content_list(content: list) -> None:
            for item in content:
                if isinstance(item, dict) and "cache_control" in item:
                    _sanitize_cache_control(item["cache_control"])

        # Process tools
        if "tools" in anthropic_messages_request:
            for tool in anthropic_messages_request["tools"]:
                if isinstance(tool, dict) and "cache_control" in tool:
                    _sanitize_cache_control(tool["cache_control"])

        # Process system (list of content blocks)
        if "system" in anthropic_messages_request:
            system = anthropic_messages_request["system"]
            if isinstance(system, list):
                _process_content_list(system)

        # Process messages
        if "messages" in anthropic_messages_request:
            for message in anthropic_messages_request["messages"]:
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if isinstance(content, list):
                        _process_content_list(content)

    def _supports_extended_thinking_on_bedrock(self, model: str) -> bool:
        """
        Check if the model supports extended thinking beta headers on Bedrock.

        On 3rd-party platforms (e.g., Amazon Bedrock), extended thinking is only
        supported on: Claude Opus 4.5, Claude Opus 4.1, Opus 4, or Sonnet 4.

        Ref: https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking

        Args:
            model: The model name

        Returns:
            True if the model supports extended thinking on Bedrock
        """
        model_lower = model.lower()

        # Supported models on Bedrock for extended thinking
        supported_patterns = [
            "opus-4.5",
            "opus_4.5",
            "opus-4-5",
            "opus_4_5",  # Opus 4.5
            "opus-4.1",
            "opus_4.1",
            "opus-4-1",
            "opus_4_1",  # Opus 4.1
            "opus-4",
            "opus_4",  # Opus 4
            "sonnet-4",
            "sonnet_4",  # Sonnet 4
            "sonnet-4.6",
            "sonnet_4.6",
            "sonnet-4-6",
            "sonnet_4_6",
            "opus-4.6",
            "opus_4.6",
            "opus-4-6",
            "opus_4_6",
            "opus-4.7",
            "opus_4.7",
            "opus-4-7",
            "opus_4_7",
        ]

        return any(pattern in model_lower for pattern in supported_patterns)

    def _ensure_thinking_for_clear_thinking_context_management(
        self,
        anthropic_messages_request: Dict,
        model: str,
    ) -> bool:
        """
        Bedrock rejects ``clear_thinking_20251015`` context-management edits unless
        extended thinking is ``enabled`` or ``adaptive``. Claude Code often sends
        context management without a top-level ``thinking`` field.

        When we detect that edit type on a model that supports extended thinking on
        Bedrock, inject a minimal ``thinking`` config so the request succeeds.

        Returns:
            True if ``thinking`` was added or upgraded for this fix (caller may
            need to add the interleaved-thinking beta header).
        """
        cm = anthropic_messages_request.get("context_management")
        if not isinstance(cm, dict):
            return False
        edits = cm.get("edits")
        if not isinstance(edits, list):
            return False
        needs_thinking = any(
            isinstance(e, dict) and e.get("type") == "clear_thinking_20251015"
            for e in edits
        )
        if not needs_thinking:
            return False
        if not self._supports_extended_thinking_on_bedrock(model):
            return False

        thinking = anthropic_messages_request.get("thinking")
        if isinstance(thinking, dict):
            t = thinking.get("type")
            if t in ("enabled", "adaptive"):
                return False
            # ``disabled`` or unknown — replace with enabled so clear_thinking is valid
            verbose_logger.debug(
                "Bedrock clear_thinking_20251015: replacing thinking=%s with minimal enabled thinking",
                thinking,
            )

        max_tokens = anthropic_messages_request.get("max_tokens")
        budget = BEDROCK_MIN_THINKING_BUDGET_TOKENS
        if isinstance(max_tokens, int) and max_tokens <= budget:
            verbose_logger.warning(
                "Bedrock clear_thinking_20251015: max_tokens=%s is not greater than "
                "minimum thinking budget (%s); cannot inject thinking safely",
                max_tokens,
                budget,
            )
            return False

        anthropic_messages_request["thinking"] = {
            "type": "enabled",
            "budget_tokens": budget,
        }
        verbose_logger.debug(
            "Bedrock clear_thinking_20251015: injected thinking with budget_tokens=%s",
            budget,
        )
        return True

    def _is_claude_opus_4_5(self, model: str) -> bool:
        """
        Check if the model is Claude Opus 4.5.

        Args:
            model: The model name

        Returns:
            True if the model is Claude Opus 4.5
        """
        model_lower = model.lower()
        opus_4_5_patterns = [
            "opus-4.5",
            "opus_4.5",
            "opus-4-5",
            "opus_4_5",
        ]
        return any(pattern in model_lower for pattern in opus_4_5_patterns)

    def _is_claude_4_5_on_bedrock(self, model: str) -> bool:
        """
        Check if the model is Claude 4.5 on Bedrock.

        Claude Sonnet 4.5, Haiku 4.5, and Opus 4.5 support 1-hour prompt caching.

        Args:
            model: The model name

        Returns:
            True if the model is Claude 4.5
        """
        return is_claude_4_5_on_bedrock(model)

    def _supports_tool_search_on_bedrock(self, model: str) -> bool:
        """
        Check if the model supports tool search on Bedrock.

        On Amazon Bedrock, server-side tool search is supported on Claude Opus 4.5
        and Claude Sonnet 4.5 with the tool-search-tool-2025-10-19 beta header.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool

        Args:
            model: The model name

        Returns:
            True if the model supports tool search on Bedrock
        """
        model_lower = model.lower()

        # Supported models for tool search on Bedrock
        supported_patterns = [
            # Opus 4.5
            "opus-4.5",
            "opus_4.5",
            "opus-4-5",
            "opus_4_5",
            # Sonnet 4.5
            "sonnet-4.5",
            "sonnet_4.5",
            "sonnet-4-5",
            "sonnet_4_5",
            # Opus 4.6
            "opus-4.6",
            "opus_4.6",
            "opus-4-6",
            "opus_4_6",
            # sonnet 4.6
            "sonnet-4.6",
            "sonnet_4.6",
            "sonnet-4-6",
            "sonnet_4_6",
            # NOTE: Opus 4.7 on Bedrock does not support server-side tool search
            # as of launch (2026-04-16). Bedrock rejects the tool type with:
            # "tool type 'tool_search_tool_..._20251119' is not supported for this model".
            # Re-add the opus-4.7 patterns here once AWS announces support.
        ]

        return any(pattern in model_lower for pattern in supported_patterns)

    def _get_tool_search_beta_header_for_bedrock(
        self,
        model: str,
        tool_search_used: bool,
        programmatic_tool_calling_used: bool,
        input_examples_used: bool,
        beta_set: set,
    ) -> None:
        """
        Adjust tool search beta header for Bedrock.

        Bedrock requires a different beta header for tool search on Opus 4 models
        when tool search is used without programmatic tool calling or input examples.

        Note: On Amazon Bedrock, server-side tool search is only supported on Claude Opus 4
        with the `tool-search-tool-2025-10-19` beta header.

        Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool

        Args:
            model: The model name
            tool_search_used: Whether tool search is used
            programmatic_tool_calling_used: Whether programmatic tool calling is used
            input_examples_used: Whether input examples are used
            beta_set: The set of beta headers to modify in-place
        """
        if tool_search_used and not (
            programmatic_tool_calling_used or input_examples_used
        ):
            beta_set.discard(ANTHROPIC_TOOL_SEARCH_BETA_HEADER)
            if self._supports_tool_search_on_bedrock(model):
                beta_set.add("tool-search-tool-2025-10-19")

    @staticmethod
    def _filter_context_management_for_bedrock_invoke(
        anthropic_messages_request: Dict,
        beta_set: set,
    ) -> None:
        """
        Bedrock InvokeModel accepts ``context_management`` only when it carries
        ``compact_20260112`` edits paired with the ``compact-2026-01-12``
        anthropic-beta header. Other edit types (notably ``clear_thinking_20251015``,
        which Claude Code sends on every request) are LiteLLM-internal and would
        cause Bedrock to 400 with ``"context_management: Extra inputs are not
        permitted"``.

        Filter the edits list to the supported subset, add the beta header when
        compact edits remain, and drop ``context_management`` entirely when no
        supported edits are left so the safety-net allowlist can pass it through.

        Ref: https://github.com/BerriAI/litellm/issues/27532
        """
        cm = anthropic_messages_request.get("context_management")
        if not isinstance(cm, dict):
            return
        edits = cm.get("edits")
        if not isinstance(edits, list):
            anthropic_messages_request.pop("context_management", None)
            return

        compact_edits = [
            e
            for e in edits
            if isinstance(e, dict) and e.get("type") == "compact_20260112"
        ]
        if compact_edits:
            beta_set.add("compact-2026-01-12")
            anthropic_messages_request["context_management"] = {
                **cm,
                "edits": compact_edits,
            }
        else:
            anthropic_messages_request.pop("context_management", None)

    def _convert_output_format_to_inline_schema(
        self,
        output_format: Dict,
        anthropic_messages_request: Dict,
    ) -> None:
        """
        Convert Anthropic output_format to inline schema in message content.

        Bedrock Invoke doesn't support the output_format parameter, so we embed
        the schema directly into the user message content as text instructions.

        This approach adds the schema to the last user message, instructing the model
        to respond in the specified JSON format.

        Args:
            output_format: The output_format dict with 'type' and 'schema'
            anthropic_messages_request: The request dict to modify in-place

        Ref: https://aws.amazon.com/blogs/machine-learning/structured-data-response-with-amazon-bedrock-prompt-engineering-and-tool-use/
        """
        import json

        # Extract schema from output_format
        schema = output_format.get("schema")
        if not schema:
            return

        # Get messages from the request
        messages = anthropic_messages_request.get("messages", [])
        if not messages:
            return

        # Find the last user message
        last_user_message_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if messages[idx].get("role") == "user":
                last_user_message_idx = idx
                break

        if last_user_message_idx is None:
            return

        last_user_message = messages[last_user_message_idx]
        content = last_user_message.get("content", [])

        # Ensure content is a list
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
            last_user_message["content"] = content

        # Add schema as text content to the message
        schema_text = {"type": "text", "text": json.dumps(schema)}
        content.append(schema_text)

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: List[Dict],
        anthropic_messages_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Dict:
        anthropic_messages_request = AnthropicMessagesConfig.transform_anthropic_messages_request(
            self=self,
            model=model,
            messages=messages,
            anthropic_messages_optional_request_params=anthropic_messages_optional_request_params,
            litellm_params=litellm_params,
            headers=headers,
        )
        #########################################################
        ############## BEDROCK Invoke SPECIFIC TRANSFORMATION ###
        #########################################################

        # 1. anthropic_version is required for all claude models
        if "anthropic_version" not in anthropic_messages_request:
            anthropic_messages_request["anthropic_version"] = (
                self.DEFAULT_BEDROCK_ANTHROPIC_API_VERSION
            )

        # 2. `stream` is not allowed in request body for bedrock invoke
        if "stream" in anthropic_messages_request:
            anthropic_messages_request.pop("stream", None)

        # 3. `model` is not allowed in request body for bedrock invoke
        if "model" in anthropic_messages_request:
            anthropic_messages_request.pop("model", None)

        injected_thinking_for_clear_thinking = (
            self._ensure_thinking_for_clear_thinking_context_management(
                anthropic_messages_request=anthropic_messages_request,
                model=model,
            )
        )

        # 4. Remove `ttl` field from cache_control in messages (Bedrock doesn't support it for older models)
        self._remove_ttl_from_cache_control(
            anthropic_messages_request=anthropic_messages_request, model=model
        )

        # 5. Convert `output_format` to inline schema (Bedrock invoke doesn't support output_format)
        output_format = anthropic_messages_request.pop("output_format", None)
        if output_format:
            self._convert_output_format_to_inline_schema(
                output_format=output_format,
                anthropic_messages_request=anthropic_messages_request,
            )

        # 5a. Remove `custom` field from tools (Bedrock doesn't support it)
        # Claude Code sends `custom: {defer_loading: true}` on tool definitions,
        # which causes Bedrock to reject the request with "Extra inputs are not permitted"
        # Ref: https://github.com/BerriAI/litellm/issues/22847
        remove_custom_field_from_tools(anthropic_messages_request)
        normalize_tool_input_schema_types_for_bedrock_invoke(anthropic_messages_request)
        ensure_bedrock_anthropic_messages_tool_names(anthropic_messages_request)

        # 6. AUTO-INJECT beta headers based on features used
        anthropic_model_info = AnthropicModelInfo()
        tools = anthropic_messages_optional_request_params.get("tools")
        messages_typed = cast(List[AllMessageValues], messages)
        tool_search_used = anthropic_model_info.is_tool_search_used(tools)
        programmatic_tool_calling_used = (
            anthropic_model_info.is_programmatic_tool_calling_used(tools)
        )
        input_examples_used = anthropic_model_info.is_input_examples_used(tools)

        user_beta_set = set(get_anthropic_beta_from_headers(headers))
        beta_set = set(user_beta_set)
        auto_betas = anthropic_model_info.get_anthropic_beta_list(
            model=model,
            optional_params=anthropic_messages_optional_request_params,
            computer_tool_used=anthropic_model_info.is_computer_tool_used(tools),
            prompt_caching_set=False,
            file_id_used=anthropic_model_info.is_file_id_used(messages_typed),
            mcp_server_used=anthropic_model_info.is_mcp_server_used(
                anthropic_messages_optional_request_params.get("mcp_servers")
            ),
        )
        beta_set.update(auto_betas)

        if injected_thinking_for_clear_thinking:
            beta_set.add("interleaved-thinking-2025-05-14")

        self._filter_context_management_for_bedrock_invoke(
            anthropic_messages_request=anthropic_messages_request,
            beta_set=beta_set,
        )

        self._get_tool_search_beta_header_for_bedrock(
            model=model,
            tool_search_used=tool_search_used,
            programmatic_tool_calling_used=programmatic_tool_calling_used,
            input_examples_used=input_examples_used,
            beta_set=beta_set,
        )

        if "tool-search-tool-2025-10-19" in beta_set:
            beta_set.add("tool-examples-2025-10-29")

        filtered_betas = sorted(
            filter_and_transform_beta_headers(
                beta_headers=list(beta_set),
                provider="bedrock",
            )
        )

        dropped_user_betas = sorted(
            b
            for b in user_beta_set
            if not filter_and_transform_beta_headers([b], provider="bedrock")
        )
        if dropped_user_betas:
            verbose_logger.warning(
                "Bedrock Invoke: dropping unsupported anthropic-beta values "
                "from client headers: %s. Bedrock has no mapping entry for "
                "these; forwarding them would cause a 400.",
                dropped_user_betas,
            )

        if filtered_betas:
            anthropic_messages_request["anthropic_beta"] = filtered_betas

        if (
            litellm.drop_params is True
            and "output_config" in anthropic_messages_request
            and not AnthropicConfig._model_supports_effort_param(model)
        ):
            verbose_logger.warning(
                DROP_UNSUPPORTED_OUTPUT_CONFIG_WARNING,
                model,
            )
            anthropic_messages_request.pop("output_config", None)

        # 7. Final safety net: filter top-level fields to the Bedrock Invoke allowlist.
        # Catches Anthropic-only extensions (output_config, speed, mcp_servers, ...)
        # and any future additions Claude Code may start sending. ``context_management``
        # has already been pre-filtered to its Bedrock-supported subset above.
        allowed = self.BEDROCK_INVOKE_ALLOWED_TOP_LEVEL_FIELDS
        stripped = sorted(k for k in anthropic_messages_request if k not in allowed)
        if stripped:
            verbose_logger.debug(
                "Bedrock Invoke: stripping unsupported top-level request fields: %s",
                stripped,
            )
        anthropic_messages_request = {
            k: v for k, v in anthropic_messages_request.items() if k in allowed
        }

        return anthropic_messages_request

    def get_async_streaming_response_iterator(
        self,
        model: str,
        httpx_response: httpx.Response,
        request_body: dict,
        litellm_logging_obj: LiteLLMLoggingObj,
    ) -> AsyncIterator:
        aws_decoder = AmazonAnthropicClaudeMessagesStreamDecoder(
            model=model,
        )
        completion_stream = aws_decoder.aiter_bytes(
            httpx_response.aiter_bytes(chunk_size=aws_decoder.DEFAULT_CHUNK_SIZE)
        )
        # Convert decoded Bedrock events to Server-Sent Events expected by Anthropic clients.
        return self.bedrock_sse_wrapper(
            completion_stream=completion_stream,
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )

    async def bedrock_sse_wrapper(
        self,
        completion_stream: AsyncIterator[
            Union[bytes, GenericStreamingChunk, ModelResponseStream, dict]
        ],
        litellm_logging_obj: LiteLLMLoggingObj,
        request_body: dict,
    ):
        """
        Bedrock invoke does not return SSE formatted data. This function is a wrapper to ensure litellm chunks are SSE formatted.

        Bedrock's Anthropic-compatible streaming usually puts cache usage fields
        (cache_creation_input_tokens, cache_read_input_tokens) on message_stop.
        Some deployments (including GovCloud) emit the cache breakdown only on
        ``message_start.message.usage``; ``message_delta`` / ``message_stop`` then
        repeat uncached ``input_tokens`` only. We promote cache fields from
        ``message_stop`` onto ``message_delta``, and when those are absent we
        merge them from ``message_start`` so logging/cost sees a consistent usage
        object (fixes negative input costs: LIT-2411).
        """
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import (
            BaseAnthropicMessagesStreamingIterator,
        )

        handler = BaseAnthropicMessagesStreamingIterator(
            litellm_logging_obj=litellm_logging_obj,
            request_body=request_body,
        )

        patched_stream = self._promote_message_stop_usage(completion_stream)

        async for chunk in handler.async_sse_wrapper(patched_stream):
            yield chunk

    @staticmethod
    def _merge_message_start_cache_into_delta_usage(
        delta_usage: Dict[str, Any],
        start_usage: Optional[Dict[str, Any]],
    ) -> None:
        """
        Copy cache breakdown from message_start onto message_delta usage when
        those keys are missing on the delta (GovCloud / some Bedrock streams).
        """
        if not start_usage:
            return
        for field in ("cache_creation_input_tokens", "cache_read_input_tokens"):
            if field not in delta_usage:
                val = start_usage.get(field)
                if val is not None:
                    delta_usage[field] = val
        if "cache_creation" not in delta_usage:
            cc = start_usage.get("cache_creation")
            if cc is not None:
                delta_usage["cache_creation"] = cc

    @staticmethod
    async def _promote_message_stop_usage(
        completion_stream: AsyncIterator[
            Union[bytes, GenericStreamingChunk, ModelResponseStream, dict]
        ],
    ) -> AsyncIterator[Union[bytes, GenericStreamingChunk, ModelResponseStream, dict]]:
        """
        Promote cache usage fields onto message_delta from message_stop (and,
        when stop lacks them, from message_start).  Ensures the final usage
        chunk that logging/cost sees is always self-consistent.
        """
        _CACHE_FIELDS = ("cache_creation_input_tokens", "cache_read_input_tokens")
        pending_delta: Optional[Dict[str, Any]] = None
        start_usage_snapshot: Optional[Dict[str, Any]] = None

        async for chunk in completion_stream:
            if not isinstance(chunk, dict):
                if pending_delta is not None:
                    yield pending_delta
                    pending_delta = None
                yield chunk
                continue

            chunk_type = chunk.get("type")

            if chunk_type == "message_start":
                msg: Dict[str, Any] = cast(Dict[str, Any], chunk.get("message") or {})
                u = msg.get("usage")
                if isinstance(u, dict):
                    start_usage_snapshot = dict(u)
                if pending_delta is not None:
                    yield pending_delta
                    pending_delta = None
                yield chunk
                continue

            if chunk_type == "message_delta":
                pending_delta = cast(Dict[str, Any], chunk)
                continue

            if chunk_type == "message_stop" and pending_delta is not None:
                stop_usage = dict(chunk.get("usage") or {})
                delta_usage = dict(pending_delta.get("usage") or {})

                for field in _CACHE_FIELDS:
                    if field in stop_usage:
                        delta_usage[field] = stop_usage[field]

                raw_input = stop_usage.get("input_tokens")
                if raw_input is not None:
                    delta_usage["input_tokens"] = (
                        raw_input if isinstance(raw_input, int) else 0
                    )

                AmazonAnthropicClaudeMessagesConfig._merge_message_start_cache_into_delta_usage(
                    delta_usage, start_usage_snapshot
                )

                if delta_usage:
                    pending_delta["usage"] = delta_usage  # type: ignore[arg-type]

                yield pending_delta
                pending_delta = None
                yield chunk
                continue

            if pending_delta is not None:
                yield pending_delta
                pending_delta = None

            yield chunk

        if pending_delta is not None:
            delta_usage = dict(pending_delta.get("usage") or {})
            AmazonAnthropicClaudeMessagesConfig._merge_message_start_cache_into_delta_usage(
                delta_usage, start_usage_snapshot
            )
            if delta_usage:
                pending_delta["usage"] = delta_usage  # type: ignore[arg-type]
            yield pending_delta


class AmazonAnthropicClaudeMessagesStreamDecoder(AWSEventStreamDecoder):
    def __init__(
        self,
        model: str,
    ) -> None:
        """
        Iterator to return Bedrock invoke response in anthropic /messages format
        """
        super().__init__(model=model)
        self.DEFAULT_CHUNK_SIZE = 1024

    def _chunk_parser(
        self, chunk_data: dict
    ) -> Union[GChunk, ModelResponseStream, dict]:
        """
        Parse the chunk data into anthropic /messages format

        Bedrock returns usage metrics using camelCase keys. Convert these to
        the Anthropic `/v1/messages` specification so callers receive a
        consistent response shape when streaming.
        """
        amazon_bedrock_invocation_metrics = chunk_data.pop(
            "amazon-bedrock-invocationMetrics", {}
        )
        if amazon_bedrock_invocation_metrics:
            anthropic_usage = {}
            if "inputTokenCount" in amazon_bedrock_invocation_metrics:
                anthropic_usage["input_tokens"] = amazon_bedrock_invocation_metrics[
                    "inputTokenCount"
                ]
            if "outputTokenCount" in amazon_bedrock_invocation_metrics:
                anthropic_usage["output_tokens"] = amazon_bedrock_invocation_metrics[
                    "outputTokenCount"
                ]
            chunk_data["usage"] = anthropic_usage
        return chunk_data

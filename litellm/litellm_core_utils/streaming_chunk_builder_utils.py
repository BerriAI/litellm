import base64
import time
from collections.abc import Iterator, Mapping, Sequence
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypedDict, Union, cast

from litellm._logging import verbose_logger
from litellm.types.llms.openai import (
    ChatCompletionAssistantContentValue,
    ChatCompletionAudioDelta,
)
from litellm.types.utils import (
    CacheCreationTokenDetails,
    ChatCompletionAudioResponse,
    ChatCompletionCustomToolCallPayload,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
    Choices,
    CompletionTokensDetails,
    CompletionTokensDetailsWrapper,
    Function,
    FunctionCall,
    ModelResponse,
    ModelResponseStream,
    PromptTokensDetailsWrapper,
    ServerToolUse,
    Usage,
)
from litellm.utils import print_verbose, token_counter

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.litellm_core_utils.streaming_chunk_builder_utils import (
        UsagePerChunk,
    )
    from litellm.types.llms.openai import (
        ChatCompletionRedactedThinkingBlock,
        ChatCompletionThinkingBlock,
    )


class _ThinkingBlockFragment(TypedDict, total=False):
    type: str | None
    data: str | None
    thinking: str | None
    signature: str | None


class _ThinkingDelta(TypedDict, total=False):
    thinking_blocks: Sequence[_ThinkingBlockFragment]


class _ThinkingChoice(TypedDict, total=False):
    delta: _ThinkingDelta


class _ThinkingChunk(TypedDict):
    choices: Sequence[_ThinkingChoice]


class _ContentChoice(TypedDict, total=False):
    delta: Mapping[str, str | None]


class _ContentChunk(TypedDict):
    choices: Sequence[_ContentChoice]


class _AudioDelta(TypedDict, total=False):
    audio: ChatCompletionAudioDelta | None


class _AudioChoice(TypedDict, total=False):
    delta: _AudioDelta


class _AudioChunk(TypedDict):
    choices: Sequence[_AudioChoice]


class _UsageBearingChunk(TypedDict, total=False):
    usage: Usage | None
    _hidden_params: Mapping[str, str]


class _UsageSummary(TypedDict):
    prompt_tokens: int | None
    completion_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    completion_tokens_details: CompletionTokensDetails | None
    prompt_tokens_details: PromptTokensDetailsWrapper | None
    cost: float | None


def capture_cache_creation_token_details(
    prompt_tokens_details: PromptTokensDetailsWrapper | None,
    current: CacheCreationTokenDetails | None,
) -> CacheCreationTokenDetails | None:
    incoming: Final = cast(
        CacheCreationTokenDetails | None,
        getattr(prompt_tokens_details, "cache_creation_token_details", None),
    )
    if incoming is not None:
        return incoming
    return current


def attach_cache_creation_token_details(
    prompt_tokens_details: PromptTokensDetailsWrapper | None,
    cache_creation_token_details: CacheCreationTokenDetails | None,
) -> PromptTokensDetailsWrapper | None:
    if prompt_tokens_details is None or cache_creation_token_details is None:
        return prompt_tokens_details
    existing: Final = cast(
        CacheCreationTokenDetails | None,
        getattr(prompt_tokens_details, "cache_creation_token_details", None),
    )
    if existing is not None:
        return prompt_tokens_details
    return prompt_tokens_details.model_copy(update={"cache_creation_token_details": cache_creation_token_details})


class ChunkProcessor:
    def __init__(self, chunks: list, messages: list | None = None):
        self.chunks = self._sort_chunks(chunks)
        self.messages = messages
        self.first_chunk = chunks[0]

    def _sort_chunks(self, chunks: list) -> list:
        if not chunks:
            return []

        first_chunk: Final = chunks[0]
        first_hidden_params: dict[str, object] = {}
        if isinstance(first_chunk, dict):
            candidate = first_chunk.get("_hidden_params", {})
            if isinstance(candidate, dict):
                first_hidden_params = candidate
        else:
            candidate = getattr(first_chunk, "_hidden_params", {})
            if isinstance(candidate, dict):
                first_hidden_params = candidate

        if first_hidden_params.get("created_at"):

            def _created_at(chunk: Any) -> int | float:
                if isinstance(chunk, dict):
                    params = chunk.get("_hidden_params", {})
                else:
                    params = getattr(chunk, "_hidden_params", {})
                if isinstance(params, dict):
                    return cast(int | float, params.get("created_at", float("inf")))
                return float("inf")

            return sorted(chunks, key=_created_at)
        return chunks

    def update_model_response_with_hidden_params(
        self, model_response: ModelResponse, chunk: dict[str, Any] | None = None
    ) -> ModelResponse:
        if chunk is None:
            return model_response
        # set hidden params from chunk to model_response
        if model_response is not None and hasattr(model_response, "_hidden_params"):
            model_response._hidden_params = chunk.get("_hidden_params", {})
        return model_response

    @staticmethod
    def apply_provider_assembled_streaming_metadata(
        response: ModelResponse,
        chunks: list[object],
        logging_obj: "Logging | None" = None,
    ) -> None:
        if not chunks:
            return

        model: Final = getattr(response, "model", None)
        if not model:
            return

        custom_llm_provider = None
        if logging_obj is not None:
            custom_llm_provider = logging_obj.model_call_details.get("custom_llm_provider")

        try:
            from litellm.litellm_core_utils.get_llm_provider_logic import (
                get_llm_provider,
            )
            from litellm.types.utils import LlmProviders
            from litellm.utils import ProviderConfigManager

            if custom_llm_provider:
                provider = LlmProviders(custom_llm_provider)
            else:
                _, provider_str, _, _ = get_llm_provider(model)
                provider = LlmProviders(provider_str)

            provider_config: Final = ProviderConfigManager.get_provider_chat_config(
                model=model,
                provider=provider,
            )
            if provider_config is not None:
                provider_config.apply_assembled_streaming_response_metadata(
                    response=response,
                    chunks=chunks,
                )
        except Exception as e:
            verbose_logger.debug(
                "apply_provider_assembled_streaming_metadata failed for model=%s: %s",
                model,
                e,
            )

    @staticmethod
    def _get_chunk_id(chunks: list[dict[str, Any]]) -> str:
        """
        Chunks:
        [{"id": ""}, {"id": "1"}, {"id": "1"}]
        """
        for chunk in chunks:
            if chunk.get("id"):
                return chunk["id"]
        return ""

    @staticmethod
    def _get_model_from_chunks(chunks: list[dict[str, Any]], first_chunk_model: str) -> str:
        """
        Get the actual model from chunks, preferring a model that differs from the first chunk.

        For Azure Model Router, the first chunk may have the request model (e.g., 'azure-model-router')
        while subsequent chunks have the actual model (e.g., 'gpt-4.1-nano-2025-04-14').
        This method finds the actual model for accurate cost calculation.
        """
        # Look for a model in chunks that differs from the first chunk's model
        for chunk in chunks:
            chunk_model = chunk.get("model")
            if chunk_model and chunk_model != first_chunk_model:
                return chunk_model
        # Fall back to first chunk's model if no different model found
        return first_chunk_model

    def build_base_response(self, chunks: list[dict[str, Any]]) -> ModelResponse:
        chunk = self.first_chunk
        id: Final = ChunkProcessor._get_chunk_id(chunks)
        object: Final = chunk["object"]
        created: Final = chunk["created"]
        first_chunk_model: Final = chunk["model"]
        # Get the actual model - for Azure Model Router, this finds the real model from later chunks
        model: Final = ChunkProcessor._get_model_from_chunks(chunks, first_chunk_model)
        system_fingerprint: Final = chunk.get("system_fingerprint", None)

        first_chunk_with_choices: Final = next((c for c in chunks if c.get("choices")), chunk)
        role: Final = first_chunk_with_choices["choices"][0]["delta"]["role"]
        finish_reason = "stop"
        for chunk in chunks:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                chunk_finish_reason = None
                if hasattr(chunk["choices"][0], "finish_reason"):
                    chunk_finish_reason = chunk["choices"][0].finish_reason
                elif "finish_reason" in chunk["choices"][0]:
                    chunk_finish_reason = chunk["choices"][0]["finish_reason"]
                if chunk_finish_reason is not None:
                    finish_reason = chunk_finish_reason

        # Initialize the response dictionary
        response = ModelResponse(
            **{
                "id": id,
                "object": object,
                "created": created,
                "model": model,
                "system_fingerprint": system_fingerprint,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": role, "content": ""},
                        "finish_reason": finish_reason,
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # Modify as needed
                    "completion_tokens": 0,  # Modify as needed
                    "total_tokens": 0,  # Modify as needed
                },
            }
        )

        response = self.update_model_response_with_hidden_params(model_response=response, chunk=chunk)
        return response

    @staticmethod
    def _iter_tool_call_fragments(
        tool_call_chunks: Sequence[Mapping[str, Any]],
    ) -> Iterator[tuple[int, str, str]]:
        for chunk in tool_call_chunks:
            for choice in chunk["choices"]:
                delta = choice.get("delta")
                if not delta:
                    continue
                for tool_call in delta.get("tool_calls", ()):
                    if not tool_call:
                        continue
                    if isinstance(tool_call, dict):
                        index = tool_call.get("index", 0)
                        function = tool_call.get("function")
                        if isinstance(function, dict):
                            if function.get("arguments"):
                                yield index, "arguments", function["arguments"]
                        elif getattr(function, "arguments", None):
                            yield index, "arguments", function.arguments
                        custom = tool_call.get("custom")
                        if isinstance(custom, dict) and custom.get("input"):
                            yield index, "custom_input", custom["input"]
                    else:
                        index = getattr(tool_call, "index", 0)
                        function = getattr(tool_call, "function", None)
                        if getattr(function, "arguments", None):
                            yield index, "arguments", function.arguments
                        custom = getattr(tool_call, "custom", None)
                        if getattr(custom, "input", None):
                            yield index, "custom_input", custom.input

    @staticmethod
    def _join_fragments_by_index_and_field(
        fragment_records: Iterator[tuple[int, str, str]],
    ) -> Mapping[tuple[int, str], str]:
        def group_key(record: tuple[int, str, str]) -> tuple[int, str]:
            return record[0], record[1]

        return MappingProxyType(
            {
                key: "".join(fragment for _, _, fragment in group)
                for key, group in groupby(sorted(fragment_records, key=group_key), key=group_key)
            }
        )

    def get_combined_tool_content(
        self, tool_call_chunks: Sequence[Mapping[str, Any]]
    ) -> list[
        ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall
    ]:  # mutable-ok: assigned verbatim to Message.tool_calls, a list field
        tool_calls_list: list[
            ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall
        ] = []  # mutable-ok: see return type
        tool_call_map: Final[dict[int, dict[str, Any]]] = {}  # Map to store tool calls by index

        for chunk in tool_call_chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta", {})
                tool_calls = delta.get("tool_calls", [])

                for tool_call in tool_calls:
                    # Handle both dict and object formats
                    if not tool_call:
                        continue

                    # Check if tool_call has function (either as attribute or dict key)
                    has_function = False
                    has_custom = False
                    if isinstance(tool_call, dict):
                        has_function = "function" in tool_call and tool_call["function"] is not None
                        has_custom = "custom" in tool_call and tool_call["custom"] is not None
                    else:
                        has_function = hasattr(tool_call, "function") and tool_call.function is not None
                        has_custom = getattr(tool_call, "custom", None) is not None

                    if not has_function and not has_custom:
                        continue

                    # Get index (handle both dict and object)
                    if isinstance(tool_call, dict):
                        index = tool_call.get("index", 0)
                    else:
                        index = getattr(tool_call, "index", 0)

                    if index not in tool_call_map:
                        tool_call_map[index] = {
                            "id": None,
                            "name": None,
                            "type": None,
                            "custom_name": None,
                            "provider_specific_fields": None,
                        }

                    # Extract id, type, and function data (handle both dict and object)
                    if isinstance(tool_call, dict):
                        if tool_call.get("id"):
                            tool_call_map[index]["id"] = tool_call["id"]
                        if tool_call.get("type"):
                            tool_call_map[index]["type"] = tool_call["type"]

                        function = tool_call.get("function", {})
                        if isinstance(function, dict):
                            if function.get("name"):
                                tool_call_map[index]["name"] = function["name"]
                        else:
                            # function is an object
                            if hasattr(function, "name") and function.name:
                                tool_call_map[index]["name"] = function.name

                        custom = tool_call.get("custom")
                        if isinstance(custom, dict):
                            if custom.get("name"):
                                tool_call_map[index]["custom_name"] = custom["name"]
                    else:
                        # tool_call is an object
                        if hasattr(tool_call, "id") and tool_call.id:
                            tool_call_map[index]["id"] = tool_call.id
                        if hasattr(tool_call, "type") and tool_call.type:
                            tool_call_map[index]["type"] = tool_call.type
                        if hasattr(tool_call, "function"):
                            if hasattr(tool_call.function, "name") and tool_call.function.name:
                                tool_call_map[index]["name"] = tool_call.function.name

                        custom = getattr(tool_call, "custom", None)
                        if custom is not None:
                            if getattr(custom, "name", None):
                                tool_call_map[index]["custom_name"] = custom.name

                    # Preserve provider_specific_fields from streaming chunks
                    provider_fields = None
                    if isinstance(tool_call, dict):
                        provider_fields = tool_call.get("provider_specific_fields")
                        if not provider_fields and isinstance(tool_call.get("function"), dict):
                            provider_fields = tool_call["function"].get("provider_specific_fields")
                    else:
                        if hasattr(tool_call, "provider_specific_fields") and tool_call.provider_specific_fields:
                            provider_fields = tool_call.provider_specific_fields
                        elif (
                            hasattr(tool_call, "function")
                            and hasattr(tool_call.function, "provider_specific_fields")
                            and tool_call.function.provider_specific_fields
                        ):
                            provider_fields = tool_call.function.provider_specific_fields

                    if provider_fields:
                        # Merge provider_specific_fields if multiple chunks have them
                        if tool_call_map[index]["provider_specific_fields"] is None:
                            tool_call_map[index]["provider_specific_fields"] = {}
                        if isinstance(provider_fields, dict):
                            tool_call_map[index]["provider_specific_fields"].update(provider_fields)

        joined_fragments: Final = self._join_fragments_by_index_and_field(
            self._iter_tool_call_fragments(tool_call_chunks)
        )

        # Convert the map to a list of tool calls
        for index in sorted(tool_call_map.keys()):
            tool_call_data = tool_call_map[index]
            if tool_call_data["id"] and tool_call_data["custom_name"]:
                tool_calls_list.append(
                    ChatCompletionMessageCustomToolCall(
                        id=tool_call_data["id"],
                        custom=ChatCompletionCustomToolCallPayload(
                            name=tool_call_data["custom_name"],
                            input=joined_fragments.get((index, "custom_input"), ""),
                        ),
                    )
                )
            elif tool_call_data["id"] and tool_call_data["name"]:
                combined_arguments = joined_fragments.get((index, "arguments"), "") or "{}"

                # Build function - provider_specific_fields should be on tool_call level, not function level
                function = Function(
                    arguments=combined_arguments,
                    name=tool_call_data["name"],
                )

                # Prepare params for ChatCompletionMessageToolCall
                tool_call_params = {
                    "id": tool_call_data["id"],
                    "function": function,
                    "type": tool_call_data["type"] or "function",
                }

                # Add provider_specific_fields if present (for thought signatures in Gemini 3)
                if tool_call_data.get("provider_specific_fields"):
                    tool_call_params["provider_specific_fields"] = tool_call_data["provider_specific_fields"]

                tool_call = ChatCompletionMessageToolCall(**tool_call_params)
                tool_calls_list.append(tool_call)

        return tool_calls_list

    def get_combined_function_call_content(self, function_call_chunks: list[dict[str, Any]]) -> FunctionCall:
        argument_list: Final = []
        delta = function_call_chunks[0]["choices"][0]["delta"]
        function_call = delta.get("function_call", "")
        function_call_name: Final = function_call.name

        for chunk in function_call_chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta", {})
                function_call = delta.get("function_call", "")

                # Check if a function call is present
                if function_call:
                    # Now, function_call is expected to be a dictionary
                    arguments = function_call.arguments
                    argument_list.append(arguments)

        combined_arguments: Final = "".join(argument_list)

        return FunctionCall(
            name=function_call_name,
            arguments=combined_arguments,
        )

    def get_combined_content(
        self, chunks: Sequence["_ContentChunk"], delta_key: str = "content"
    ) -> ChatCompletionAssistantContentValue:
        content_list: Final[list[str]] = []
        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta", {})
                content = delta.get(delta_key, "")
                if content is None:
                    continue  # openai v1.0.0 sets content = None for chunks
                content_list.append(content)

        # Combine the "content" strings into a single string || combine the 'function' strings into a single string
        combined_content: Final = "".join(content_list)

        # Update the "content" field within the response dictionary
        return combined_content

    def get_combined_thinking_content(
        self, chunks: Sequence["_ThinkingChunk"]
    ) -> list[Union["ChatCompletionThinkingBlock", "ChatCompletionRedactedThinkingBlock"]] | None:
        from litellm.types.llms.openai import (
            ChatCompletionRedactedThinkingBlock,
            ChatCompletionThinkingBlock,
        )

        thinking_blocks: Final[list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock]] = []
        current_thinking_text_parts: list[str] = []
        current_signature: str | None = None

        def _flush_thinking_block() -> None:
            nonlocal current_thinking_text_parts, current_signature
            if len(current_thinking_text_parts) > 0 and current_signature:
                thinking_blocks.append(
                    ChatCompletionThinkingBlock(
                        type="thinking",
                        thinking="".join(current_thinking_text_parts),
                        signature=current_signature,
                    )
                )
            current_thinking_text_parts = []
            current_signature = None

        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta", {})
                thinking = delta.get("thinking_blocks", None)
                if thinking and isinstance(thinking, list):
                    for thinking_block in thinking:
                        thinking_type = thinking_block.get("type", None)
                        if thinking_type and thinking_type == "redacted_thinking":
                            _flush_thinking_block()
                            redacted_data = thinking_block.get("data", None)
                            if redacted_data:
                                thinking_blocks.append(
                                    ChatCompletionRedactedThinkingBlock(
                                        type="redacted_thinking",
                                        data=redacted_data,
                                    )
                                )
                        else:
                            thinking_text = thinking_block.get("thinking", None)
                            if thinking_text:
                                current_thinking_text_parts.append(thinking_text)
                            signature = thinking_block.get("signature", None)
                            if signature:
                                current_signature = signature
                                _flush_thinking_block()

        _flush_thinking_block()

        if len(thinking_blocks) > 0:
            return thinking_blocks
        return None

    def get_combined_reasoning_content(self, chunks: Sequence["_ContentChunk"]) -> ChatCompletionAssistantContentValue:
        return self.get_combined_content(chunks, delta_key="reasoning_content")

    def get_combined_audio_content(self, chunks: Sequence["_AudioChunk"]) -> ChatCompletionAudioResponse:
        base64_data_list: Final[list[str]] = []
        transcript_list: Final[list[str]] = []
        expires_at: int | None = None
        id: str | None = None

        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta: _AudioDelta = choice.get("delta") or {}
                audio: ChatCompletionAudioDelta | None = delta.get("audio")
                if audio is not None:
                    for k, v in audio.items():
                        if k == "data" and v is not None and isinstance(v, str):
                            base64_data_list.append(v)
                        elif k == "transcript" and v is not None and isinstance(v, str):
                            transcript_list.append(v)
                        elif k == "expires_at" and v is not None and isinstance(v, int):
                            expires_at = v
                        elif k == "id" and v is not None and isinstance(v, str):
                            id = v

        concatenated_audio: Final = concatenate_base64_list(base64_data_list)
        return ChatCompletionAudioResponse(
            data=concatenated_audio,
            expires_at=expires_at or int(time.time() + 3600),
            transcript="".join(transcript_list),
            id=id,
        )

    def _usage_chunk_calculation_helper(self, usage_chunk: Usage) -> "_UsageSummary":
        prompt_tokens = 0
        completion_tokens = 0
        ## anthropic prompt caching information ##
        cache_creation_input_tokens: int | None = None
        cache_read_input_tokens: int | None = None
        completion_tokens_details: CompletionTokensDetails | None = None
        prompt_tokens_details: PromptTokensDetailsWrapper | None = None
        cost: float | None = None

        if "prompt_tokens" in usage_chunk:
            prompt_tokens = usage_chunk.get("prompt_tokens", 0) or 0
        if "completion_tokens" in usage_chunk:
            completion_tokens = usage_chunk.get("completion_tokens", 0) or 0
        if "cache_creation_input_tokens" in usage_chunk:
            cache_creation_input_tokens = usage_chunk.get("cache_creation_input_tokens")
        if "cache_read_input_tokens" in usage_chunk:
            cache_read_input_tokens = usage_chunk.get("cache_read_input_tokens")
        if "cost" in usage_chunk:
            cost = usage_chunk.get("cost")
        if hasattr(usage_chunk, "completion_tokens_details"):
            if isinstance(usage_chunk.completion_tokens_details, dict):
                completion_tokens_details = CompletionTokensDetails(**usage_chunk.completion_tokens_details)
            elif isinstance(usage_chunk.completion_tokens_details, CompletionTokensDetails):
                completion_tokens_details = usage_chunk.completion_tokens_details
        if hasattr(usage_chunk, "prompt_tokens_details"):
            if isinstance(usage_chunk.prompt_tokens_details, dict):
                prompt_tokens_details = PromptTokensDetailsWrapper(**usage_chunk.prompt_tokens_details)
            elif isinstance(usage_chunk.prompt_tokens_details, PromptTokensDetailsWrapper):
                prompt_tokens_details = usage_chunk.prompt_tokens_details

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "completion_tokens_details": completion_tokens_details,
            "prompt_tokens_details": prompt_tokens_details,
            "cost": cost,
        }

    def count_reasoning_tokens(self, response: ModelResponse) -> int | None:
        reasoning_tokens: int | None = None
        for choice in response.choices:
            if (
                hasattr(cast(Choices, choice).message, "reasoning_content")
                and cast(Choices, choice).message.reasoning_content is not None
            ):
                if reasoning_tokens is None:
                    reasoning_tokens = 0
                reasoning_tokens += token_counter(
                    text=cast(Choices, choice).message.reasoning_content,
                    count_response_tokens=True,
                )

        return reasoning_tokens

    @staticmethod
    def _extract_usage_chunk(chunk: "_UsageBearingChunk | ModelResponse | ModelResponseStream") -> Usage | None:
        usage_chunk: Usage | None = None
        if hasattr(chunk, "usage") and chunk.usage is not None:
            usage_chunk = chunk.usage
        elif "usage" in chunk:
            usage_chunk = chunk["usage"]
        elif (isinstance(chunk, ModelResponse) or isinstance(chunk, ModelResponseStream)) and hasattr(
            chunk, "_hidden_params"
        ):
            usage_chunk = chunk._hidden_params.get("usage", None)

        if isinstance(usage_chunk, dict):
            return Usage(**usage_chunk)
        return usage_chunk

    def _calculate_usage_per_chunk(
        self,
        chunks: Sequence["_UsageBearingChunk | ModelResponse"],
    ) -> "UsagePerChunk":
        from litellm.types.litellm_core_utils.streaming_chunk_builder_utils import (
            UsagePerChunk,
        )

        # # Update usage information if needed
        prompt_tokens = 0
        completion_tokens = 0
        # Anthropic's `message_start` SSE event carries usage.output_tokens=1 as a
        # cursor/placeholder; the real value only arrives in `message_delta`.
        # If a stream is cancelled before `message_delta` lands, the last-wins
        # accumulator below leaves completion_tokens stuck at 1 — which then
        # bypasses the `completion_tokens or token_counter(...)` fallback in
        # calculate_usage() because 1 is truthy. Count the completion-bearing
        # usage events so `_reset_anthropic_cursor_completion_tokens` can tell a
        # legitimate single-token reply (Anthropic emits 1 in BOTH message_start
        # AND message_delta, so >=2 events is positive evidence message_delta
        # arrived) from a stale lone cursor.
        completion_usage_updates = 0
        ## anthropic prompt caching information ##
        cache_creation_input_tokens: int | None = None
        cache_read_input_tokens: int | None = None

        server_tool_use: ServerToolUse | None = None
        web_search_requests: int | None = None
        completion_tokens_details: CompletionTokensDetails | None = None
        prompt_tokens_details: PromptTokensDetailsWrapper | None = None
        # Anthropic emits the cache-creation TTL breakdown (5m/1h split) only on
        # the `message_start` event; the later `message_delta` carries the flat
        # cache-creation count but drops the nested breakdown. prompt_tokens_details
        # is last-wins, so without preserving this separately the 1h breakdown is
        # lost and 1h cache writes get billed at the 5m rate.
        cache_creation_token_details: CacheCreationTokenDetails | None = None
        cost: float | None = None

        for chunk in chunks:
            usage_chunk = self._extract_usage_chunk(chunk)

            if usage_chunk is not None:
                usage_chunk_dict = self._usage_chunk_calculation_helper(usage_chunk)
                if usage_chunk_dict["prompt_tokens"] is not None and usage_chunk_dict["prompt_tokens"] > 0:
                    prompt_tokens = usage_chunk_dict["prompt_tokens"]
                if usage_chunk_dict["completion_tokens"] is not None and usage_chunk_dict["completion_tokens"] > 0:
                    completion_tokens = usage_chunk_dict["completion_tokens"]
                    completion_usage_updates += 1
                if usage_chunk_dict["cache_creation_input_tokens"] is not None and (
                    usage_chunk_dict["cache_creation_input_tokens"] > 0 or cache_creation_input_tokens is None
                ):
                    cache_creation_input_tokens = usage_chunk_dict["cache_creation_input_tokens"]
                if usage_chunk_dict["cache_read_input_tokens"] is not None and (
                    usage_chunk_dict["cache_read_input_tokens"] > 0 or cache_read_input_tokens is None
                ):
                    cache_read_input_tokens = usage_chunk_dict["cache_read_input_tokens"]
                if usage_chunk_dict["completion_tokens_details"] is not None:
                    completion_tokens_details = usage_chunk_dict["completion_tokens_details"]
                if hasattr(usage_chunk, "server_tool_use") and usage_chunk.server_tool_use is not None:
                    # Coerce dict to ServerToolUse so downstream cost-calc code
                    # (which accesses .web_search_requests as an attribute)
                    # doesn't raise AttributeError. Some providers / streaming
                    # paths leave server_tool_use as a plain dict on the chunk.
                    if isinstance(usage_chunk.server_tool_use, dict):
                        server_tool_use = ServerToolUse(**usage_chunk.server_tool_use)
                    elif isinstance(usage_chunk.server_tool_use, ServerToolUse):
                        server_tool_use = usage_chunk.server_tool_use
                    else:
                        server_tool_use = ServerToolUse.model_validate(usage_chunk.server_tool_use)
                if (
                    usage_chunk_dict["prompt_tokens_details"] is not None
                    and getattr(
                        usage_chunk_dict["prompt_tokens_details"],
                        "web_search_requests",
                        None,
                    )
                    is not None
                ):
                    web_search_requests = getattr(
                        usage_chunk_dict["prompt_tokens_details"],
                        "web_search_requests",
                    )

                prompt_tokens_details = usage_chunk_dict["prompt_tokens_details"] or prompt_tokens_details

                cache_creation_token_details = capture_cache_creation_token_details(
                    prompt_tokens_details, cache_creation_token_details
                )

                if usage_chunk_dict["cost"] is not None:
                    cost = usage_chunk_dict["cost"]

        prompt_tokens_details = attach_cache_creation_token_details(prompt_tokens_details, cache_creation_token_details)

        completion_tokens = self._reset_anthropic_cursor_completion_tokens(
            chunks=chunks,
            completion_tokens=completion_tokens,
            completion_usage_updates=completion_usage_updates,
        )

        return UsagePerChunk(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            server_tool_use=server_tool_use,
            web_search_requests=web_search_requests,
            completion_tokens_details=completion_tokens_details,
            prompt_tokens_details=prompt_tokens_details,
            cost=cost,
        )

    @staticmethod
    def _reset_anthropic_cursor_completion_tokens(
        chunks: Sequence["_UsageBearingChunk | ModelResponse"],
        completion_tokens: int,
        completion_usage_updates: int,
    ) -> int:
        """Reset a stale Anthropic ``message_start`` cursor placeholder to 0.

        See the ``completion_usage_updates`` comment in
        ``_calculate_usage_per_chunk``. The accumulated value is NOT a stale
        cursor when either it is > 1 (definitely not a placeholder) or we saw
        >= 2 completion-bearing usage events (positive evidence ``message_delta``
        arrived). Otherwise — the only completion update we ever saw was the
        Anthropic ``message_start`` cursor (=1) — reset to 0 so
        ``calculate_usage()``'s ``or token_counter(text=...)`` fallback estimates
        from the actually-received completion text instead of trusting the
        placeholder. Gated on ``custom_llm_provider == "anthropic"`` so the
        heuristic (which encodes Anthropic's specific message_start SSE shape)
        does not silently affect other providers that may legitimately report
        ``completion_tokens=1`` from a single usage event.
        """
        saw_non_cursor_completion: Final = completion_tokens > 1 or completion_usage_updates >= 2
        if saw_non_cursor_completion:
            return completion_tokens

        custom_llm_provider: str | None = None
        if chunks:
            first_chunk: Final = chunks[0]
            if isinstance(first_chunk, dict):
                hp = first_chunk.get("_hidden_params")
            else:
                hp = getattr(first_chunk, "_hidden_params", None)
            if isinstance(hp, dict):
                custom_llm_provider = hp.get("custom_llm_provider")

        if custom_llm_provider == "anthropic" and completion_tokens == 1:
            return 0
        return completion_tokens

    def calculate_usage(
        self,
        chunks: Sequence["_UsageBearingChunk | ModelResponse"],
        model: str,
        completion_output: str,
        messages: list | None = None,
        reasoning_tokens: int | None = None,
    ) -> Usage:
        """
        Calculate usage for the given chunks.
        """
        returned_usage = Usage()
        # # Update usage information if needed

        calculated_usage_per_chunk: Final = self._calculate_usage_per_chunk(chunks=chunks)
        prompt_tokens: Final = calculated_usage_per_chunk["prompt_tokens"]
        completion_tokens: Final = calculated_usage_per_chunk["completion_tokens"]
        ## anthropic prompt caching information ##
        cache_creation_input_tokens: Final[int | None] = calculated_usage_per_chunk["cache_creation_input_tokens"]
        cache_read_input_tokens: Final[int | None] = calculated_usage_per_chunk["cache_read_input_tokens"]

        server_tool_use: Final[ServerToolUse | None] = calculated_usage_per_chunk["server_tool_use"]
        web_search_requests: Final[int | None] = calculated_usage_per_chunk["web_search_requests"]
        completion_tokens_details: Final[CompletionTokensDetails | None] = calculated_usage_per_chunk[
            "completion_tokens_details"
        ]
        prompt_tokens_details: PromptTokensDetailsWrapper | None = calculated_usage_per_chunk["prompt_tokens_details"]
        cost: Final[float | None] = calculated_usage_per_chunk["cost"]

        try:
            returned_usage.prompt_tokens = prompt_tokens or token_counter(model=model, messages=messages)
        except Exception:  # don't allow this failing to block a complete streaming response from being returned
            print_verbose("token_counter failed, assuming prompt tokens is 0")
            returned_usage.prompt_tokens = 0
        returned_usage.completion_tokens = (
            completion_tokens
            or token_counter(
                model=model,
                text=completion_output,
                count_response_tokens=True,  # count_response_tokens is a Flag to tell token counter this is a response, No need to add extra tokens we do for input messages
            )
        )
        returned_usage.total_tokens = returned_usage.prompt_tokens + returned_usage.completion_tokens

        if cache_creation_input_tokens is not None:
            returned_usage._cache_creation_input_tokens = cache_creation_input_tokens
            setattr(
                returned_usage,
                "cache_creation_input_tokens",
                cache_creation_input_tokens,
            )  # for anthropic
        if cache_read_input_tokens is not None:
            returned_usage._cache_read_input_tokens = cache_read_input_tokens
            setattr(returned_usage, "cache_read_input_tokens", cache_read_input_tokens)  # for anthropic
        if completion_tokens_details is not None:
            if isinstance(completion_tokens_details, CompletionTokensDetails):
                returned_usage.completion_tokens_details = CompletionTokensDetailsWrapper.model_validate(
                    completion_tokens_details.model_dump()
                )
            else:
                returned_usage.completion_tokens_details = completion_tokens_details

        if reasoning_tokens is not None:
            if returned_usage.completion_tokens_details is None:
                returned_usage.completion_tokens_details = CompletionTokensDetailsWrapper(
                    reasoning_tokens=reasoning_tokens
                )
            elif (
                returned_usage.completion_tokens_details is not None
                and returned_usage.completion_tokens_details.reasoning_tokens is None
            ):
                returned_usage.completion_tokens_details.reasoning_tokens = reasoning_tokens
        if prompt_tokens_details is not None:
            returned_usage.prompt_tokens_details = prompt_tokens_details

        if server_tool_use is not None:
            returned_usage.server_tool_use = server_tool_use
        if web_search_requests is not None:
            if returned_usage.prompt_tokens_details is None:
                returned_usage.prompt_tokens_details = PromptTokensDetailsWrapper(
                    web_search_requests=web_search_requests
                )
            else:
                returned_usage.prompt_tokens_details.web_search_requests = web_search_requests

        if cost is not None:
            setattr(returned_usage, "cost", cost)

        # Return a new usage object with the new values

        returned_usage = Usage(**returned_usage.model_dump())

        return returned_usage


def concatenate_base64_list(base64_strings: list[str]) -> str:
    """
    Concatenates a list of base64-encoded strings.

    Args:
        base64_strings (List[str]): A list of base64 strings to concatenate.

    Returns:
        str: The concatenated result as a base64-encoded string.
    """
    # Decode each base64 string and collect the resulting bytes
    combined_bytes: Final = b"".join(base64.b64decode(b64_str) for b64_str in base64_strings)

    # Encode the concatenated bytes back to base64
    return base64.b64encode(combined_bytes).decode("utf-8")

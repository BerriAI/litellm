import base64
import time
from collections.abc import Iterator, Mapping, Sequence
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypedDict, TypeGuard, Union

from litellm._logging import verbose_logger
from litellm.types.llms.openai import (
    ChatCompletionAssistantContentValue,
)
from litellm.types.utils import (
    CacheCreationTokenDetails,
    ChatCompletionAudioResponse,
    ChatCompletionCustomToolCallPayload,
    ChatCompletionMessageCustomToolCall,
    ChatCompletionMessageToolCall,
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
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.types.litellm_core_utils.streaming_chunk_builder_utils import (
        UsagePerChunk,
    )
    from litellm.types.llms.openai import (
        ChatCompletionRedactedThinkingBlock,
        ChatCompletionThinkingBlock,
    )


def _is_str_keyed_dict(value: object) -> TypeGuard[dict[str, object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, dict)


def _is_object_list(value: object) -> TypeGuard[list[object]]:  # guard-ok: isinstance narrows correctly; predicate is trivially correct  # fmt: skip
    return isinstance(value, list)


def _field(source: object, key: str) -> object:
    if _is_str_keyed_dict(source):
        return source.get(key)
    value: Final[object] = getattr(source, key, None)
    return value


def _chunk_hidden_params(chunk: object) -> dict[str, object]:
    candidate: Final = _field(chunk, "_hidden_params")
    if _is_str_keyed_dict(candidate):
        return candidate
    return {}


class _UsageChunkFields(TypedDict):
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
    incoming: Final[CacheCreationTokenDetails | None] = getattr(
        prompt_tokens_details, "cache_creation_token_details", None
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
    existing: Final[CacheCreationTokenDetails | None] = getattr(
        prompt_tokens_details, "cache_creation_token_details", None
    )
    if existing is not None:
        return prompt_tokens_details
    return prompt_tokens_details.model_copy(update={"cache_creation_token_details": cache_creation_token_details})


class ChunkProcessor:
    def __init__(self, chunks: list, messages: list | None = None):
        self.chunks = self._sort_chunks(chunks)
        self.messages = messages
        self.first_chunk: object = chunks[0]

    def _sort_chunks(self, chunks: list) -> list:
        if not chunks:
            return []

        first_hidden_params: Final = _chunk_hidden_params(chunks[0])

        if first_hidden_params.get("created_at"):

            def _created_at(chunk: object) -> int | float:
                created_at: Final = _chunk_hidden_params(chunk).get("created_at", float("inf"))
                if isinstance(created_at, (int, float)):
                    return created_at
                return float("inf")

            return sorted(chunks, key=_created_at)
        return chunks

    def update_model_response_with_hidden_params(
        self, model_response: ModelResponse, chunk: object = None
    ) -> ModelResponse:
        if chunk is None:
            return model_response
        # set hidden params from chunk to model_response
        if model_response is not None and hasattr(model_response, "_hidden_params"):
            hidden_params: Final = _field(chunk, "_hidden_params")
            model_response._hidden_params = hidden_params if _is_str_keyed_dict(hidden_params) else {}
        return model_response

    @staticmethod
    def apply_provider_assembled_streaming_metadata(
        response: ModelResponse,
        chunks: list[object],
        logging_obj: "LiteLLMLoggingObj | None" = None,
    ) -> None:
        if not chunks:
            return

        model: Final[str | None] = getattr(response, "model", None)
        if not model:
            return

        custom_llm_provider: object = None
        if logging_obj is not None:
            custom_llm_provider = _field(logging_obj.model_call_details, "custom_llm_provider")

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
    def _get_chunk_id(chunks: Sequence[Mapping[str, object]]) -> str:
        """
        Chunks:
        [{"id": ""}, {"id": "1"}, {"id": "1"}]
        """
        for chunk in chunks:
            chunk_id = chunk.get("id")
            if isinstance(chunk_id, str) and chunk_id:
                return chunk_id
        return ""

    @staticmethod
    def _get_model_from_chunks(chunks: Sequence[Mapping[str, object]], first_chunk_model: str) -> str:
        """
        Get the actual model from chunks, preferring a model that differs from the first chunk.

        For Azure Model Router, the first chunk may have the request model (e.g., 'azure-model-router')
        while subsequent chunks have the actual model (e.g., 'gpt-4.1-nano-2025-04-14').
        This method finds the actual model for accurate cost calculation.
        """
        # Look for a model in chunks that differs from the first chunk's model
        for chunk in chunks:
            chunk_model = chunk.get("model")
            if isinstance(chunk_model, str) and chunk_model and chunk_model != first_chunk_model:
                return chunk_model
        # Fall back to first chunk's model if no different model found
        return first_chunk_model

    def build_base_response(self, chunks: Sequence[Mapping[str, object]]) -> ModelResponse:
        chunk = self.first_chunk
        id: Final = ChunkProcessor._get_chunk_id(chunks)
        object: Final = _field(chunk, "object")
        created: Final = _field(chunk, "created")
        first_chunk_model_raw: Final = _field(chunk, "model")
        first_chunk_model: Final = first_chunk_model_raw if isinstance(first_chunk_model_raw, str) else ""
        # Get the actual model - for Azure Model Router, this finds the real model from later chunks
        model: Final = ChunkProcessor._get_model_from_chunks(chunks, first_chunk_model)
        system_fingerprint: Final = _field(chunk, "system_fingerprint")

        first_chunk_with_choices: Final = next((c for c in chunks if c.get("choices")), None)
        if first_chunk_with_choices is not None:
            role_choices = first_chunk_with_choices.get("choices")
        else:
            role_choices = _field(chunk, "choices")
        role = None
        if _is_object_list(role_choices):
            first_role_choice: Final = role_choices[0]
            role = _field(_field(first_role_choice, "delta"), "role")
        finish_reason = "stop"
        for chunk in chunks:
            chunk_choices = chunk.get("choices") if "choices" in chunk else None
            if _is_object_list(chunk_choices) and len(chunk_choices) > 0:
                first_choice = chunk_choices[0]
                chunk_finish_reason = _field(first_choice, "finish_reason")
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
        tool_call_chunks: Sequence[Mapping[str, object]],
    ) -> Iterator[tuple[int, str, str]]:
        for chunk in tool_call_chunks:
            choices = chunk["choices"]
            if not _is_object_list(choices):
                continue
            for choice in choices:
                delta = _field(choice, "delta")
                if not delta:
                    continue
                tool_calls = _field(delta, "tool_calls")
                if not _is_object_list(tool_calls):
                    continue
                for tool_call in tool_calls:
                    if not tool_call:
                        continue
                    index_raw = _field(tool_call, "index")
                    index = index_raw if isinstance(index_raw, int) else 0
                    function = _field(tool_call, "function")
                    arguments = _field(function, "arguments")
                    if isinstance(arguments, str) and arguments:
                        yield index, "arguments", arguments
                    custom = _field(tool_call, "custom")
                    custom_input = _field(custom, "input")
                    if isinstance(custom_input, str) and custom_input:
                        yield index, "custom_input", custom_input

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
        self, tool_call_chunks: Sequence[Mapping[str, object]]
    ) -> list[
        ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall
    ]:  # mutable-ok: assigned verbatim to Message.tool_calls, a list field
        tool_calls_list: list[
            ChatCompletionMessageToolCall | ChatCompletionMessageCustomToolCall
        ] = []  # mutable-ok: see return type
        tool_call_map: Final[dict[int, dict[str, object]]] = {}  # Map to store tool calls by index

        for chunk in tool_call_chunks:
            choices = chunk["choices"]
            if not _is_object_list(choices):
                continue
            for choice in choices:
                delta = _field(choice, "delta")
                tool_calls = _field(delta, "tool_calls")
                if not _is_object_list(tool_calls):
                    continue

                for tool_call in tool_calls:
                    # Handle both dict and object formats
                    if not tool_call:
                        continue

                    # Check if tool_call has function (either as attribute or dict key)
                    has_function = _field(tool_call, "function") is not None
                    has_custom = _field(tool_call, "custom") is not None

                    if not has_function and not has_custom:
                        continue

                    # Get index (handle both dict and object)
                    index_raw = _field(tool_call, "index")
                    index = index_raw if isinstance(index_raw, int) else 0

                    if index not in tool_call_map:
                        tool_call_map[index] = {
                            "id": None,
                            "name": None,
                            "type": None,
                            "custom_name": None,
                            "provider_specific_fields": None,
                        }

                    # Extract id, type, and function data (handle both dict and object)
                    tool_call_id = _field(tool_call, "id")
                    if tool_call_id:
                        tool_call_map[index]["id"] = tool_call_id
                    tool_call_type = _field(tool_call, "type")
                    if tool_call_type:
                        tool_call_map[index]["type"] = tool_call_type

                    function = _field(tool_call, "function")
                    function_name = _field(function, "name")
                    if function_name:
                        tool_call_map[index]["name"] = function_name

                    custom = _field(tool_call, "custom")
                    custom_name = _field(custom, "name")
                    if custom_name:
                        tool_call_map[index]["custom_name"] = custom_name

                    # Preserve provider_specific_fields from streaming chunks
                    provider_fields = _field(tool_call, "provider_specific_fields")
                    if not provider_fields:
                        provider_fields = _field(function, "provider_specific_fields")

                    if provider_fields:
                        # Merge provider_specific_fields if multiple chunks have them
                        if tool_call_map[index]["provider_specific_fields"] is None:
                            tool_call_map[index]["provider_specific_fields"] = {}
                        existing_provider_fields = tool_call_map[index]["provider_specific_fields"]
                        if _is_str_keyed_dict(provider_fields) and _is_str_keyed_dict(existing_provider_fields):
                            existing_provider_fields.update(provider_fields)

        joined_fragments: Final = self._join_fragments_by_index_and_field(
            self._iter_tool_call_fragments(tool_call_chunks)
        )

        # Convert the map to a list of tool calls
        for index in sorted(tool_call_map.keys()):
            tool_call_data = tool_call_map[index]
            data_id = tool_call_data["id"]
            data_custom_name = tool_call_data["custom_name"]
            data_name = tool_call_data["name"]
            data_type = tool_call_data["type"]
            if isinstance(data_id, str) and data_id and isinstance(data_custom_name, str) and data_custom_name:
                tool_calls_list.append(
                    ChatCompletionMessageCustomToolCall(
                        id=data_id,
                        custom=ChatCompletionCustomToolCallPayload(
                            name=data_custom_name,
                            input=joined_fragments.get((index, "custom_input"), ""),
                        ),
                    )
                )
            elif isinstance(data_id, str) and data_id and isinstance(data_name, str) and data_name:
                combined_arguments = joined_fragments.get((index, "arguments"), "") or "{}"

                # Build function - provider_specific_fields should be on tool_call level, not function level
                function = Function(
                    arguments=combined_arguments,
                    name=data_name,
                )

                resolved_type = data_type if isinstance(data_type, str) and data_type else "function"

                # Add provider_specific_fields if present (for thought signatures in Gemini 3)
                provider_specific_fields = tool_call_data.get("provider_specific_fields")
                if _is_str_keyed_dict(provider_specific_fields) and provider_specific_fields:
                    tool_call = ChatCompletionMessageToolCall(
                        id=data_id,
                        function=function,
                        type=resolved_type,
                        provider_specific_fields=provider_specific_fields,
                    )
                else:
                    tool_call = ChatCompletionMessageToolCall(
                        id=data_id,
                        function=function,
                        type=resolved_type,
                    )
                tool_calls_list.append(tool_call)

        return tool_calls_list

    def get_combined_function_call_content(self, function_call_chunks: Sequence[Mapping[str, object]]) -> FunctionCall:
        argument_list: Final[list[str]] = []
        first_choices: Final = function_call_chunks[0]["choices"]
        first_choice: Final = first_choices[0] if _is_object_list(first_choices) else None
        first_function_call: Final = _field(_field(first_choice, "delta"), "function_call")
        function_call_name_raw: Final = _field(first_function_call, "name")
        function_call_name: Final = function_call_name_raw if isinstance(function_call_name_raw, str) else None

        for chunk in function_call_chunks:
            choices = chunk["choices"]
            if not _is_object_list(choices):
                continue
            for choice in choices:
                delta = _field(choice, "delta")
                function_call = _field(delta, "function_call")

                # Check if a function call is present
                if function_call:
                    arguments = _field(function_call, "arguments")
                    if isinstance(arguments, str):
                        argument_list.append(arguments)

        combined_arguments: Final = "".join(argument_list)

        return FunctionCall(
            name=function_call_name,
            arguments=combined_arguments,
        )

    def get_combined_content(
        self, chunks: Sequence[Mapping[str, object]], delta_key: str = "content"
    ) -> ChatCompletionAssistantContentValue:
        content_list: Final[list[str]] = []
        for chunk in chunks:
            choices = chunk["choices"]
            if not _is_object_list(choices):
                continue
            for choice in choices:
                delta = _field(choice, "delta")
                content = _field(delta, delta_key)
                if content is None:
                    continue  # openai v1.0.0 sets content = None for chunks
                if isinstance(content, str):
                    content_list.append(content)

        # Combine the "content" strings into a single string || combine the 'function' strings into a single string
        combined_content: Final = "".join(content_list)

        # Update the "content" field within the response dictionary
        return combined_content

    def get_combined_thinking_content(
        self, chunks: Sequence[Mapping[str, object]]
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
            if not _is_object_list(choices):
                continue
            for choice in choices:
                delta = _field(choice, "delta")
                thinking = _field(delta, "thinking_blocks")
                if thinking and _is_object_list(thinking):
                    for thinking_block in thinking:
                        if not _is_str_keyed_dict(thinking_block):
                            continue
                        thinking_type = thinking_block.get("type", None)
                        if thinking_type and thinking_type == "redacted_thinking":
                            _flush_thinking_block()
                            redacted_data = thinking_block.get("data", None)
                            if isinstance(redacted_data, str) and redacted_data:
                                thinking_blocks.append(
                                    ChatCompletionRedactedThinkingBlock(
                                        type="redacted_thinking",
                                        data=redacted_data,
                                    )
                                )
                        else:
                            thinking_text = thinking_block.get("thinking", None)
                            if isinstance(thinking_text, str) and thinking_text:
                                current_thinking_text_parts.append(thinking_text)
                            signature = thinking_block.get("signature", None)
                            if isinstance(signature, str) and signature:
                                current_signature = signature
                                _flush_thinking_block()

        _flush_thinking_block()

        if len(thinking_blocks) > 0:
            return thinking_blocks
        return None

    def get_combined_reasoning_content(
        self, chunks: Sequence[Mapping[str, object]]
    ) -> ChatCompletionAssistantContentValue:
        return self.get_combined_content(chunks, delta_key="reasoning_content")

    def get_combined_audio_content(self, chunks: Sequence[Mapping[str, object]]) -> ChatCompletionAudioResponse:
        base64_data_list: Final[list[str]] = []
        transcript_list: Final[list[str]] = []
        expires_at: int | None = None
        id: str | None = None

        for chunk in chunks:
            choices = chunk["choices"]
            if not _is_object_list(choices):
                continue
            for choice in choices:
                delta = _field(choice, "delta")
                audio = _field(delta, "audio")
                if _is_str_keyed_dict(audio):
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

    def _usage_chunk_calculation_helper(self, usage_chunk: Usage) -> _UsageChunkFields:
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
            reasoning_content: object = getattr(choice.message, "reasoning_content", None)
            if isinstance(reasoning_content, str):
                if reasoning_tokens is None:
                    reasoning_tokens = 0
                reasoning_tokens += token_counter(
                    text=reasoning_content,
                    count_response_tokens=True,
                )

        return reasoning_tokens

    @staticmethod
    def _extract_usage_chunk(chunk: dict[str, Any] | ModelResponse | ModelResponseStream) -> Usage | None:
        usage_chunk: Usage | dict[str, Any] | None = None
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
        chunks: list[dict[str, object] | ModelResponse],
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
                chunk_prompt_tokens_details = usage_chunk_dict["prompt_tokens_details"]
                if chunk_prompt_tokens_details is not None:
                    web_search_value: int | None = getattr(chunk_prompt_tokens_details, "web_search_requests", None)
                    if web_search_value is not None:
                        web_search_requests = web_search_value

                prompt_tokens_details = chunk_prompt_tokens_details or prompt_tokens_details

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
        chunks: list[dict[str, object] | ModelResponse],
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

        custom_llm_provider: object = None
        if chunks:
            first_chunk: Final = chunks[0]
            if isinstance(first_chunk, dict):
                hp: object = first_chunk.get("_hidden_params")
            else:
                hp = getattr(first_chunk, "_hidden_params", None)
            if _is_str_keyed_dict(hp):
                custom_llm_provider = hp.get("custom_llm_provider")

        if custom_llm_provider == "anthropic" and completion_tokens == 1:
            return 0
        return completion_tokens

    def calculate_usage(
        self,
        chunks: list[dict[str, object] | ModelResponse],
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

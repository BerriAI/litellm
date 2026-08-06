import base64
import time
from collections.abc import Iterator, Mapping, Sequence
from itertools import groupby
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypedDict, Union, cast

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.types.llms.openai import ChatCompletionAssistantContentValue
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
    from litellm.litellm_core_utils.redact_messages import LiteLLMLoggingObject
    from litellm.types.litellm_core_utils.streaming_chunk_builder_utils import (
        UsagePerChunk,
    )
    from litellm.types.llms.openai import (
        ChatCompletionRedactedThinkingBlock,
        ChatCompletionThinkingBlock,
    )


_MAPPING_ADAPTER: Final = TypeAdapter(dict[str, object])
_SEQUENCE_ADAPTER: Final = TypeAdapter(list[object])


def _as_mapping(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    return _MAPPING_ADAPTER.validate_python(value)


def _as_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return _SEQUENCE_ADAPTER.validate_python(value)


def _str_attr(obj: object, name: str) -> str | None:
    value: Final = getattr(obj, name, None)
    return value if isinstance(value, str) else None


def _int_attr(obj: object, name: str) -> int | None:
    value: Final = getattr(obj, name, None)
    return value if isinstance(value, int) else None


def _dict_style_get(obj: object, key: str, default: object = None) -> object:
    getter: Final = getattr(obj, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(obj, key, default)


def _hidden_params_of(chunk: object) -> dict[str, object] | None:
    mapping: Final = _as_mapping(chunk)
    if mapping is not None:
        return _as_mapping(mapping.get("_hidden_params"))
    return _as_mapping(getattr(chunk, "_hidden_params", None))


class _ToolCallAccumulator(TypedDict):
    id: str | None
    name: str | None
    type: str | None
    custom_name: str | None
    provider_specific_fields: dict[str, object] | None


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
        self.first_chunk: Mapping[str, object] = chunks[0]

    def _sort_chunks(self, chunks: list) -> list:
        if not chunks:
            return []

        first_chunk: Final = chunks[0]
        first_hidden_params: Final = _hidden_params_of(first_chunk) or {}

        if first_hidden_params.get("created_at"):

            def _created_at(chunk: object) -> int | float:
                params: Final = _hidden_params_of(chunk)
                if params is None:
                    return float("inf")
                created_at: Final = params.get("created_at", float("inf"))
                return created_at if isinstance(created_at, (int, float)) else float("inf")

            return sorted(chunks, key=_created_at)
        return chunks

    def update_model_response_with_hidden_params(
        self, model_response: ModelResponse, chunk: Mapping[str, object] | None = None
    ) -> ModelResponse:
        if chunk is None:
            return model_response
        # set hidden params from chunk to model_response
        if model_response is not None and hasattr(model_response, "_hidden_params"):
            model_response._hidden_params = _as_mapping(chunk.get("_hidden_params")) or {}
        return model_response

    @staticmethod
    def apply_provider_assembled_streaming_metadata(
        response: ModelResponse,
        chunks: list[object],
        logging_obj: "LiteLLMLoggingObject | None" = None,
    ) -> None:
        if not chunks:
            return

        model: Final = response.model
        if not model:
            return

        provider_value: Final = (
            logging_obj.model_call_details.get("custom_llm_provider") if logging_obj is not None else None
        )
        custom_llm_provider: Final = provider_value if isinstance(provider_value, str) else None

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
    def _get_chunk_id(chunks: list[Mapping[str, object]]) -> str:
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
    def _get_model_from_chunks(chunks: list[Mapping[str, object]], first_chunk_model: str) -> str:
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

    def build_base_response(self, chunks: list[Mapping[str, object]]) -> ModelResponse:
        chunk = self.first_chunk
        id: Final = ChunkProcessor._get_chunk_id(chunks)
        object: Final = chunk["object"]
        created: Final = chunk["created"]
        first_chunk_model_value: Final = chunk["model"]
        first_chunk_model: Final = first_chunk_model_value if isinstance(first_chunk_model_value, str) else ""
        # Get the actual model - for Azure Model Router, this finds the real model from later chunks
        model: Final = ChunkProcessor._get_model_from_chunks(chunks, first_chunk_model)
        system_fingerprint: Final = chunk.get("system_fingerprint", None)

        first_chunk_with_choices: Final = next((c for c in chunks if c.get("choices")), chunk)
        first_role_choices: Final = _as_list(first_chunk_with_choices.get("choices")) or []
        first_role_choice: Final = first_role_choices[0] if first_role_choices else None
        first_role_delta: Final = _dict_style_get(first_role_choice, "delta") if first_role_choice is not None else None
        role: Final = _dict_style_get(first_role_delta, "role") if first_role_delta is not None else None
        finish_reason = "stop"
        for chunk in chunks:
            choices_value = _as_list(chunk.get("choices"))
            if choices_value is None or len(choices_value) == 0:
                continue
            first_choice = choices_value[0]
            if hasattr(first_choice, "finish_reason"):
                chunk_finish_reason = _str_attr(first_choice, "finish_reason")
            else:
                chunk_finish_reason = None
                first_choice_mapping = _as_mapping(first_choice)
                if first_choice_mapping is not None and "finish_reason" in first_choice_mapping:
                    candidate = first_choice_mapping.get("finish_reason")
                    chunk_finish_reason = candidate if isinstance(candidate, str) else None
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
            for choice in _as_list(chunk.get("choices")) or ():
                delta = _dict_style_get(choice, "delta")
                if not delta:
                    continue
                tool_calls = _as_list(_dict_style_get(delta, "tool_calls", ()))
                for tool_call in tool_calls or ():
                    if not tool_call:
                        continue
                    tool_call_mapping = _as_mapping(tool_call)
                    if tool_call_mapping is not None:
                        index_value = tool_call_mapping.get("index", 0)
                        index = index_value if isinstance(index_value, int) else 0
                        function = tool_call_mapping.get("function")
                        function_mapping = _as_mapping(function)
                        if function_mapping is not None:
                            arguments = function_mapping.get("arguments")
                            if isinstance(arguments, str) and arguments:
                                yield index, "arguments", arguments
                        else:
                            arguments = _str_attr(function, "arguments")
                            if arguments:
                                yield index, "arguments", arguments
                        custom = tool_call_mapping.get("custom")
                        custom_mapping = _as_mapping(custom)
                        if custom_mapping is not None:
                            custom_input = custom_mapping.get("input")
                            if isinstance(custom_input, str) and custom_input:
                                yield index, "custom_input", custom_input
                    else:
                        index = _int_attr(tool_call, "index") or 0
                        function = getattr(tool_call, "function", None)
                        arguments = _str_attr(function, "arguments")
                        if arguments:
                            yield index, "arguments", arguments
                        custom = getattr(tool_call, "custom", None)
                        custom_input = _str_attr(custom, "input")
                        if custom_input:
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
        tool_call_map: Final[dict[int, _ToolCallAccumulator]] = {}  # Map to store tool calls by index

        for chunk in tool_call_chunks:
            choices = _as_list(chunk.get("choices")) or ()
            for choice in choices:
                delta = _dict_style_get(choice, "delta", {})
                tool_calls = _as_list(_dict_style_get(delta, "tool_calls", [])) or ()

                for tool_call in tool_calls:
                    # Handle both dict and object formats
                    if not tool_call:
                        continue

                    tool_call_mapping = _as_mapping(tool_call)

                    # Check if tool_call has function (either as attribute or dict key)
                    has_function = False
                    has_custom = False
                    if tool_call_mapping is not None:
                        has_function = "function" in tool_call_mapping and tool_call_mapping["function"] is not None
                        has_custom = "custom" in tool_call_mapping and tool_call_mapping["custom"] is not None
                    else:
                        has_function = (
                            hasattr(tool_call, "function") and getattr(tool_call, "function", None) is not None
                        )
                        has_custom = getattr(tool_call, "custom", None) is not None

                    if not has_function and not has_custom:
                        continue

                    # Get index (handle both dict and object)
                    if tool_call_mapping is not None:
                        index_value = tool_call_mapping.get("index", 0)
                    else:
                        index_value = getattr(tool_call, "index", 0)
                    index = index_value if isinstance(index_value, int) else 0

                    if index not in tool_call_map:
                        tool_call_map[index] = {
                            "id": None,
                            "name": None,
                            "type": None,
                            "custom_name": None,
                            "provider_specific_fields": None,
                        }

                    # Extract id, type, and function data (handle both dict and object)
                    if tool_call_mapping is not None:
                        tool_call_id = tool_call_mapping.get("id")
                        if isinstance(tool_call_id, str) and tool_call_id:
                            tool_call_map[index]["id"] = tool_call_id
                        tool_call_type = tool_call_mapping.get("type")
                        if isinstance(tool_call_type, str) and tool_call_type:
                            tool_call_map[index]["type"] = tool_call_type

                        function = tool_call_mapping.get("function", {})
                        function_mapping = _as_mapping(function)
                        if function_mapping is not None:
                            function_name = function_mapping.get("name")
                            if isinstance(function_name, str) and function_name:
                                tool_call_map[index]["name"] = function_name
                        else:
                            # function is an object
                            function_name = _str_attr(function, "name")
                            if function_name:
                                tool_call_map[index]["name"] = function_name

                        custom = tool_call_mapping.get("custom")
                        custom_mapping = _as_mapping(custom)
                        if custom_mapping is not None:
                            custom_name = custom_mapping.get("name")
                            if isinstance(custom_name, str) and custom_name:
                                tool_call_map[index]["custom_name"] = custom_name
                    else:
                        # tool_call is an object
                        tool_call_id = _str_attr(tool_call, "id")
                        if tool_call_id:
                            tool_call_map[index]["id"] = tool_call_id
                        tool_call_type = _str_attr(tool_call, "type")
                        if tool_call_type:
                            tool_call_map[index]["type"] = tool_call_type
                        if hasattr(tool_call, "function"):
                            tool_call_function = getattr(tool_call, "function", None)
                            function_name = _str_attr(tool_call_function, "name")
                            if function_name:
                                tool_call_map[index]["name"] = function_name

                        custom_name = _str_attr(getattr(tool_call, "custom", None), "name")
                        if custom_name:
                            tool_call_map[index]["custom_name"] = custom_name

                    # Preserve provider_specific_fields from streaming chunks
                    provider_fields: object = None
                    if tool_call_mapping is not None:
                        provider_fields = tool_call_mapping.get("provider_specific_fields")
                        if not provider_fields:
                            nested_function_mapping = _as_mapping(tool_call_mapping.get("function"))
                            if nested_function_mapping is not None:
                                provider_fields = nested_function_mapping.get("provider_specific_fields")
                    else:
                        if hasattr(tool_call, "provider_specific_fields") and getattr(
                            tool_call, "provider_specific_fields", None
                        ):
                            provider_fields = getattr(tool_call, "provider_specific_fields", None)
                        else:
                            tool_call_function = getattr(tool_call, "function", None)
                            if (
                                hasattr(tool_call, "function")
                                and hasattr(tool_call_function, "provider_specific_fields")
                                and getattr(tool_call_function, "provider_specific_fields", None)
                            ):
                                provider_fields = getattr(tool_call_function, "provider_specific_fields", None)

                    if provider_fields:
                        # Merge provider_specific_fields if multiple chunks have them
                        provider_fields_mapping = _as_mapping(provider_fields)
                        if provider_fields_mapping is not None:
                            existing_fields = tool_call_map[index]["provider_specific_fields"] or {}
                            tool_call_map[index]["provider_specific_fields"] = {
                                **existing_fields,
                                **provider_fields_mapping,
                            }

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

                # Add provider_specific_fields if present (for thought signatures in Gemini 3)
                provider_specific_fields = tool_call_data.get("provider_specific_fields")
                if provider_specific_fields:
                    tool_call = ChatCompletionMessageToolCall(
                        id=tool_call_data["id"],
                        function=function,
                        type=tool_call_data["type"] or "function",
                        provider_specific_fields=provider_specific_fields,
                    )
                else:
                    tool_call = ChatCompletionMessageToolCall(
                        id=tool_call_data["id"],
                        function=function,
                        type=tool_call_data["type"] or "function",
                    )
                tool_calls_list.append(tool_call)

        return tool_calls_list

    def get_combined_function_call_content(self, function_call_chunks: list[Mapping[str, object]]) -> FunctionCall:
        argument_list: Final[list[str]] = []
        first_choices: Final = _as_list(function_call_chunks[0].get("choices")) or []
        first_choice: Final = first_choices[0] if first_choices else None
        first_delta: Final = _dict_style_get(first_choice, "delta") if first_choice is not None else None
        first_function_call: Final = _dict_style_get(first_delta, "function_call") if first_delta is not None else None
        function_call_name: Final = _str_attr(first_function_call, "name")

        for chunk in function_call_chunks:
            choices = _as_list(chunk.get("choices")) or ()
            for choice in choices:
                delta = _dict_style_get(choice, "delta", {})
                function_call = _dict_style_get(delta, "function_call", "")

                # Check if a function call is present
                if function_call:
                    # Now, function_call is expected to be a dictionary
                    arguments = _str_attr(function_call, "arguments")
                    if arguments is not None:
                        argument_list.append(arguments)

        combined_arguments: Final = "".join(argument_list)

        return FunctionCall(
            name=function_call_name,
            arguments=combined_arguments,
        )

    def get_combined_content(
        self, chunks: list[Mapping[str, object]], delta_key: str = "content"
    ) -> ChatCompletionAssistantContentValue:
        content_list: Final[list[str]] = []
        for chunk in chunks:
            choices = _as_list(chunk.get("choices")) or ()
            for choice in choices:
                delta = _dict_style_get(choice, "delta", {})
                content = _dict_style_get(delta, delta_key, "")
                if content is None:
                    continue  # openai v1.0.0 sets content = None for chunks
                if isinstance(content, str):
                    content_list.append(content)

        # Combine the "content" strings into a single string || combine the 'function' strings into a single string
        combined_content: Final = "".join(content_list)

        # Update the "content" field within the response dictionary
        return combined_content

    def get_combined_thinking_content(
        self, chunks: list[Mapping[str, object]]
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
            choices = _as_list(chunk.get("choices")) or ()
            for choice in choices:
                delta = _dict_style_get(choice, "delta", {})
                thinking = _as_list(_dict_style_get(delta, "thinking_blocks", None))
                if thinking:
                    for thinking_block_value in thinking:
                        thinking_block = _as_mapping(thinking_block_value)
                        if thinking_block is None:
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

    def get_combined_reasoning_content(self, chunks: list[Mapping[str, object]]) -> ChatCompletionAssistantContentValue:
        return self.get_combined_content(chunks, delta_key="reasoning_content")

    def get_combined_audio_content(self, chunks: list[Mapping[str, object]]) -> ChatCompletionAudioResponse:
        base64_data_list: Final[list[str]] = []
        transcript_list: Final[list[str]] = []
        expires_at: int | None = None
        id: str | None = None

        for chunk in chunks:
            choices = _as_list(chunk.get("choices")) or ()
            for choice in choices:
                delta = _dict_style_get(choice, "delta", {})
                audio = _as_mapping(_dict_style_get(delta, "audio"))
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

    def _usage_chunk_calculation_helper(self, usage_chunk: Usage) -> dict:
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
    def _extract_usage_chunk(chunk: Mapping[str, object] | ModelResponse | ModelResponseStream) -> Usage | None:
        def _usage_value() -> object:
            if hasattr(chunk, "usage") and getattr(chunk, "usage", None) is not None:
                return getattr(chunk, "usage", None)
            if "usage" in chunk:
                return chunk.get("usage")
            if (isinstance(chunk, ModelResponse) or isinstance(chunk, ModelResponseStream)) and hasattr(
                chunk, "_hidden_params"
            ):
                return chunk._hidden_params.get("usage", None)
            return None

        usage_value: Final = _usage_value()
        if isinstance(usage_value, Usage):
            return usage_value
        if isinstance(usage_value, dict):
            return Usage(**usage_value)
        return None

    def _calculate_usage_per_chunk(
        self,
        chunks: list[Mapping[str, object] | ModelResponse],
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
                if usage_chunk_dict["prompt_tokens_details"] is not None:
                    candidate_web_search_requests = _int_attr(
                        usage_chunk_dict["prompt_tokens_details"], "web_search_requests"
                    )
                    if candidate_web_search_requests is not None:
                        web_search_requests = candidate_web_search_requests

                prompt_tokens_details = (
                    cast(
                        PromptTokensDetailsWrapper | None,
                        usage_chunk_dict["prompt_tokens_details"],
                    )
                    or prompt_tokens_details
                )

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
        chunks: list[Mapping[str, object] | ModelResponse],
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

        first_chunk: Final = chunks[0] if chunks else None
        hp: Final = (
            (
                first_chunk.get("_hidden_params")
                if isinstance(first_chunk, dict)
                else getattr(first_chunk, "_hidden_params", None)
            )
            if first_chunk is not None
            else None
        )
        provider_value: Final = hp.get("custom_llm_provider") if isinstance(hp, dict) else None
        custom_llm_provider: Final = provider_value if isinstance(provider_value, str) else None

        if custom_llm_provider == "anthropic" and completion_tokens == 1:
            return 0
        return completion_tokens

    def calculate_usage(
        self,
        chunks: list[Mapping[str, object] | ModelResponse],
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
                returned_usage.completion_tokens_details = CompletionTokensDetailsWrapper(
                    accepted_prediction_tokens=completion_tokens_details.accepted_prediction_tokens,
                    audio_tokens=completion_tokens_details.audio_tokens,
                    reasoning_tokens=completion_tokens_details.reasoning_tokens,
                    rejected_prediction_tokens=completion_tokens_details.rejected_prediction_tokens,
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

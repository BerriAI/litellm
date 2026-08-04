import base64
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from pydantic import TypeAdapter
from typing_extensions import Required, TypedDict

from litellm._logging import verbose_logger
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionAssistantContentValue,
    ChatCompletionAudioDelta,
    ChatCompletionRedactedThinkingBlock,
    ChatCompletionThinkingBlock,
)
from litellm.types.utils import (
    CacheCreationTokenDetails,
    ChatCompletionAudioResponse,
    ChatCompletionDeltaToolCall,
    ChatCompletionMessageToolCall,
    CompletionTokensDetails,
    CompletionTokensDetailsWrapper,
    Function,
    FunctionCall,
    ModelResponse,
    ModelResponseStream,
    PromptTokensDetailsWrapper,
    ServerToolUse,
    StreamingChoices,
    Usage,
)
from litellm.utils import print_verbose, token_counter

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.types.litellm_core_utils.streaming_chunk_builder_utils import (
        UsagePerChunk,
    )


RawHiddenParamsDict = dict[str, object]


class ChunkHiddenParamsDict(TypedDict, total=False):
    created_at: int | float
    custom_llm_provider: str | None


class UsageDict(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    cost: float | None
    prompt_tokens_details: dict[str, object] | None
    completion_tokens_details: dict[str, object] | None
    server_tool_use: dict[str, object] | None


class ToolCallFunctionDict(TypedDict, total=False):
    name: str | None
    arguments: str
    provider_specific_fields: dict[str, object] | None


class ToolCallDict(TypedDict, total=False):
    id: str | None
    type: str | None
    index: int
    function: ToolCallFunctionDict | Function | None
    provider_specific_fields: dict[str, object] | None


class DeltaDict(TypedDict, total=False):
    role: str | None
    content: str | None
    reasoning_content: str | None
    function_call: FunctionCall | None
    tool_calls: list[ToolCallDict | ChatCompletionDeltaToolCall] | None
    audio: ChatCompletionAudioDelta | None
    thinking_blocks: list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] | None


class ChoiceDict(TypedDict, total=False):
    index: int
    delta: Required[DeltaDict]
    finish_reason: str | None


class ChunkDict(TypedDict, total=False):
    id: Required[str]
    object: Required[str]
    created: Required[int]
    model: Required[str]
    choices: Required[list[ChoiceDict | StreamingChoices]]
    system_fingerprint: str | None
    usage: Usage | UsageDict | None
    _hidden_params: RawHiddenParamsDict


class AccumulatedToolCallDict(TypedDict):
    id: str | None
    name: str | None
    type: str | None
    arguments: list[str]
    provider_specific_fields: dict[str, object] | None


class UsageChunkCalculationDict(TypedDict):
    prompt_tokens: int | None
    completion_tokens: int | None
    cache_creation_input_tokens: int | None
    cache_read_input_tokens: int | None
    completion_tokens_details: CompletionTokensDetails | None
    prompt_tokens_details: PromptTokensDetailsWrapper | None
    cost: float | None


class CompletionTokensDetailsDumpDict(TypedDict, total=False):
    accepted_prediction_tokens: int | None
    audio_tokens: int | None
    reasoning_tokens: int | None
    rejected_prediction_tokens: int | None
    text_tokens: int | None
    image_tokens: int | None
    video_tokens: int | None


class UsageDumpDict(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    completion_tokens_details: dict[str, object] | None
    prompt_tokens_details: dict[str, object] | None
    server_tool_use: dict[str, object] | None
    cost: float | None
    cache_creation_input_tokens: int
    cache_read_input_tokens: int


_CHUNK_HIDDEN_PARAMS_ADAPTER: TypeAdapter[ChunkHiddenParamsDict] = TypeAdapter(ChunkHiddenParamsDict)
_AUDIO_DELTA_ADAPTER: TypeAdapter[ChatCompletionAudioDelta | None] = TypeAdapter(ChatCompletionAudioDelta | None)
_COMPLETION_TOKENS_DETAILS_DUMP_ADAPTER: TypeAdapter[CompletionTokensDetailsDumpDict] = TypeAdapter(
    CompletionTokensDetailsDumpDict
)
_USAGE_DUMP_ADAPTER: TypeAdapter[UsageDumpDict] = TypeAdapter(UsageDumpDict)


class ChunkProcessor:
    def __init__(self, chunks: list[ChunkDict], messages: list[AllMessageValues] | None = None):
        self.chunks = self._sort_chunks(chunks)
        self.messages = messages
        self.first_chunk = chunks[0]

    @staticmethod
    def _chunk_hidden_params(chunk: "ChunkDict | ModelResponseStream") -> ChunkHiddenParamsDict:
        candidate: object = (
            chunk.get("_hidden_params", {}) if isinstance(chunk, dict) else getattr(chunk, "_hidden_params", {})
        )
        if isinstance(candidate, dict):
            return _CHUNK_HIDDEN_PARAMS_ADAPTER.validate_python(candidate)
        return {}

    @staticmethod
    def _chunk_created_at(chunk: "ChunkDict | ModelResponseStream") -> int | float:
        return ChunkProcessor._chunk_hidden_params(chunk).get("created_at", float("inf"))

    def _sort_chunks(self, chunks: list[ChunkDict]) -> list[ChunkDict]:
        if not chunks:
            return []

        first_hidden_params = self._chunk_hidden_params(chunks[0])
        if first_hidden_params.get("created_at"):
            return sorted(chunks, key=self._chunk_created_at)
        return chunks

    def update_model_response_with_hidden_params(
        self, model_response: ModelResponse, chunk: ChunkDict | None = None
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
        chunks: list[ChunkDict],
        logging_obj: "Logging | None" = None,
    ) -> None:
        if not chunks:
            return

        model: str | None = getattr(response, "model", None)
        if not model:
            return

        custom_llm_provider: str | None = None
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

            provider_config = ProviderConfigManager.get_provider_chat_config(
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
    def _get_chunk_id(chunks: Sequence[ChunkDict]) -> str:
        """
        Chunks:
        [{"id": ""}, {"id": "1"}, {"id": "1"}]
        """
        for chunk in chunks:
            if chunk.get("id"):
                return chunk["id"]
        return ""

    @staticmethod
    def _get_model_from_chunks(chunks: Sequence[ChunkDict], first_chunk_model: str) -> str:
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

    def build_base_response(self, chunks: Sequence[ChunkDict]) -> ModelResponse:
        chunk = self.first_chunk
        id = ChunkProcessor._get_chunk_id(chunks)
        object = chunk["object"]
        created = chunk["created"]
        first_chunk_model = chunk["model"]
        # Get the actual model - for Azure Model Router, this finds the real model from later chunks
        model = ChunkProcessor._get_model_from_chunks(chunks, first_chunk_model)
        system_fingerprint = chunk.get("system_fingerprint", None)

        first_chunk_with_choices = next((c for c in chunks if c.get("choices")), chunk)
        first_choice = first_chunk_with_choices["choices"][0]
        if isinstance(first_choice, dict):
            role = first_choice["delta"].get("role")
        else:
            role = first_choice.delta.role
        finish_reason = "stop"
        for chunk in chunks:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                finish_reason_choice = chunk["choices"][0]
                if isinstance(finish_reason_choice, dict):
                    chunk_finish_reason = finish_reason_choice.get("finish_reason")
                else:
                    chunk_finish_reason = finish_reason_choice.finish_reason
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

    def get_combined_tool_content(self, tool_call_chunks: Sequence[ChunkDict]) -> list[ChatCompletionMessageToolCall]:
        tool_calls_list: list[ChatCompletionMessageToolCall] = []
        tool_call_map: dict[int, AccumulatedToolCallDict] = {}  # Map to store tool calls by index

        for chunk in tool_call_chunks:
            choices = chunk["choices"]
            for choice in choices:
                tool_calls: Sequence[ToolCallDict | ChatCompletionDeltaToolCall]
                if isinstance(choice, dict):
                    tool_calls = choice.get("delta", {}).get("tool_calls") or []
                else:
                    tool_calls = choice.delta.tool_calls or []

                for tool_call in tool_calls:
                    # Handle both dict and object formats
                    if not tool_call:
                        continue

                    # Check if tool_call has function (either as attribute or dict key)
                    has_function = False
                    if isinstance(tool_call, dict):
                        has_function = "function" in tool_call and tool_call["function"] is not None
                    else:
                        has_function = getattr(tool_call, "function", None) is not None

                    if not has_function:
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
                            "arguments": [],
                            "provider_specific_fields": None,
                        }

                    # Extract id, type, and function data (handle both dict and object)
                    if isinstance(tool_call, dict):
                        tool_call_id = tool_call.get("id")
                        if tool_call_id:
                            tool_call_map[index]["id"] = tool_call_id
                        tool_call_type = tool_call.get("type")
                        if tool_call_type:
                            tool_call_map[index]["type"] = tool_call_type

                        function = tool_call.get("function", {})
                        if isinstance(function, dict):
                            function_name = function.get("name")
                            if function_name:
                                tool_call_map[index]["name"] = function_name
                            function_arguments = function.get("arguments")
                            if function_arguments:
                                tool_call_map[index]["arguments"].append(function_arguments)
                        else:
                            # function is an object
                            if function is not None and hasattr(function, "name") and function.name:
                                tool_call_map[index]["name"] = function.name
                            if function is not None and hasattr(function, "arguments") and function.arguments:
                                tool_call_map[index]["arguments"].append(function.arguments)
                    else:
                        # tool_call is an object
                        if hasattr(tool_call, "id") and tool_call.id:
                            tool_call_map[index]["id"] = tool_call.id
                        if hasattr(tool_call, "type") and tool_call.type:
                            tool_call_map[index]["type"] = tool_call.type
                        if hasattr(tool_call, "function"):
                            if hasattr(tool_call.function, "name") and tool_call.function.name:
                                tool_call_map[index]["name"] = tool_call.function.name
                            if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                tool_call_map[index]["arguments"].append(tool_call.function.arguments)

                    # Preserve provider_specific_fields from streaming chunks
                    provider_fields: dict[str, object] | None = None
                    if isinstance(tool_call, dict):
                        provider_fields = tool_call.get("provider_specific_fields")
                        if not provider_fields:
                            function_value = tool_call.get("function")
                            if isinstance(function_value, dict):
                                provider_fields = function_value.get("provider_specific_fields")
                    else:
                        object_provider_fields = getattr(tool_call, "provider_specific_fields", None) or getattr(
                            getattr(tool_call, "function", None), "provider_specific_fields", None
                        )
                        if isinstance(object_provider_fields, dict) and object_provider_fields:
                            provider_fields = object_provider_fields

                    if isinstance(provider_fields, dict) and provider_fields:
                        # Merge provider_specific_fields if multiple chunks have them
                        existing_fields = tool_call_map[index]["provider_specific_fields"] or {}
                        tool_call_map[index]["provider_specific_fields"] = {**existing_fields, **provider_fields}

        # Convert the map to a list of tool calls
        for index in sorted(tool_call_map.keys()):
            tool_call_data = tool_call_map[index]
            if tool_call_data["id"] and tool_call_data["name"]:
                combined_arguments = "".join(tool_call_data["arguments"]) or "{}"

                # Build function - provider_specific_fields should be on tool_call level, not function level
                function = Function(
                    arguments=combined_arguments,
                    name=tool_call_data["name"],
                )

                # Add provider_specific_fields if present (for thought signatures in Gemini 3)
                provider_specific_fields = tool_call_data["provider_specific_fields"]
                if provider_specific_fields:
                    tool_calls_list.append(
                        ChatCompletionMessageToolCall(
                            id=tool_call_data["id"],
                            function=function,
                            type=tool_call_data["type"] or "function",
                            provider_specific_fields=provider_specific_fields,
                        )
                    )
                else:
                    tool_calls_list.append(
                        ChatCompletionMessageToolCall(
                            id=tool_call_data["id"],
                            function=function,
                            type=tool_call_data["type"] or "function",
                        )
                    )

        return tool_calls_list

    def get_combined_function_call_content(self, function_call_chunks: Sequence[ChunkDict]) -> FunctionCall:
        argument_list: list[str] = []
        first_choice = function_call_chunks[0]["choices"][0]
        if isinstance(first_choice, dict):
            first_function_call = first_choice["delta"].get("function_call")
        else:
            first_function_call = getattr(first_choice.delta, "function_call", None)
        function_call_name = first_function_call.name

        for chunk in function_call_chunks:
            choices = chunk["choices"]
            for choice in choices:
                if isinstance(choice, dict):
                    function_call = choice.get("delta", {}).get("function_call")
                else:
                    function_call = getattr(choice.delta, "function_call", None)

                # Check if a function call is present
                if function_call:
                    argument_list.append(function_call.arguments)

        combined_arguments = "".join(argument_list)

        return FunctionCall(
            name=function_call_name,
            arguments=combined_arguments,
        )

    def get_combined_content(
        self, chunks: Sequence[ChunkDict], delta_key: Literal["content", "reasoning_content"] = "content"
    ) -> ChatCompletionAssistantContentValue:
        content_list: list[str] = []
        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                if isinstance(choice, dict):
                    content = choice.get("delta", {}).get(delta_key, "")
                else:
                    content = getattr(choice.delta, delta_key, "")
                if content is None:
                    continue  # openai v1.0.0 sets content = None for chunks
                content_list.append(content)

        # Combine the "content" strings into a single string || combine the 'function' strings into a single string
        combined_content = "".join(content_list)

        # Update the "content" field within the response dictionary
        return combined_content

    def get_combined_thinking_content(
        self, chunks: Sequence[ChunkDict]
    ) -> list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] | None:
        thinking_blocks: list[ChatCompletionThinkingBlock | ChatCompletionRedactedThinkingBlock] = []
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
                if isinstance(choice, dict):
                    thinking = choice.get("delta", {}).get("thinking_blocks", None)
                else:
                    thinking = getattr(choice.delta, "thinking_blocks", None)
                if isinstance(thinking, list) and thinking:
                    for thinking_block in thinking:
                        if thinking_block.get("type", None) == "redacted_thinking":
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

    def get_combined_reasoning_content(self, chunks: Sequence[ChunkDict]) -> ChatCompletionAssistantContentValue:
        return self.get_combined_content(chunks, delta_key="reasoning_content")

    def get_combined_audio_content(self, chunks: Sequence[ChunkDict]) -> ChatCompletionAudioResponse:
        base64_data_list: list[str] = []
        transcript_list: list[str] = []
        expires_at: int | None = None
        id: str | None = None

        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                audio: ChatCompletionAudioDelta | None
                if isinstance(choice, dict):
                    delta: DeltaDict = choice.get("delta") or {}
                    audio = delta.get("audio")
                else:
                    audio = _AUDIO_DELTA_ADAPTER.validate_python(getattr(choice.delta, "audio", None))
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

        concatenated_audio = concatenate_base64_list(base64_data_list)
        return ChatCompletionAudioResponse(
            data=concatenated_audio,
            expires_at=expires_at or int(time.time() + 3600),
            transcript="".join(transcript_list),
            id=id,
        )

    def _usage_chunk_calculation_helper(self, usage_chunk: Usage) -> UsageChunkCalculationDict:
        prompt_tokens: int = 0
        completion_tokens: int = 0
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
            if hasattr(choice.message, "reasoning_content") and choice.message.reasoning_content is not None:
                if reasoning_tokens is None:
                    reasoning_tokens = 0
                reasoning_tokens += token_counter(
                    text=choice.message.reasoning_content,
                    count_response_tokens=True,
                )

        return reasoning_tokens

    @staticmethod
    def _extract_usage_chunk(chunk: ChunkDict | ModelResponse | ModelResponseStream) -> Usage | None:
        usage_chunk: Usage | UsageDict | None = None
        if isinstance(chunk, dict):
            usage_chunk = chunk.get("usage")
        elif hasattr(chunk, "usage"):
            usage_chunk = getattr(chunk, "usage", None)
        elif hasattr(chunk, "_hidden_params"):
            hidden_usage: object = chunk._hidden_params.get("usage", None)
            if isinstance(hidden_usage, Usage):
                return hidden_usage
            if isinstance(hidden_usage, dict):
                return Usage(**hidden_usage)
            return None

        if isinstance(usage_chunk, dict):
            return Usage(**usage_chunk)
        return usage_chunk

    def _calculate_usage_per_chunk(
        self,
        chunks: Sequence[ChunkDict],
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
                web_search_requests_value: int | None = getattr(
                    usage_chunk_dict["prompt_tokens_details"],
                    "web_search_requests",
                    None,
                )
                if usage_chunk_dict["prompt_tokens_details"] is not None and web_search_requests_value is not None:
                    web_search_requests = web_search_requests_value

                prompt_tokens_details = usage_chunk_dict["prompt_tokens_details"]

                cache_creation_token_details = self._capture_cache_creation_token_details(
                    prompt_tokens_details, cache_creation_token_details
                )

                if usage_chunk_dict["cost"] is not None:
                    cost = usage_chunk_dict["cost"]

        prompt_tokens_details = self._attach_cache_creation_token_details(
            prompt_tokens_details, cache_creation_token_details
        )

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
    def _capture_cache_creation_token_details(
        prompt_tokens_details: PromptTokensDetailsWrapper | None,
        current: CacheCreationTokenDetails | None,
    ) -> CacheCreationTokenDetails | None:
        if prompt_tokens_details is None:
            return current
        incoming: CacheCreationTokenDetails | None = getattr(
            prompt_tokens_details, "cache_creation_token_details", None
        )
        if incoming is not None:
            return incoming
        return current

    @staticmethod
    def _attach_cache_creation_token_details(
        prompt_tokens_details: PromptTokensDetailsWrapper | None,
        cache_creation_token_details: CacheCreationTokenDetails | None,
    ) -> PromptTokensDetailsWrapper | None:
        if prompt_tokens_details is None or cache_creation_token_details is None:
            return prompt_tokens_details
        existing: CacheCreationTokenDetails | None = getattr(
            prompt_tokens_details, "cache_creation_token_details", None
        )
        if existing is not None:
            return prompt_tokens_details
        return prompt_tokens_details.model_copy(update={"cache_creation_token_details": cache_creation_token_details})

    @staticmethod
    def _reset_anthropic_cursor_completion_tokens(
        chunks: Sequence[ChunkDict],
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
        saw_non_cursor_completion = completion_tokens > 1 or completion_usage_updates >= 2
        if saw_non_cursor_completion:
            return completion_tokens

        custom_llm_provider: str | None = None
        if chunks:
            custom_llm_provider = ChunkProcessor._chunk_hidden_params(chunks[0]).get("custom_llm_provider")

        if custom_llm_provider == "anthropic" and completion_tokens == 1:
            return 0
        return completion_tokens

    def calculate_usage(
        self,
        chunks: Sequence[ChunkDict],
        model: str,
        completion_output: str,
        messages: list[AllMessageValues] | None = None,
        reasoning_tokens: int | None = None,
    ) -> Usage:
        """
        Calculate usage for the given chunks.
        """
        returned_usage = Usage()
        # # Update usage information if needed

        calculated_usage_per_chunk = self._calculate_usage_per_chunk(chunks=chunks)
        prompt_tokens = calculated_usage_per_chunk["prompt_tokens"]
        completion_tokens = calculated_usage_per_chunk["completion_tokens"]
        ## anthropic prompt caching information ##
        cache_creation_input_tokens: int | None = calculated_usage_per_chunk["cache_creation_input_tokens"]
        cache_read_input_tokens: int | None = calculated_usage_per_chunk["cache_read_input_tokens"]

        server_tool_use: ServerToolUse | None = calculated_usage_per_chunk["server_tool_use"]
        web_search_requests: int | None = calculated_usage_per_chunk["web_search_requests"]
        completion_tokens_details: CompletionTokensDetails | None = calculated_usage_per_chunk[
            "completion_tokens_details"
        ]
        prompt_tokens_details: PromptTokensDetailsWrapper | None = calculated_usage_per_chunk["prompt_tokens_details"]
        cost: float | None = calculated_usage_per_chunk["cost"]

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
                    **_COMPLETION_TOKENS_DETAILS_DUMP_ADAPTER.validate_python(completion_tokens_details.model_dump())
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

        returned_usage = Usage(**_USAGE_DUMP_ADAPTER.validate_python(returned_usage.model_dump()))

        return returned_usage


def concatenate_base64_list(base64_strings: Sequence[str]) -> str:
    """
    Concatenates a list of base64-encoded strings.

    Args:
        base64_strings (List[str]): A list of base64 strings to concatenate.

    Returns:
        str: The concatenated result as a base64-encoded string.
    """
    # Decode each base64 string and collect the resulting bytes
    combined_bytes = b"".join(base64.b64decode(b64_str) for b64_str in base64_strings)

    # Encode the concatenated bytes back to base64
    return base64.b64encode(combined_bytes).decode("utf-8")

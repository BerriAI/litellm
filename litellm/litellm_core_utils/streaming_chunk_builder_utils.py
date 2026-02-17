import base64
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union, cast

from litellm.types.llms.openai import (
    ChatCompletionAssistantContentValue,
    ChatCompletionAudioDelta,
)
from litellm.types.utils import (
    ChatCompletionAudioResponse,
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
    from litellm.types.litellm_core_utils.streaming_chunk_builder_utils import (
        UsagePerChunk,
    )
    from litellm.types.llms.openai import (
        ChatCompletionRedactedThinkingBlock,
        ChatCompletionThinkingBlock,
    )


class ChunkProcessor:
    def __init__(self, chunks: List, messages: Optional[list] = None):
        self.chunks = self._sort_chunks(chunks)
        self.messages = messages
        self.first_chunk = chunks[0]

    def _sort_chunks(self, chunks: list) -> list:
        if not chunks:
            return []
        if chunks[0]._hidden_params.get("created_at"):
            return sorted(
                chunks, key=lambda x: x._hidden_params.get("created_at", float("inf"))
            )
        return chunks

    def update_model_response_with_hidden_params(
        self, model_response: ModelResponse, chunk: Optional[Dict[str, Any]] = None
    ) -> ModelResponse:
        if chunk is None:
            return model_response
        # set hidden params from chunk to model_response
        if model_response is not None and hasattr(model_response, "_hidden_params"):
            model_response._hidden_params = chunk.get("_hidden_params", {})
        return model_response

    @staticmethod
    def _get_chunk_id(chunks: List[Dict[str, Any]]) -> str:
        """
        Chunks:
        [{"id": ""}, {"id": "1"}, {"id": "1"}]
        """
        for chunk in chunks:
            if chunk.get("id"):
                return chunk["id"]
        return ""

    @staticmethod
    def _get_model_from_chunks(chunks: List[Dict[str, Any]], first_chunk_model: str) -> str:
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

    def build_base_response(self, chunks: List[Dict[str, Any]]) -> ModelResponse:
        chunk = self.first_chunk
        id = ChunkProcessor._get_chunk_id(chunks)
        object = chunk["object"]
        created = chunk["created"]
        first_chunk_model = chunk["model"]
        # Get the actual model - for Azure Model Router, this finds the real model from later chunks
        model = ChunkProcessor._get_model_from_chunks(chunks, first_chunk_model)
        system_fingerprint = chunk.get("system_fingerprint", None)

        role = chunk["choices"][0]["delta"]["role"]
        finish_reason = "stop"
        for chunk in chunks:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                if hasattr(chunk["choices"][0], "finish_reason"):
                    finish_reason = chunk["choices"][0].finish_reason
                elif "finish_reason" in chunk["choices"][0]:
                    finish_reason = chunk["choices"][0]["finish_reason"]

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

        response = self.update_model_response_with_hidden_params(
            model_response=response, chunk=chunk
        )
        return response

    def get_combined_tool_content( # noqa: PLR0915
        self, tool_call_chunks: List[Dict[str, Any]]
    ) -> List[ChatCompletionMessageToolCall]:
        tool_calls_list: List[ChatCompletionMessageToolCall] = []
        tool_call_map: Dict[int, Dict[str, Any]] = (
            {}
        )  # Map to store tool calls by index

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
                    if isinstance(tool_call, dict):
                        has_function = "function" in tool_call and tool_call["function"] is not None
                    else:
                        has_function = hasattr(tool_call, "function") and tool_call.function is not None
                    
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
                        if tool_call.get("id"):
                            tool_call_map[index]["id"] = tool_call["id"]
                        if tool_call.get("type"):
                            tool_call_map[index]["type"] = tool_call["type"]
                        
                        function = tool_call.get("function", {})
                        if isinstance(function, dict):
                            if function.get("name"):
                                tool_call_map[index]["name"] = function["name"]
                            if function.get("arguments"):
                                tool_call_map[index]["arguments"].append(function["arguments"])
                        else:
                            # function is an object
                            if hasattr(function, "name") and function.name:
                                tool_call_map[index]["name"] = function.name
                            if hasattr(function, "arguments") and function.arguments:
                                tool_call_map[index]["arguments"].append(function.arguments)
                    else:
                        # tool_call is an object
                        if hasattr(tool_call, "id") and tool_call.id:
                            tool_call_map[index]["id"] = tool_call.id
                        if hasattr(tool_call, "type") and tool_call.type:
                            tool_call_map[index]["type"] = tool_call.type
                        if hasattr(tool_call, "function"):
                            if (
                                hasattr(tool_call.function, "name")
                                and tool_call.function.name
                            ):
                                tool_call_map[index]["name"] = tool_call.function.name
                            if (
                                hasattr(tool_call.function, "arguments")
                                and tool_call.function.arguments
                            ):
                                tool_call_map[index]["arguments"].append(
                                    tool_call.function.arguments
                                )
                    
                    # Preserve provider_specific_fields from streaming chunks
                    provider_fields = None
                    if isinstance(tool_call, dict):
                        provider_fields = tool_call.get("provider_specific_fields")
                        if not provider_fields and isinstance(tool_call.get("function"), dict):
                            provider_fields = tool_call["function"].get("provider_specific_fields")
                    else:
                        if hasattr(tool_call, "provider_specific_fields") and tool_call.provider_specific_fields:
                            provider_fields = tool_call.provider_specific_fields
                        elif hasattr(tool_call, "function") and hasattr(tool_call.function, "provider_specific_fields") and tool_call.function.provider_specific_fields:
                            provider_fields = tool_call.function.provider_specific_fields
                    
                    if provider_fields:
                        # Merge provider_specific_fields if multiple chunks have them
                        if tool_call_map[index]["provider_specific_fields"] is None:
                            tool_call_map[index]["provider_specific_fields"] = {}
                        if isinstance(provider_fields, dict):
                            tool_call_map[index]["provider_specific_fields"].update(
                                provider_fields
                            )

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


    def get_combined_function_call_content(
        self, function_call_chunks: List[Dict[str, Any]]
    ) -> FunctionCall:
        argument_list = []
        delta = function_call_chunks[0]["choices"][0]["delta"]
        function_call = delta.get("function_call", "")
        function_call_name = function_call.name

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

        combined_arguments = "".join(argument_list)

        return FunctionCall(
            name=function_call_name,
            arguments=combined_arguments,
        )

    def get_combined_content(
        self, chunks: List[Dict[str, Any]], delta_key: str = "content"
    ) -> ChatCompletionAssistantContentValue:
        content_list: List[str] = []
        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta", {})
                content = delta.get(delta_key, "")
                if content is None:
                    continue  # openai v1.0.0 sets content = None for chunks
                content_list.append(content)

        # Combine the "content" strings into a single string || combine the 'function' strings into a single string
        combined_content = "".join(content_list)

        # Update the "content" field within the response dictionary
        return combined_content

    def get_combined_thinking_content(
        self, chunks: List[Dict[str, Any]]
    ) -> Optional[
        List[
            Union["ChatCompletionThinkingBlock", "ChatCompletionRedactedThinkingBlock"]
        ]
    ]:
        from litellm.types.llms.openai import (
            ChatCompletionRedactedThinkingBlock,
            ChatCompletionThinkingBlock,
        )

        thinking_blocks: List[
            Union["ChatCompletionThinkingBlock", "ChatCompletionRedactedThinkingBlock"]
        ] = []
        current_thinking_text_parts: List[str] = []
        current_signature: Optional[str] = None

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

    def get_combined_reasoning_content(
        self, chunks: List[Dict[str, Any]]
    ) -> ChatCompletionAssistantContentValue:
        return self.get_combined_content(chunks, delta_key="reasoning_content")

    def get_combined_audio_content(
        self, chunks: List[Dict[str, Any]]
    ) -> ChatCompletionAudioResponse:
        base64_data_list: List[str] = []
        transcript_list: List[str] = []
        expires_at: Optional[int] = None
        id: Optional[str] = None

        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta") or {}
                audio: Optional[ChatCompletionAudioDelta] = delta.get("audio")
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

    def _usage_chunk_calculation_helper(self, usage_chunk: Usage) -> dict:
        prompt_tokens = 0
        completion_tokens = 0
        ## anthropic prompt caching information ##
        cache_creation_input_tokens: Optional[int] = None
        cache_read_input_tokens: Optional[int] = None
        completion_tokens_details: Optional[CompletionTokensDetails] = None
        prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None

        if "prompt_tokens" in usage_chunk:
            prompt_tokens = usage_chunk.get("prompt_tokens", 0) or 0
        if "completion_tokens" in usage_chunk:
            completion_tokens = usage_chunk.get("completion_tokens", 0) or 0
        if "cache_creation_input_tokens" in usage_chunk:
            cache_creation_input_tokens = usage_chunk.get("cache_creation_input_tokens")
        if "cache_read_input_tokens" in usage_chunk:
            cache_read_input_tokens = usage_chunk.get("cache_read_input_tokens")
        if hasattr(usage_chunk, "completion_tokens_details"):
            if isinstance(usage_chunk.completion_tokens_details, dict):
                completion_tokens_details = CompletionTokensDetails(
                    **usage_chunk.completion_tokens_details
                )
            elif isinstance(
                usage_chunk.completion_tokens_details, CompletionTokensDetails
            ):
                completion_tokens_details = usage_chunk.completion_tokens_details
        if hasattr(usage_chunk, "prompt_tokens_details"):
            if isinstance(usage_chunk.prompt_tokens_details, dict):
                prompt_tokens_details = PromptTokensDetailsWrapper(
                    **usage_chunk.prompt_tokens_details
                )
            elif isinstance(
                usage_chunk.prompt_tokens_details, PromptTokensDetailsWrapper
            ):
                prompt_tokens_details = usage_chunk.prompt_tokens_details

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "completion_tokens_details": completion_tokens_details,
            "prompt_tokens_details": prompt_tokens_details,
        }

    def count_reasoning_tokens(self, response: ModelResponse) -> int:
        reasoning_tokens = 0
        for choice in response.choices:
            if (
                hasattr(cast(Choices, choice).message, "reasoning_content")
                and cast(Choices, choice).message.reasoning_content is not None
            ):
                reasoning_tokens += token_counter(
                    text=cast(Choices, choice).message.reasoning_content,
                    count_response_tokens=True,
                )

        return reasoning_tokens

    def _calculate_usage_per_chunk(
        self,
        chunks: List[Union[Dict[str, Any], ModelResponse]],
    ) -> "UsagePerChunk":
        from litellm.types.litellm_core_utils.streaming_chunk_builder_utils import (
            UsagePerChunk,
        )

        # # Update usage information if needed
        prompt_tokens = 0
        completion_tokens = 0
        ## anthropic prompt caching information ##
        cache_creation_input_tokens: Optional[int] = None
        cache_read_input_tokens: Optional[int] = None
        
        server_tool_use: Optional[ServerToolUse] = None
        web_search_requests: Optional[int] = None
        completion_tokens_details: Optional[CompletionTokensDetails] = None
        prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = None
        for chunk in chunks:
            usage_chunk: Optional[Usage] = None
            if "usage" in chunk:
                usage_chunk = chunk["usage"]
            elif (
                isinstance(chunk, ModelResponse)
                or isinstance(chunk, ModelResponseStream)
            ) and hasattr(chunk, "_hidden_params"):
                usage_chunk = chunk._hidden_params.get("usage", None)

            if usage_chunk is not None:
                usage_chunk_dict = self._usage_chunk_calculation_helper(usage_chunk)
                if (
                    usage_chunk_dict["prompt_tokens"] is not None
                    and usage_chunk_dict["prompt_tokens"] > 0
                ):
                    prompt_tokens = usage_chunk_dict["prompt_tokens"]
                if (
                    usage_chunk_dict["completion_tokens"] is not None
                    and usage_chunk_dict["completion_tokens"] > 0
                ):
                    completion_tokens = usage_chunk_dict["completion_tokens"]
                if usage_chunk_dict["cache_creation_input_tokens"] is not None and (
                    usage_chunk_dict["cache_creation_input_tokens"] > 0
                    or cache_creation_input_tokens is None
                ):
                    cache_creation_input_tokens = usage_chunk_dict[
                        "cache_creation_input_tokens"
                    ]
                if usage_chunk_dict["cache_read_input_tokens"] is not None and (
                    usage_chunk_dict["cache_read_input_tokens"] > 0
                    or cache_read_input_tokens is None
                ):
                    cache_read_input_tokens = usage_chunk_dict[
                        "cache_read_input_tokens"
                    ]
                if usage_chunk_dict["completion_tokens_details"] is not None:
                    completion_tokens_details = usage_chunk_dict[
                        "completion_tokens_details"
                    ]
                if hasattr(usage_chunk, 'server_tool_use') and usage_chunk.server_tool_use is not None:
                    server_tool_use = usage_chunk.server_tool_use
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

                prompt_tokens_details = usage_chunk_dict["prompt_tokens_details"]

        return UsagePerChunk(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
            server_tool_use=server_tool_use,
            web_search_requests=web_search_requests,
            completion_tokens_details=completion_tokens_details,
            prompt_tokens_details=prompt_tokens_details,
        )

    def calculate_usage(
        self,
        chunks: List[Union[Dict[str, Any], ModelResponse]],
        model: str,
        completion_output: str,
        messages: Optional[List] = None,
        reasoning_tokens: Optional[int] = None,
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
        cache_creation_input_tokens: Optional[int] = calculated_usage_per_chunk[
            "cache_creation_input_tokens"
        ]
        cache_read_input_tokens: Optional[int] = calculated_usage_per_chunk[
            "cache_read_input_tokens"
        ]

        server_tool_use: Optional[ServerToolUse] = calculated_usage_per_chunk[
            "server_tool_use"
        ]
        web_search_requests: Optional[int] = calculated_usage_per_chunk[
            "web_search_requests"
        ]
        completion_tokens_details: Optional[CompletionTokensDetails] = (
            calculated_usage_per_chunk["completion_tokens_details"]
        )
        prompt_tokens_details: Optional[PromptTokensDetailsWrapper] = (
            calculated_usage_per_chunk["prompt_tokens_details"]
        )

        try:
            returned_usage.prompt_tokens = prompt_tokens or token_counter(
                model=model, messages=messages
            )
        except (
            Exception
        ):  # don't allow this failing to block a complete streaming response from being returned
            print_verbose("token_counter failed, assuming prompt tokens is 0")
            returned_usage.prompt_tokens = 0
        returned_usage.completion_tokens = completion_tokens or token_counter(
            model=model,
            text=completion_output,
            count_response_tokens=True,  # count_response_tokens is a Flag to tell token counter this is a response, No need to add extra tokens we do for input messages
        )
        returned_usage.total_tokens = (
            returned_usage.prompt_tokens + returned_usage.completion_tokens
        )

        if cache_creation_input_tokens is not None:
            returned_usage._cache_creation_input_tokens = cache_creation_input_tokens
            setattr(
                returned_usage,
                "cache_creation_input_tokens",
                cache_creation_input_tokens,
            )  # for anthropic
        if cache_read_input_tokens is not None:
            returned_usage._cache_read_input_tokens = cache_read_input_tokens
            setattr(
                returned_usage, "cache_read_input_tokens", cache_read_input_tokens
            )  # for anthropic
        if completion_tokens_details is not None:
            if isinstance(completion_tokens_details, CompletionTokensDetails):
                returned_usage.completion_tokens_details = CompletionTokensDetailsWrapper(
                    **completion_tokens_details.model_dump()
                )
            else:
                returned_usage.completion_tokens_details = completion_tokens_details

        if reasoning_tokens is not None:
            if returned_usage.completion_tokens_details is None:
                returned_usage.completion_tokens_details = (
                    CompletionTokensDetailsWrapper(reasoning_tokens=reasoning_tokens)
                )
            elif (
                returned_usage.completion_tokens_details is not None
                and returned_usage.completion_tokens_details.reasoning_tokens is None
            ):
                returned_usage.completion_tokens_details.reasoning_tokens = (
                    reasoning_tokens
                )
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
                returned_usage.prompt_tokens_details.web_search_requests = (
                    web_search_requests
                )

        # Return a new usage object with the new values

        returned_usage = Usage(**returned_usage.model_dump())

        return returned_usage


def concatenate_base64_list(base64_strings: List[str]) -> str:
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

import base64
import time
from typing import Any, Dict, List, Optional, Union

from litellm.types.llms.openai import (
    ChatCompletionAssistantContentValue,
    ChatCompletionAudioDelta,
)
from litellm.types.utils import (
    ChatCompletionAudioResponse,
    ChatCompletionMessageToolCall,
    CompletionTokensDetails,
    Function,
    FunctionCall,
    ModelResponse,
    PromptTokensDetails,
    Usage,
)
from litellm.utils import print_verbose, token_counter


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

    def build_base_response(self, chunks: List[Dict[str, Any]]) -> ModelResponse:
        chunk = self.first_chunk
        id = ChunkProcessor._get_chunk_id(chunks)
        object = chunk["object"]
        created = chunk["created"]
        model = chunk["model"]
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

    def get_combined_tool_content(
        self, tool_call_chunks: List[Dict[str, Any]]
    ) -> List[ChatCompletionMessageToolCall]:

        argument_list: List[str] = []
        delta = tool_call_chunks[0]["choices"][0]["delta"]
        id = None
        name = None
        type = None
        tool_calls_list: List[ChatCompletionMessageToolCall] = []
        prev_index = None
        prev_name = None
        prev_id = None
        curr_id = None
        curr_index = 0
        for chunk in tool_call_chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta", {})
                tool_calls = delta.get("tool_calls", "")
                # Check if a tool call is present
                if tool_calls and tool_calls[0].function is not None:
                    if tool_calls[0].id:
                        id = tool_calls[0].id
                        curr_id = id
                        if prev_id is None:
                            prev_id = curr_id
                    if tool_calls[0].index:
                        curr_index = tool_calls[0].index
                    if tool_calls[0].function.arguments:
                        # Now, tool_calls is expected to be a dictionary
                        arguments = tool_calls[0].function.arguments
                        argument_list.append(arguments)
                    if tool_calls[0].function.name:
                        name = tool_calls[0].function.name
                    if tool_calls[0].type:
                        type = tool_calls[0].type
            if prev_index is None:
                prev_index = curr_index
            if prev_name is None:
                prev_name = name
            if curr_index != prev_index:  # new tool call
                combined_arguments = "".join(argument_list)
                tool_calls_list.append(
                    ChatCompletionMessageToolCall(
                        id=prev_id,
                        function=Function(
                            arguments=combined_arguments,
                            name=prev_name,
                        ),
                        type=type,
                    )
                )
                argument_list = []  # reset
                prev_index = curr_index
                prev_id = curr_id
                prev_name = name

        combined_arguments = (
            "".join(argument_list) or "{}"
        )  # base case, return empty dict

        tool_calls_list.append(
            ChatCompletionMessageToolCall(
                id=id,
                type="function",
                function=Function(
                    arguments=combined_arguments,
                    name=name,
                ),
            )
        )

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
        self, chunks: List[Dict[str, Any]]
    ) -> ChatCompletionAssistantContentValue:
        content_list: List[str] = []
        for chunk in chunks:
            choices = chunk["choices"]
            for choice in choices:
                delta = choice.get("delta", {})
                content = delta.get("content", "")
                if content is None:
                    continue  # openai v1.0.0 sets content = None for chunks
                content_list.append(content)

        # Combine the "content" strings into a single string || combine the 'function' strings into a single string
        combined_content = "".join(content_list)

        # Update the "content" field within the response dictionary
        return combined_content

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
        prompt_tokens_details: Optional[PromptTokensDetails] = None

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
                prompt_tokens_details = PromptTokensDetails(
                    **usage_chunk.prompt_tokens_details
                )
            elif isinstance(usage_chunk.prompt_tokens_details, PromptTokensDetails):
                prompt_tokens_details = usage_chunk.prompt_tokens_details

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
            "completion_tokens_details": completion_tokens_details,
            "prompt_tokens_details": prompt_tokens_details,
        }

    def calculate_usage(
        self,
        chunks: List[Union[Dict[str, Any], ModelResponse]],
        model: str,
        completion_output: str,
        messages: Optional[List] = None,
    ) -> Usage:
        """
        Calculate usage for the given chunks.
        """
        returned_usage = Usage()
        # # Update usage information if needed
        prompt_tokens = 0
        completion_tokens = 0
        ## anthropic prompt caching information ##
        cache_creation_input_tokens: Optional[int] = None
        cache_read_input_tokens: Optional[int] = None
        completion_tokens_details: Optional[CompletionTokensDetails] = None
        prompt_tokens_details: Optional[PromptTokensDetails] = None
        for chunk in chunks:
            usage_chunk: Optional[Usage] = None
            if "usage" in chunk:
                usage_chunk = chunk["usage"]
            elif isinstance(chunk, ModelResponse) and hasattr(chunk, "_hidden_params"):
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
                if usage_chunk_dict["cache_creation_input_tokens"] is not None:
                    cache_creation_input_tokens = usage_chunk_dict[
                        "cache_creation_input_tokens"
                    ]
                if usage_chunk_dict["cache_read_input_tokens"] is not None:
                    cache_read_input_tokens = usage_chunk_dict[
                        "cache_read_input_tokens"
                    ]
                if usage_chunk_dict["completion_tokens_details"] is not None:
                    completion_tokens_details = usage_chunk_dict[
                        "completion_tokens_details"
                    ]
                prompt_tokens_details = usage_chunk_dict["prompt_tokens_details"]
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
            returned_usage.completion_tokens_details = completion_tokens_details
        if prompt_tokens_details is not None:
            returned_usage.prompt_tokens_details = prompt_tokens_details

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

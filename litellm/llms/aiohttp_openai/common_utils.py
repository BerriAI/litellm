import json
from typing import List, Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
)


class AioHttpOpenAIError(BaseLLMException):
    def __init__(self, status_code, message):
        super().__init__(status_code=status_code, message=message)


def validate_environment(
    headers: dict,
    model: str,
    messages: List[AllMessageValues],
    optional_params: dict,
    api_key: Optional[str] = None,
) -> dict:
    """
    Return headers to use for aiopenhttp_openai chat completion request
    """
    headers.update(
        {
            "Request-Source": "unspecified:litellm",
            "accept": "application/json",
            "content-type": "application/json",
        }
    )
    if api_key:
        headers["Authorization"] = f"bearer {api_key}"
    return headers


class ModelResponseIterator:
    def __init__(
        self, streaming_response, sync_stream: bool, json_mode: Optional[bool] = False
    ):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.json_mode = json_mode

    def chunk_parser(self, chunk: dict) -> Union[GenericStreamingChunk, ModelResponseStream]:
        try:
            # Initialize default values
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None

            # Extract the index from the chunk
            index = int(chunk.get("choices", [{}])[0].get("index", 0))

            # Extract the text or delta content from the first choice
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                text = delta["content"]

            # Check for finish_reason
            finish_reason = chunk.get("choices", [{}])[0].get("finish_reason", "")

            # Determine if the stream has finished
            is_finished = finish_reason in ("length", "stop")

            # Create and return the parsed chunk
            returned_chunk = GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
                provider_specific_fields=provider_specific_fields,
            )

            return returned_chunk

        except json.JSONDecodeError:
            raise ValueError(f"Failed to decode JSON from chunk: {chunk}")


    # Sync iterator
    def __iter__(self):
        return self

    def _handle_string_chunk(
        self, str_line: str
    ) -> Union[GenericStreamingChunk, ModelResponseStream]:
        # chunk is a str at this point
        if "[DONE]" in str_line:
            return GenericStreamingChunk(
                text="",
                is_finished=True,
                finish_reason="stop",
                usage=None,
                index=0,
                tool_use=None,
            )
        elif str_line.startswith("data:"):
            data_json = json.loads(str_line[5:])
            return self.chunk_parser(chunk=data_json)
        else:
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

    def __next__(self):
        try:
            chunk = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            str_line = chunk
            if isinstance(chunk, bytes):  # Handle binary data
                str_line = chunk.decode("utf-8")  # Convert bytes to string
                index = str_line.find("data:")
                if index != -1:
                    str_line = str_line[index:]
            # chunk is a str at this point
            return self._handle_string_chunk(str_line=str_line)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            str_line = chunk
            if isinstance(chunk, bytes):  # Handle binary data
                str_line = chunk.decode("utf-8")  # Convert bytes to string
                index = str_line.find("data:")
                if index != -1:
                    str_line = str_line[index:]

            # chunk is a str at this point
            return self._handle_string_chunk(str_line=str_line)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")
        

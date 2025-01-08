import json
from typing import List, Optional

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
)


class CohereError(BaseLLMException):
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
    Return headers to use for cohere chat completion request

    Cohere API Ref: https://docs.cohere.com/reference/chat
    Expected headers:
    {
        "Request-Source": "unspecified:litellm",
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "bearer $CO_API_KEY"
    }
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
        self.content_blocks: List = []
        self.tool_index = -1
        self.json_mode = json_mode

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            text = ""
            tool_use: Optional[ChatCompletionToolCallChunk] = None
            is_finished = False
            finish_reason = ""
            usage: Optional[ChatCompletionUsageBlock] = None
            provider_specific_fields = None

            index = int(chunk.get("index", 0))

            if "text" in chunk:
                text = chunk["text"]
            elif "is_finished" in chunk and chunk["is_finished"] is True:
                is_finished = chunk["is_finished"]
                finish_reason = chunk["finish_reason"]

            if "citations" in chunk:
                provider_specific_fields = {"citations": chunk["citations"]}

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
            data_json = json.loads(str_line)
            return self.chunk_parser(chunk=data_json)
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

            data_json = json.loads(str_line)
            return self.chunk_parser(chunk=data_json)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

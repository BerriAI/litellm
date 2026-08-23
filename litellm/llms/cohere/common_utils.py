import json
from typing import Final, Literal

from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import (
    ChatCompletionToolCallChunk,
    ChatCompletionUsageBlock,
    GenericStreamingChunk,
    ProviderSpecificModelInfo,
)


class CohereError(BaseLLMException):
    def __init__(self, status_code, message):
        super().__init__(status_code=status_code, message=message)


class CohereModelInfo(BaseLLMModelInfo):
    def get_provider_info(
        self,
        model: str,
    ) -> ProviderSpecificModelInfo | None:
        """
        Default values all models of this provider support.
        """
        return None

    def get_models(self, api_key: str | None = None, api_base: str | None = None) -> list[str]:
        """
        Returns a list of models supported by this provider.
        """
        return []

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return api_key

    @staticmethod
    def get_api_base(
        api_base: str | None = None,
    ) -> str | None:
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        return {}

    @staticmethod
    def get_base_model(model: str) -> str | None:
        """
        Returns the base model name from the given model name.

        Some providers like bedrock - can receive model=`invoke/anthropic.claude-3-opus-20240229-v1:0` or `converse/anthropic.claude-3-opus-20240229-v1:0`
            This function will return `anthropic.claude-3-opus-20240229-v1:0`
        """

    @staticmethod
    def get_cohere_route(model: str) -> Literal["v1", "v2"]:
        """
        Get the Cohere route for the given model.

        Args:
            model: The model name (e.g., "cohere_chat/v2/command-r-plus", "command-r-plus")

        Returns:
            "v2" for standard Cohere v2 API (default), "v1" for Cohere v1 API
        """
        # Check for explicit v1 route
        if "v1/" in model:
            return "v1"

        # Default to v2 for all other cases
        return "v2"


def validate_environment(
    headers: dict,
    model: str,
    messages: list[AllMessageValues],
    optional_params: dict,
    api_key: str | None = None,
) -> dict:
    """
    Return headers to use for cohere chat completion request

    Cohere API Ref: https://docs.cohere.com/reference/chat
    Expected headers:
    {
        "Request-Source": "unspecified:litellm",
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer $CO_API_KEY"
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
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


class ModelResponseIterator:
    def __init__(self, streaming_response, sync_stream: bool, json_mode: bool | None = False):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.content_blocks: list = []
        self.tool_index = -1
        self.json_mode = json_mode

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        try:
            text = ""
            tool_use: Final[ChatCompletionToolCallChunk | None] = None
            is_finished = False
            finish_reason = ""
            usage: Final[ChatCompletionUsageBlock | None] = None
            provider_specific_fields = None

            index: Final = int(chunk.get("index", 0))

            if "text" in chunk:
                text = chunk["text"]
            elif "is_finished" in chunk and chunk["is_finished"] is True:
                is_finished = chunk["is_finished"]
                finish_reason = chunk["finish_reason"]

            if "citations" in chunk:
                provider_specific_fields = {"citations": chunk["citations"]}

            returned_chunk: Final = GenericStreamingChunk(
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
            chunk: Final = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    def convert_str_chunk_to_generic_chunk(self, chunk: str) -> GenericStreamingChunk:
        """
        Convert a string chunk to a GenericStreamingChunk

        Note: This is used for Cohere pass through streaming logging
        """
        str_line = chunk
        if isinstance(chunk, bytes):  # Handle binary data
            str_line = chunk.decode("utf-8")  # Convert bytes to string
            index: Final = str_line.find("data:")
            if index != -1:
                str_line = str_line[index:]

        data_json: Final = json.loads(str_line)
        return self.chunk_parser(chunk=data_json)

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk: Final = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")


class CohereV2ModelResponseIterator:
    """V2-specific response iterator for Cohere streaming"""

    def __init__(self, streaming_response, sync_stream: bool, json_mode: bool | None = False):
        self.streaming_response = streaming_response
        self.response_iterator = self.streaming_response
        self.content_blocks: list = []
        self.tool_index = -1
        self.json_mode = json_mode

    def _parse_content_delta(self, chunk: dict) -> str:
        """Parse content-delta chunks to extract text."""
        delta: Final = chunk.get("delta", {})
        message: Final = delta.get("message", {})
        content: Final = message.get("content", {})
        if isinstance(content, dict) and "text" in content:
            return content["text"]
        elif isinstance(content, str):
            return content
        return ""

    def _parse_tool_call_delta(self, chunk: dict) -> ChatCompletionToolCallChunk | None:
        """Parse tool-call-delta chunks to extract tool calls."""
        delta: Final = chunk.get("delta", {})
        tool_calls: Final = delta.get("tool_calls", [])
        if tool_calls:
            return {
                "id": tool_calls[0].get("id", ""),
                "type": "function",
                "function": {
                    "name": tool_calls[0].get("name", ""),
                    "arguments": tool_calls[0].get("arguments", ""),
                },
            }
        return None

    def _parse_tool_plan_delta(self, chunk: dict) -> dict | None:
        """Parse tool-plan-delta events to extract tool plan."""
        data: Final = chunk.get("data", {})
        delta: Final = data.get("delta", {})
        message: Final = delta.get("message", {})
        tool_plan: Final = message.get("tool_plan", "")
        if tool_plan:
            return {"tool_plan": tool_plan}
        return None

    def _parse_citation_start(self, chunk: dict) -> dict | None:
        """Parse citation-start events to extract citations."""
        data: Final = chunk.get("data", {})
        delta: Final = data.get("delta", {})
        message: Final = delta.get("message", {})
        citations: Final = message.get("citations", {})
        if citations:
            citation_data: Final = {
                "start": citations.get("start", 0),
                "end": citations.get("end", 0),
                "text": citations.get("text", ""),
                "sources": citations.get("sources", []),
                "type": citations.get("type", "TEXT_CONTENT"),
            }
            return {"citations": [citation_data]}
        return None

    def _parse_message_end(self, chunk: dict) -> tuple[bool, str, ChatCompletionUsageBlock | None]:
        """Parse message-end events to extract finish info and usage."""
        data: Final = chunk.get("data", {})
        delta: Final = data.get("delta", {})
        is_finished: Final = True
        finish_reason: Final = delta.get("finish_reason", "stop")

        usage = None
        usage_data: Final = delta.get("usage", {})
        if usage_data:
            tokens_data: Final = usage_data.get("tokens", {})
            usage = ChatCompletionUsageBlock(
                prompt_tokens=tokens_data.get("input_tokens", 0),
                completion_tokens=tokens_data.get("output_tokens", 0),
                total_tokens=tokens_data.get("input_tokens", 0) + tokens_data.get("output_tokens", 0),
            )

        return is_finished, finish_reason, usage

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        """
        Parse Cohere v2 streaming chunks.

        v2 format:
        - Content: chunk.type == "content-delta" -> chunk.delta.message.content.text
        - Tool calls: chunk.type == "tool-call-delta" -> chunk.delta.tool_calls
        - Tool plan: chunk.event == "tool-plan-delta" -> chunk.data.delta.message.tool_plan
        - Citations: chunk.event == "citation-start" -> chunk.data.delta.message.citations
        - Finish: chunk.event == "message-end" -> chunk.data.delta.finish_reason
        """
        try:
            text = ""
            tool_use: ChatCompletionToolCallChunk | None = None
            is_finished = False
            finish_reason = ""
            usage: ChatCompletionUsageBlock | None = None
            provider_specific_fields = None

            index: Final = int(chunk.get("index", 0))
            chunk_type: Final = chunk.get("type", "")
            event_type: Final = chunk.get("event", "")

            # Handle different chunk types
            if chunk_type == "content-delta":
                text = self._parse_content_delta(chunk)
            elif chunk_type == "tool-call-delta":
                tool_use = self._parse_tool_call_delta(chunk)
            elif event_type == "tool-plan-delta":
                provider_specific_fields = self._parse_tool_plan_delta(chunk)
            elif event_type == "citation-start":
                provider_specific_fields = self._parse_citation_start(chunk)
            elif event_type == "message-end":
                is_finished, finish_reason, usage = self._parse_message_end(chunk)

            # Handle citations in any chunk type (fallback)
            if "citations" in chunk:
                if provider_specific_fields is None:
                    provider_specific_fields = {}
                provider_specific_fields["citations"] = chunk["citations"]

            return GenericStreamingChunk(
                text=text,
                tool_use=tool_use,
                is_finished=is_finished,
                finish_reason=finish_reason,
                usage=usage,
                index=index,
                provider_specific_fields=provider_specific_fields,
            )

        except Exception as e:
            raise ValueError(f"Failed to parse v2 chunk: {e}, chunk: {chunk}")

    # Sync iterator
    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk: Final = self.response_iterator.__next__()
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopIteration:
            raise StopIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

    def convert_str_chunk_to_generic_chunk(self, chunk: str) -> GenericStreamingChunk:
        """
        Convert a string chunk to a GenericStreamingChunk for v2

        Note: This is used for Cohere v2 pass through streaming logging
        """
        str_line = chunk
        if isinstance(chunk, bytes):  # Handle binary data
            str_line = chunk.decode("utf-8")  # Convert bytes to string
            index: Final = str_line.find("data:")
            if index != -1:
                str_line = str_line[index:]

        data_json: Final = json.loads(str_line)
        return self.chunk_parser(chunk=data_json)

    # Async iterator
    def __aiter__(self):
        self.async_response_iterator = self.streaming_response.__aiter__()
        return self

    async def __anext__(self):
        try:
            chunk: Final = await self.async_response_iterator.__anext__()
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error receiving chunk from stream: {e}")

        try:
            return self.convert_str_chunk_to_generic_chunk(chunk=chunk)
        except StopAsyncIteration:
            raise StopAsyncIteration
        except ValueError as e:
            raise RuntimeError(f"Error parsing chunk: {e},\nReceived chunk: {chunk}")

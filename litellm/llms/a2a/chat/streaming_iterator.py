"""
A2A Streaming Response Iterator
"""

from typing import Final

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.llms.openai import ChatCompletionToolCallChunk
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

from ..common_utils import A2AError, extract_text_from_a2a_response


class A2AModelResponseIterator(BaseModelResponseIterator):
    """
    Iterator for parsing A2A streaming responses.

    Converts A2A JSON-RPC streaming chunks to OpenAI-compatible format.
    """

    def __init__(
        self,
        streaming_response,
        sync_stream: bool,
        json_mode: bool | None = False,
        model: str = "a2a/agent",
    ):
        super().__init__(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
        self.model = model

    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk | ModelResponseStream:
        """
        Parse A2A streaming chunk to OpenAI format.

        A2A chunk format:
        {
            "jsonrpc": "2.0",
            "id": "request-id",
            "result": {
                "message": {
                    "parts": [{"kind": "text", "text": "content"}]
                }
            }
        }

        Or for tasks:
        {
            "jsonrpc": "2.0",
            "result": {
                "kind": "task",
                "status": {"state": "running"},
                "artifacts": [{"parts": [{"kind": "text", "text": "content"}]}]
            }
        }
        """
        if "error" in chunk:
            error_value: Final = chunk["error"]
            error_message: Final = (
                error_value.get("message")
                if isinstance(error_value, dict) and isinstance(error_value.get("message"), str)
                else str(error_value)
            )
            raise A2AError(status_code=500, message=f"A2A error: {error_message}")

        try:
            # Extract text from A2A response
            text: Final = extract_text_from_a2a_response(chunk)

            # Determine finish reason
            finish_reason: Final = self._get_finish_reason(chunk)
            tool_calls: Final = self._get_tool_calls(chunk)

            # Return generic streaming chunk
            return GenericStreamingChunk(
                text=text,
                is_finished=bool(finish_reason or tool_calls),
                finish_reason=finish_reason or ("tool_calls" if tool_calls else ""),
                usage=None,
                index=0,
                tool_use=tool_calls,
            )
        except Exception:
            # Return empty chunk on parse error
            return GenericStreamingChunk(
                text="",
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

    def _handle_string_chunk(self, str_line: str | dict) -> GenericStreamingChunk | ModelResponseStream:
        if isinstance(str_line, dict):
            return self.chunk_parser(chunk=str_line)
        return super()._handle_string_chunk(str_line=str_line)

    def _get_finish_reason(self, chunk: dict) -> str | None:
        """Extract finish reason from A2A chunk"""
        result: Final = chunk.get("result", {})

        # Check for task completion
        if isinstance(result, dict):
            status: Final = result.get("status", {})
            if isinstance(status, dict):
                state: Final = status.get("state")
                if state == "completed":
                    return "stop"
                elif state == "failed":
                    return "stop"  # Map failed state to 'stop' (valid finish_reason)

        # Check for [DONE] marker
        if chunk.get("done") is True:
            return "stop"

        return None

    def _get_tool_calls(self, chunk: dict) -> ChatCompletionToolCallChunk | None:
        result: Final = chunk.get("result", {})
        if not isinstance(result, dict):
            return None
        tool_calls = result.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            first_tool_call: Final = tool_calls[0]
            return first_tool_call if isinstance(first_tool_call, dict) else None
        message = result.get("message")
        if isinstance(message, dict) and isinstance(message.get("tool_calls"), list) and message["tool_calls"]:
            first_tool_call = message["tool_calls"][0]
            return first_tool_call if isinstance(first_tool_call, dict) else None
        return None

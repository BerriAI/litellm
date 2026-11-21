"""
A2A Streaming Response Iterator
"""

from typing import Final

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

from ..common_utils import (
    A2A_TASK_FAILURE_STATES,
    A2A_TASK_INTERRUPTED_STATES,
    A2A_TASK_STATE_COMPLETED,
    A2AError,
    a2a_task_state,
    extract_text_from_a2a_response,
)


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
        self._last_task_state: str | None = None

    def _ensure_task_terminal_before_close(self) -> None:
        if self._last_task_state is None or self._last_task_state == A2A_TASK_STATE_COMPLETED:
            return
        if self._last_task_state in A2A_TASK_FAILURE_STATES:
            raise A2AError(
                status_code=500,
                message=f"A2A task ended with state: {self._last_task_state}",
            )
        if self._last_task_state in A2A_TASK_INTERRUPTED_STATES:
            raise A2AError(
                status_code=500,
                message=f"A2A task requires further action (state: {self._last_task_state})",
            )
        raise A2AError(
            status_code=500,
            message=f"A2A stream ended before task completed (state: {self._last_task_state})",
        )

    def __next__(self):
        try:
            return super().__next__()
        except StopIteration:
            self._ensure_task_terminal_before_close()
            raise

    async def __anext__(self):
        try:
            return await super().__anext__()
        except StopAsyncIteration:
            self._ensure_task_terminal_before_close()
            raise

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
        error: Final = chunk.get("error")
        if isinstance(error, dict):
            raise A2AError(status_code=500, message=f"A2A error: {error.get('message', 'Unknown error')}")

        try:
            # Extract text from A2A response
            text: Final = extract_text_from_a2a_response(chunk)

            # Determine finish reason
            finish_reason: Final = self._get_finish_reason(chunk)

            # Return generic streaming chunk
            return GenericStreamingChunk(
                text=text,
                is_finished=bool(finish_reason),
                finish_reason=finish_reason or "",
                usage=None,
                index=0,
                tool_use=None,
            )
        except A2AError:
            raise
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

    def _get_finish_reason(self, chunk: dict) -> str | None:
        """Extract finish reason from A2A chunk"""
        result: Final = chunk.get("result", {})

        if isinstance(result, dict):
            state: Final[str | None] = a2a_task_state(result)
            if state is not None:
                self._last_task_state = state
            if state == A2A_TASK_STATE_COMPLETED:
                return "stop"
            if state in A2A_TASK_FAILURE_STATES:
                raise A2AError(status_code=500, message=f"A2A task ended with state: {state}")
            if state in A2A_TASK_INTERRUPTED_STATES:
                raise A2AError(status_code=500, message=f"A2A task requires further action (state: {state})")

        if chunk.get("done") is True:
            if self._last_task_state not in (None, A2A_TASK_STATE_COMPLETED):
                self._ensure_task_terminal_before_close()
            return "stop"

        return None

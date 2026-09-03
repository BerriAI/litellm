"""
A2A Streaming Response Iterator
"""

from typing import Final

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.utils import GenericStreamingChunk, ModelResponseStream

from ..common_utils import extract_text_from_a2a_response


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
        # Text already emitted downstream, used to collapse cumulative snapshots.
        self._emitted_text: str = ""
        self._delta_count: int = 0

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
        try:
            # Extract text from A2A response, then reduce it to what is actually new.
            text: Final = self._to_incremental_text(extract_text_from_a2a_response(chunk))

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

    def _to_incremental_text(self, text: str) -> str:
        """
        Reduce an A2A event's text to the portion not yet emitted.

        A2A servers interleave true deltas with cumulative snapshots of the whole reply: a
        terminal non-partial ``status-update`` and an ``artifact-update`` carrying
        ``append: false`` both repeat everything produced so far. Forwarding those verbatim
        makes the client render the reply two or three times over, so emit only new text.

        Handles both streaming styles: servers that send deltas ("O", "K") and servers that
        send growing snapshots ("O", "OK") collapse to the same output.
        """
        if not text:
            return ""

        emitted: str = self._emitted_text
        if emitted and text.startswith(emitted):
            suffix: str = text[len(emitted) :]
            if suffix:
                self._emitted_text = text
                return suffix
            # text == emitted. Treat as a snapshot repeat, except while only a single delta
            # has been emitted, where a genuinely repeated delta is still indistinguishable.
            if self._delta_count > 1:
                return ""

        self._emitted_text = emitted + text
        self._delta_count += 1
        return text

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

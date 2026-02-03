"""
A2A Streaming Response Iterator
"""
from typing import Optional, Union

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
        json_mode: Optional[bool] = False,
        model: str = "a2a/agent",
    ):
        super().__init__(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
        self.model = model
    
    def chunk_parser(self, chunk: dict) -> Union[GenericStreamingChunk, ModelResponseStream]:
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
            # Extract text from A2A response
            text = extract_text_from_a2a_response(chunk)
            
            # Determine finish reason
            finish_reason = self._get_finish_reason(chunk)
            
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
    
    def _get_finish_reason(self, chunk: dict) -> Optional[str]:
        """Extract finish reason from A2A chunk"""
        result = chunk.get("result", {})
        
        # Check for task completion
        if isinstance(result, dict):
            status = result.get("status", {})
            if isinstance(status, dict):
                state = status.get("state")
                if state == "completed":
                    return "stop"
                elif state == "failed":
                    return "stop"  # Map failed state to 'stop' (valid finish_reason)
        
        # Check for [DONE] marker
        if chunk.get("done") is True:
            return "stop"
        
        return None

"""
A2A Streaming Iterator

Handles transformation of A2A Server-Sent Events (SSE) to OpenAI streaming format.

A2A Streaming Events:
1. Task event (kind: "task") - Initial task creation with status "submitted"
2. Status update (kind: "status-update") - Status changes (working, completed)
3. Artifact update (kind: "artifact-update") - Content/artifact delivery

OpenAI Streaming Format:
    data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}
"""

import json
import time
from typing import Any, AsyncIterator, Dict, Iterator, Optional, Union
from uuid import uuid4

from litellm._logging import verbose_logger
from litellm.types.utils import (
    Delta,
    GenericStreamingChunk,
    ModelResponseStream,
    StreamingChoices,
    Usage,
)


class A2AStreamingIterator:
    """
    Iterator that transforms A2A streaming events to OpenAI streaming format.
    
    Handles both sync and async iteration.
    """

    def __init__(
        self,
        streaming_response: Union[Iterator[str], AsyncIterator[str]],
        sync_stream: bool,
        json_mode: bool = False,
    ):
        self.streaming_response = streaming_response
        self.sync_stream = sync_stream
        self.json_mode = json_mode
        
        self._response_id = f"chatcmpl-{uuid4().hex[:8]}"
        self._created = int(time.time())
        self._accumulated_text = ""
        self._is_finished = False
        self._task_id: Optional[str] = None
        self._context_id: Optional[str] = None

    def __iter__(self) -> "A2AStreamingIterator":
        return self

    def __aiter__(self) -> "A2AStreamingIterator":
        return self

    def __next__(self) -> GenericStreamingChunk:
        if self._is_finished:
            raise StopIteration
        
        try:
            # Get next line from sync iterator
            line = next(self.streaming_response)  # type: ignore
            return self._process_line(line)
        except StopIteration:
            self._is_finished = True
            raise

    async def __anext__(self) -> GenericStreamingChunk:
        if self._is_finished:
            raise StopAsyncIteration
        
        try:
            # Get next line from async iterator
            line = await self.streaming_response.__anext__()  # type: ignore
            return self._process_line(line)
        except StopAsyncIteration:
            self._is_finished = True
            raise

    def _process_line(self, line: str) -> GenericStreamingChunk:
        """Process a single SSE line and transform to OpenAI format."""
        line = line.strip()
        
        # Skip empty lines and SSE comments
        if not line or line.startswith(":"):
            return self._create_empty_chunk()
        
        # Handle SSE data format
        if line.startswith("data:"):
            data = line[5:].strip()
            
            # Handle "[DONE]" marker
            if data == "[DONE]":
                self._is_finished = True
                return self._create_final_chunk()
            
            try:
                event_data = json.loads(data)
                return self._transform_a2a_event(event_data)
            except json.JSONDecodeError as e:
                verbose_logger.debug(f"Failed to parse A2A SSE data: {e}")
                return self._create_empty_chunk()
        
        # Try parsing as raw JSON (some implementations don't use SSE format)
        try:
            event_data = json.loads(line)
            return self._transform_a2a_event(event_data)
        except json.JSONDecodeError:
            return self._create_empty_chunk()

    def _transform_a2a_event(self, event: Dict[str, Any]) -> GenericStreamingChunk:
        """Transform an A2A streaming event to OpenAI chunk format."""
        result = event.get("result", {})
        
        # Determine event kind
        kind = result.get("kind", "")
        
        if kind == "task":
            # Initial task event
            self._task_id = result.get("id")
            self._context_id = result.get("contextId")
            return self._create_empty_chunk()
        
        elif kind == "status-update":
            # Status update event
            status = result.get("status", {})
            state = status.get("state", "").lower().replace("task_state_", "")
            final = result.get("final", False)
            
            # Check for message in status
            status_message = status.get("message", {})
            text = self._extract_text_from_parts(status_message.get("parts", []))
            
            if state == "completed" or final:
                self._is_finished = True
                return self._create_chunk(text="", is_finished=True, finish_reason="stop")
            
            if text:
                return self._create_chunk(text=text)
            
            return self._create_empty_chunk()
        
        elif kind == "artifact-update":
            # Artifact content event
            artifact = result.get("artifact", {})
            text = self._extract_text_from_parts(artifact.get("parts", []))
            
            if text:
                self._accumulated_text += text
                return self._create_chunk(text=text)
            
            return self._create_empty_chunk()
        
        elif kind == "message":
            # Direct message response
            message = result.get("message", result)
            text = self._extract_text_from_parts(message.get("parts", []))
            
            if text:
                self._accumulated_text += text
                return self._create_chunk(text=text)
            
            return self._create_empty_chunk()
        
        else:
            # Try to extract any text content
            text = ""
            
            # Check for message at top level
            if "message" in result:
                text = self._extract_text_from_parts(result["message"].get("parts", []))
            
            # Check for artifact at top level
            elif "artifact" in result:
                text = self._extract_text_from_parts(result["artifact"].get("parts", []))
            
            if text:
                self._accumulated_text += text
                return self._create_chunk(text=text)
            
            return self._create_empty_chunk()

    def _extract_text_from_parts(self, parts: list) -> str:
        """Extract text content from A2A parts."""
        text_parts = []
        for part in parts:
            if part.get("kind") == "text":
                text_parts.append(part.get("text", ""))
        return "".join(text_parts)

    def _create_chunk(
        self,
        text: str = "",
        is_finished: bool = False,
        finish_reason: Optional[str] = None,
    ) -> GenericStreamingChunk:
        """Create an OpenAI-format streaming chunk."""
        return GenericStreamingChunk(
            text=text,
            is_finished=is_finished,
            finish_reason=finish_reason or "",
            usage=None,
            index=0,
        )

    def _create_empty_chunk(self) -> GenericStreamingChunk:
        """Create an empty chunk (for non-content events)."""
        return GenericStreamingChunk(
            text="",
            is_finished=False,
            finish_reason="",
            usage=None,
            index=0,
        )

    def _create_final_chunk(self) -> GenericStreamingChunk:
        """Create the final chunk marking end of stream."""
        return GenericStreamingChunk(
            text="",
            is_finished=True,
            finish_reason="stop",
            usage=None,
            index=0,
        )


def create_streaming_response(
    chunk: GenericStreamingChunk,
    model: str,
    response_id: Optional[str] = None,
) -> ModelResponseStream:
    """
    Create a ModelResponseStream from a GenericStreamingChunk.
    
    This formats the chunk in the standard OpenAI streaming response format.
    """
    if response_id is None:
        response_id = f"chatcmpl-{uuid4().hex[:8]}"
    
    response = ModelResponseStream(
        id=response_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=[
            StreamingChoices(
                index=chunk.get("index", 0),
                delta=Delta(
                    role="assistant" if not chunk.get("is_finished") else None,
                    content=chunk.get("text") or None,
                ),
                finish_reason=chunk.get("finish_reason") if chunk.get("is_finished") else None,
            )
        ],
    )
    
    return response

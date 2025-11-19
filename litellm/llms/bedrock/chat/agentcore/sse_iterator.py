"""
SSE Stream Iterator for Bedrock AgentCore.

Handles Server-Sent Events (SSE) streaming responses from AgentCore.
"""

import json
from typing import TYPE_CHECKING

import httpx

from litellm._logging import verbose_logger
from litellm._uuid import uuid
from litellm.types.llms.bedrock_agentcore import AgentCoreUsage
from litellm.types.utils import Delta, ModelResponse, StreamingChoices, Usage

if TYPE_CHECKING:
    pass


class AgentCoreSSEStreamIterator:
    """Iterator for AgentCore SSE streaming responses. Supports both sync and async iteration."""

    def __init__(self, response: httpx.Response, model: str):
        self.response = response
        self.model = model
        self.finished = False
        self.line_iterator = None
        self.async_line_iterator = None

    def __iter__(self):
        """Initialize sync iteration."""
        self.line_iterator = self.response.iter_lines()
        return self

    def __aiter__(self):
        """Initialize async iteration."""
        self.async_line_iterator = self.response.aiter_lines()
        return self

    def __next__(self) -> ModelResponse:
        """Sync iteration - parse SSE events and yield ModelResponse chunks."""
        try:
            if self.line_iterator is None:
                raise StopIteration
            for line in self.line_iterator:
                line = line.strip()
                
                if not line or not line.startswith('data:'):
                    continue
                
                # Extract JSON from SSE line
                json_str = line[5:].strip()
                if not json_str:
                    continue
                
                try:
                    data = json.loads(json_str)
                    
                    # Skip non-dict data
                    if not isinstance(data, dict):
                        continue
                    
                    # Process content delta events
                    if "event" in data and isinstance(data["event"], dict):
                        event_payload = data["event"]
                        content_block_delta = event_payload.get("contentBlockDelta")
                        
                        if content_block_delta:
                            delta = content_block_delta.get("delta", {})
                            text = delta.get("text", "")
                            
                            if text:
                                # Yield chunk with text
                                chunk = ModelResponse(
                                    id=f"chatcmpl-{uuid.uuid4()}",
                                    created=0,
                                    model=self.model,
                                    object="chat.completion.chunk",
                                )
                                
                                chunk.choices = [
                                    StreamingChoices(
                                        finish_reason=None,
                                        index=0,
                                        delta=Delta(content=text, role="assistant"),
                                    )
                                ]
                                
                                return chunk
                        
                        # Check for metadata/usage
                        metadata = event_payload.get("metadata")
                        if metadata and "usage" in metadata:
                            # This is the final chunk with usage
                            chunk = ModelResponse(
                                id=f"chatcmpl-{uuid.uuid4()}",
                                created=0,
                                model=self.model,
                                object="chat.completion.chunk",
                            )
                            
                            chunk.choices = [
                                StreamingChoices(
                                    finish_reason="stop",
                                    index=0,
                                    delta=Delta(),
                                )
                            ]
                            
                            usage_data: AgentCoreUsage = metadata["usage"]  # type: ignore
                            setattr(chunk, "usage", Usage(
                                prompt_tokens=usage_data.get("inputTokens", 0),
                                completion_tokens=usage_data.get("outputTokens", 0),
                                total_tokens=usage_data.get("totalTokens", 0),
                            ))
                            
                            self.finished = True
                            return chunk
                    
                    # Check for final message (alternative finish signal)
                    if "message" in data and isinstance(data["message"], dict):
                        if not self.finished:
                            chunk = ModelResponse(
                                id=f"chatcmpl-{uuid.uuid4()}",
                                created=0,
                                model=self.model,
                                object="chat.completion.chunk",
                            )
                            
                            chunk.choices = [
                                StreamingChoices(
                                    finish_reason="stop",
                                    index=0,
                                    delta=Delta(),
                                )
                            ]
                            
                            self.finished = True
                            return chunk
                
                except json.JSONDecodeError:
                    verbose_logger.debug(f"Skipping non-JSON SSE line: {line[:100]}")
                    continue
            
            # Stream ended naturally
            raise StopIteration

        except StopIteration:
            raise
        except httpx.StreamConsumed:
            # This is expected when the stream has been fully consumed
            raise StopIteration
        except httpx.StreamClosed:
            # This is expected when the stream is closed
            raise StopIteration
        except Exception as e:
            verbose_logger.error(f"Error in AgentCore SSE stream: {str(e)}")
            raise StopIteration

    async def __anext__(self) -> ModelResponse:
        """Async iteration - parse SSE events and yield ModelResponse chunks."""
        try:
            if self.async_line_iterator is None:
                raise StopAsyncIteration
            async for line in self.async_line_iterator:
                line = line.strip()
                
                if not line or not line.startswith('data:'):
                    continue
                
                # Extract JSON from SSE line
                json_str = line[5:].strip()
                if not json_str:
                    continue
                
                try:
                    data = json.loads(json_str)
                    
                    # Skip non-dict data
                    if not isinstance(data, dict):
                        continue
                    
                    # Process content delta events
                    if "event" in data and isinstance(data["event"], dict):
                        event_payload = data["event"]
                        content_block_delta = event_payload.get("contentBlockDelta")
                        
                        if content_block_delta:
                            delta = content_block_delta.get("delta", {})
                            text = delta.get("text", "")
                            
                            if text:
                                # Yield chunk with text
                                chunk = ModelResponse(
                                    id=f"chatcmpl-{uuid.uuid4()}",
                                    created=0,
                                    model=self.model,
                                    object="chat.completion.chunk",
                                )
                                
                                chunk.choices = [
                                    StreamingChoices(
                                        finish_reason=None,
                                        index=0,
                                        delta=Delta(content=text, role="assistant"),
                                    )
                                ]
                                
                                return chunk
                        
                        # Check for metadata/usage
                        metadata = event_payload.get("metadata")
                        if metadata and "usage" in metadata:
                            # This is the final chunk with usage
                            chunk = ModelResponse(
                                id=f"chatcmpl-{uuid.uuid4()}",
                                created=0,
                                model=self.model,
                                object="chat.completion.chunk",
                            )
                            
                            chunk.choices = [
                                StreamingChoices(
                                    finish_reason="stop",
                                    index=0,
                                    delta=Delta(),
                                )
                            ]
                            
                            usage_data: AgentCoreUsage = metadata["usage"]  # type: ignore
                            setattr(chunk, "usage", Usage(
                                prompt_tokens=usage_data.get("inputTokens", 0),
                                completion_tokens=usage_data.get("outputTokens", 0),
                                total_tokens=usage_data.get("totalTokens", 0),
                            ))
                            
                            self.finished = True
                            return chunk
                    
                    # Check for final message (alternative finish signal)
                    if "message" in data and isinstance(data["message"], dict):
                        if not self.finished:
                            chunk = ModelResponse(
                                id=f"chatcmpl-{uuid.uuid4()}",
                                created=0,
                                model=self.model,
                                object="chat.completion.chunk",
                            )
                            
                            chunk.choices = [
                                StreamingChoices(
                                    finish_reason="stop",
                                    index=0,
                                    delta=Delta(),
                                )
                            ]
                            
                            self.finished = True
                            return chunk
                
                except json.JSONDecodeError:
                    verbose_logger.debug(f"Skipping non-JSON SSE line: {line[:100]}")
                    continue
            
            # Stream ended naturally
            raise StopAsyncIteration

        except StopAsyncIteration:
            raise
        except httpx.StreamConsumed:
            # This is expected when the stream has been fully consumed
            raise StopAsyncIteration
        except httpx.StreamClosed:
            # This is expected when the stream is closed
            raise StopAsyncIteration
        except Exception as e:
            verbose_logger.error(f"Error in AgentCore SSE stream: {str(e)}")
            raise StopAsyncIteration


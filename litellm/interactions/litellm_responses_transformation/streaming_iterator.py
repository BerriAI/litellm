"""
Streaming iterator for transforming Responses API stream to Interactions API stream.
"""

from typing import Any, AsyncIterator, Dict, Iterator, Optional, cast

from litellm.responses.streaming_iterator import (
    BaseResponsesAPIStreamingIterator,
    ResponsesAPIStreamingIterator,
    SyncResponsesAPIStreamingIterator,
)
from litellm.types.interactions import (
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIStreamingResponse,
)
from litellm.types.llms.openai import (
    OutputTextDeltaEvent,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseInProgressEvent,
    ResponsesAPIStreamingResponse,
)


class LiteLLMResponsesInteractionsStreamingIterator:
    """
    Iterator that wraps Responses API streaming and transforms chunks to Interactions API format.
    
    This class handles both sync and async iteration, transforming Responses API
    streaming events (output.text.delta, response.completed, etc.) to Interactions
    API streaming events (content.delta, interaction.complete, etc.).
    """

    def __init__(
        self,
        model: str,
        litellm_custom_stream_wrapper: BaseResponsesAPIStreamingIterator,
        request_input: Optional[InteractionInput],
        optional_params: InteractionsAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.responses_stream_iterator = litellm_custom_stream_wrapper
        self.request_input = request_input
        self.optional_params = optional_params
        self.custom_llm_provider = custom_llm_provider
        self.litellm_metadata = litellm_metadata or {}
        self.finished = False
        self.collected_text = ""
        self.sent_interaction_start = False
        self.sent_content_start = False

    def _transform_responses_chunk_to_interactions_chunk(
        self,
        responses_chunk: ResponsesAPIStreamingResponse,
    ) -> Optional[InteractionsAPIStreamingResponse]:
        """
        Transform a Responses API streaming chunk to an Interactions API streaming chunk.
        
        Responses API events:
        - output.text.delta -> content.delta
        - response.completed -> interaction.complete
        
        Interactions API events:
        - interaction.start
        - content.start
        - content.delta
        - content.stop
        - interaction.complete
        """
        if not responses_chunk:
            return None
        
        # Handle OutputTextDeltaEvent -> content.delta
        if isinstance(responses_chunk, OutputTextDeltaEvent):
            delta_text = responses_chunk.delta if isinstance(responses_chunk.delta, str) else ""
            self.collected_text += delta_text
            
            # Send interaction.start if not sent
            if not self.sent_interaction_start:
                self.sent_interaction_start = True
                return InteractionsAPIStreamingResponse(
                    event_type="interaction.start",
                    id=getattr(responses_chunk, "item_id", None) or f"interaction_{id(self)}",
                    object="interaction",
                    status="in_progress",
                    model=self.model,
                )
            
            # Send content.start if not sent
            if not self.sent_content_start:
                self.sent_content_start = True
                return InteractionsAPIStreamingResponse(
                    event_type="content.start",
                    id=getattr(responses_chunk, "item_id", None),
                    object="content",
                    delta={"type": "text", "text": ""},
                )
            
            # Send content.delta
            return InteractionsAPIStreamingResponse(
                event_type="content.delta",
                id=getattr(responses_chunk, "item_id", None),
                object="content",
                delta={"text": delta_text},
            )
        
        # Handle ResponseCreatedEvent or ResponseInProgressEvent -> interaction.start
        if isinstance(responses_chunk, (ResponseCreatedEvent, ResponseInProgressEvent)):
            if not self.sent_interaction_start:
                self.sent_interaction_start = True
                response_id = getattr(responses_chunk.response, "id", None) if hasattr(responses_chunk, "response") else None
                return InteractionsAPIStreamingResponse(
                    event_type="interaction.start",
                    id=response_id or f"interaction_{id(self)}",
                    object="interaction",
                    status="in_progress",
                    model=self.model,
                )
        
        # Handle ResponseCompletedEvent -> interaction.complete
        if isinstance(responses_chunk, ResponseCompletedEvent):
            self.finished = True
            response = responses_chunk.response
            
            # Send content.stop first if content was started
            if self.sent_content_start:
                # Note: We'll send this in the iterator, not here
                pass
            
            # Send interaction.complete
            return InteractionsAPIStreamingResponse(
                event_type="interaction.complete",
                id=getattr(response, "id", None) or f"interaction_{id(self)}",
                object="interaction",
                status="completed",
                model=self.model,
                outputs=[
                    {
                        "type": "text",
                        "text": self.collected_text,
                    }
                ],
            )
        
        # For other event types, return None (skip)
        return None

    def __iter__(self) -> Iterator[InteractionsAPIStreamingResponse]:
        """Sync iterator implementation."""
        return self

    def __next__(self) -> InteractionsAPIStreamingResponse:
        """Get next chunk in sync mode."""
        if self.finished:
            raise StopIteration
        
        # Check if we have a pending interaction.complete to send
        if hasattr(self, "_pending_interaction_complete"):
            pending: InteractionsAPIStreamingResponse = getattr(self, "_pending_interaction_complete")
            delattr(self, "_pending_interaction_complete")
            return pending
        
        # Use a loop instead of recursion to avoid stack overflow
        sync_iterator = cast(SyncResponsesAPIStreamingIterator, self.responses_stream_iterator)
        while True:
            try:
                # Get next chunk from responses API stream
                chunk = next(sync_iterator)
                
                # Transform chunk (chunk is already a ResponsesAPIStreamingResponse)
                transformed = self._transform_responses_chunk_to_interactions_chunk(chunk)
                
                if transformed:
                    # If we finished and content was started, send content.stop before interaction.complete
                    if self.finished and self.sent_content_start and transformed.event_type == "interaction.complete":
                        # Send content.stop first
                        content_stop = InteractionsAPIStreamingResponse(
                            event_type="content.stop",
                            id=transformed.id,
                            object="content",
                            delta={"type": "text", "text": self.collected_text},
                        )
                        # Store the interaction.complete to send next
                        self._pending_interaction_complete = transformed
                        return content_stop
                    return transformed
                
                # If no transformation, continue to next chunk (loop continues)
                
            except StopIteration:
                self.finished = True
                
                # Send final events if needed
                if self.sent_content_start:
                    return InteractionsAPIStreamingResponse(
                        event_type="content.stop",
                        object="content",
                        delta={"type": "text", "text": self.collected_text},
                    )
                
                raise StopIteration

    def __aiter__(self) -> AsyncIterator[InteractionsAPIStreamingResponse]:
        """Async iterator implementation."""
        return self

    async def __anext__(self) -> InteractionsAPIStreamingResponse:
        """Get next chunk in async mode."""
        if self.finished:
            raise StopAsyncIteration
        
        # Check if we have a pending interaction.complete to send
        if hasattr(self, "_pending_interaction_complete"):
            pending: InteractionsAPIStreamingResponse = getattr(self, "_pending_interaction_complete")
            delattr(self, "_pending_interaction_complete")
            return pending
        
        # Use a loop instead of recursion to avoid stack overflow
        async_iterator = cast(ResponsesAPIStreamingIterator, self.responses_stream_iterator)
        while True:
            try:
                # Get next chunk from responses API stream
                chunk = await async_iterator.__anext__()
                
                # Transform chunk (chunk is already a ResponsesAPIStreamingResponse)
                transformed = self._transform_responses_chunk_to_interactions_chunk(chunk)
                
                if transformed:
                    # If we finished and content was started, send content.stop before interaction.complete
                    if self.finished and self.sent_content_start and transformed.event_type == "interaction.complete":
                        # Send content.stop first
                        content_stop = InteractionsAPIStreamingResponse(
                            event_type="content.stop",
                            id=transformed.id,
                            object="content",
                            delta={"type": "text", "text": self.collected_text},
                        )
                        # Store the interaction.complete to send next
                        self._pending_interaction_complete = transformed
                        return content_stop
                    return transformed
                
                # If no transformation, continue to next chunk (loop continues)
                
            except StopAsyncIteration:
                self.finished = True
                
                # Send final events if needed
                if self.sent_content_start:
                    return InteractionsAPIStreamingResponse(
                        event_type="content.stop",
                        object="content",
                        delta={"type": "text", "text": self.collected_text},
                    )
                
                raise StopAsyncIteration


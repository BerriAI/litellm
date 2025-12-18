"""
Streaming iterator for transforming Responses API streaming events to Interactions API format.

This wraps the Responses API streaming iterator and transforms each event to the
Interactions API streaming event format.
"""

import time
import uuid
from typing import Any, Dict, List, Optional

from litellm.interactions.litellm_responses_transformation.transformation import (
    LiteLLMResponsesInteractionsConfig,
)
from litellm.types.interactions import (
    InteractionInput,
    InteractionsAPIOptionalRequestParams,
    InteractionsAPIStreamingResponse,
)


class LiteLLMResponsesStreamingIterator:
    """
    Streaming iterator that wraps Responses API streaming and transforms events
    to Interactions API streaming format.
    
    Supports both synchronous and asynchronous iteration.
    """

    def __init__(
        self,
        model: str,
        responses_stream_iterator: Any,  # BaseResponsesAPIStreamingIterator or similar
        request_input: Optional[InteractionInput],
        interactions_api_request: InteractionsAPIOptionalRequestParams,
        custom_llm_provider: Optional[str] = None,
        litellm_metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.responses_stream_iterator = responses_stream_iterator
        self.request_input = request_input
        self.interactions_api_request = interactions_api_request
        self.custom_llm_provider = custom_llm_provider
        self.litellm_metadata = litellm_metadata or {}
        
        # State tracking
        self.finished = False
        self.interaction_id = f"interaction_{uuid.uuid4().hex[:24]}"
        self.created_at = int(time.time())
        
        # Track events sent
        self.sent_interaction_start = False
        self.sent_content_start = False
        self.collected_text = ""
        self.collected_outputs: List[Dict[str, Any]] = []
        self.final_usage: Optional[Dict[str, Any]] = None

    def __iter__(self):
        return self

    def __aiter__(self):
        return self

    def _create_interaction_start_event(self) -> InteractionsAPIStreamingResponse:
        """Create the interaction.start event."""
        return InteractionsAPIStreamingResponse(
            event_type="interaction.start",
            id=self.interaction_id,
            object="interaction",
            model=self.model,
            status="in_progress",
            created=str(self.created_at),
        )

    def _create_content_start_event(self, index: int = 0) -> InteractionsAPIStreamingResponse:
        """Create a content.start event."""
        return InteractionsAPIStreamingResponse(
            event_type="content.start",
            id=self.interaction_id,
            delta={
                "type": "text",
                "text": "",
            },
        )

    def _create_content_delta_event(
        self,
        text: str,
        index: int = 0,
    ) -> InteractionsAPIStreamingResponse:
        """Create a content.delta event for text."""
        return InteractionsAPIStreamingResponse(
            event_type="content.delta",
            id=self.interaction_id,
            delta={
                "type": "text",
                "text": text,
            },
        )

    def _create_content_stop_event(self, index: int = 0) -> InteractionsAPIStreamingResponse:
        """Create a content.stop event."""
        return InteractionsAPIStreamingResponse(
            event_type="content.stop",
            id=self.interaction_id,
        )

    def _create_interaction_complete_event(self) -> InteractionsAPIStreamingResponse:
        """Create the interaction.complete event with final response."""
        # Build final outputs
        if self.collected_text:
            self.collected_outputs.append({
                "type": "text",
                "text": self.collected_text,
            })
        
        return InteractionsAPIStreamingResponse(
            event_type="interaction.complete",
            id=self.interaction_id,
            object="interaction",
            model=self.model,
            status="completed",
            created=str(self.created_at),
            role="model",
            outputs=self.collected_outputs,
            usage=self.final_usage,
        )

    def _transform_responses_chunk(
        self,
        chunk: Any,
    ) -> Optional[InteractionsAPIStreamingResponse]:
        """
        Transform a Responses API streaming chunk to Interactions API format.
        
        Returns the transformed event or None if the chunk should be skipped.
        """
        if chunk is None:
            return None

        # Get chunk as dict
        chunk_dict = chunk
        if hasattr(chunk, 'model_dump'):
            chunk_dict = chunk.model_dump()
        elif hasattr(chunk, '__dict__'):
            chunk_dict = dict(chunk)

        event_type = chunk_dict.get("type")

        # Handle response.created -> interaction.start
        if event_type == "response.created":
            # We handle this separately in the iterator
            return None

        # Handle text delta events
        elif event_type == "response.output_text.delta":
            delta_text = chunk_dict.get("delta", "")
            if delta_text:
                self.collected_text += delta_text
                return self._create_content_delta_event(text=delta_text)

        # Handle reasoning/thinking delta
        elif event_type == "response.reasoning_summary_text.delta":
            delta_text = chunk_dict.get("delta", "")
            if delta_text:
                return InteractionsAPIStreamingResponse(
                    event_type="content.delta",
                    id=self.interaction_id,
                    delta={
                        "type": "thought_summary",
                        "content": {
                            "type": "text",
                            "text": delta_text,
                        },
                    },
                )

        # Handle output item added (could be tool call)
        elif event_type == "response.output_item.added":
            item = chunk_dict.get("item", {})
            if item.get("type") == "function_call":
                # Store for final output
                self.collected_outputs.append({
                    "type": "function_call",
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", {}),
                    "id": item.get("call_id") or item.get("id", ""),
                })
            return None

        # Handle response.completed
        elif event_type == "response.completed":
            response_data = chunk_dict.get("response", {})
            # Extract usage from completed response
            usage = response_data.get("usage")
            if usage:
                self.final_usage = LiteLLMResponsesInteractionsConfig._transform_responses_usage_to_interactions_usage(
                    usage
                )
            return None

        # Handle content part events
        elif event_type == "response.content_part.added":
            if not self.sent_content_start:
                self.sent_content_start = True
                return self._create_content_start_event()

        elif event_type == "response.content_part.done":
            return self._create_content_stop_event()

        # Skip other events
        return None

    def __next__(self) -> InteractionsAPIStreamingResponse:
        """Synchronous iteration."""
        try:
            while True:
                if self.finished:
                    raise StopIteration

                # Send interaction.start first
                if not self.sent_interaction_start:
                    self.sent_interaction_start = True
                    return self._create_interaction_start_event()

                # Get next chunk from responses iterator
                try:
                    chunk = next(self.responses_stream_iterator)
                    
                    # Transform the chunk
                    result = self._transform_responses_chunk(chunk)
                    if result is not None:
                        return result
                    # Continue if chunk was skipped
                    
                except StopIteration:
                    # Stream finished - send completion event
                    self.finished = True
                    return self._create_interaction_complete_event()

        except Exception as e:
            self.finished = True
            raise e

    async def __anext__(self) -> InteractionsAPIStreamingResponse:
        """Asynchronous iteration."""
        try:
            while True:
                if self.finished:
                    raise StopAsyncIteration

                # Send interaction.start first
                if not self.sent_interaction_start:
                    self.sent_interaction_start = True
                    return self._create_interaction_start_event()

                # Get next chunk from responses iterator
                try:
                    chunk = await self.responses_stream_iterator.__anext__()
                    
                    # Transform the chunk
                    result = self._transform_responses_chunk(chunk)
                    if result is not None:
                        return result
                    # Continue if chunk was skipped
                    
                except StopAsyncIteration:
                    # Stream finished - send completion event
                    self.finished = True
                    return self._create_interaction_complete_event()

        except Exception as e:
            self.finished = True
            raise e

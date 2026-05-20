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
    API streaming events.

    Schema selection:
    - New schema (default, use_legacy_interactions_schema=False):
        interaction.created → step.start → step.delta … → step.stop → interaction.completed
    - Legacy schema (use_legacy_interactions_schema=True, remove after June 8 2026):
        interaction.start → content.start → content.delta … → content.stop → interaction.complete
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
        import litellm

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
        # Capture the schema flag once at construction time so all events
        # emitted by this stream use a consistent schema, even if the global
        # flag is mutated mid-stream (e.g. by a config reload).
        self._use_legacy: bool = litellm.use_legacy_interactions_schema

    def _transform_responses_chunk_to_interactions_chunk(
        self,
        responses_chunk: ResponsesAPIStreamingResponse,
    ) -> Optional[InteractionsAPIStreamingResponse]:
        """
        Transform a Responses API streaming chunk to an Interactions API streaming chunk.

        Emits new-schema events by default; falls back to legacy events when
        ``litellm.use_legacy_interactions_schema`` is True.
        Remove legacy branch after June 8, 2026.
        """
        if not responses_chunk:
            return None

        use_legacy = self._use_legacy

        # Handle OutputTextDeltaEvent
        if isinstance(responses_chunk, OutputTextDeltaEvent):
            delta_text = (
                responses_chunk.delta if isinstance(responses_chunk.delta, str) else ""
            )
            self.collected_text += delta_text
            item_id = (
                getattr(responses_chunk, "item_id", None) or f"interaction_{id(self)}"
            )

            # Send the "interaction started" event on the first delta
            if not self.sent_interaction_start:
                self.sent_interaction_start = True
                if use_legacy:
                    return InteractionsAPIStreamingResponse(
                        event_type="interaction.start",
                        id=item_id,
                        object="interaction",
                        status="in_progress",
                        model=self.model,
                    )
                else:
                    return InteractionsAPIStreamingResponse(
                        event_type="interaction.created",
                        id=item_id,
                        object="interaction",
                        status="in_progress",
                        model=self.model,
                    )

            # Send the "content/step started" event on the second delta
            if not self.sent_content_start:
                self.sent_content_start = True
                if use_legacy:
                    return InteractionsAPIStreamingResponse(
                        event_type="content.start",
                        id=item_id,
                        object="content",
                        delta={"type": "text", "text": ""},
                    )
                else:
                    return InteractionsAPIStreamingResponse(
                        event_type="step.start",
                        index=0,
                        step={"type": "model_output", "content": []},
                    )

            # Emit the delta itself
            if use_legacy:
                return InteractionsAPIStreamingResponse(
                    event_type="content.delta",
                    id=item_id,
                    object="content",
                    delta={"type": "text", "text": delta_text},
                )
            else:
                return InteractionsAPIStreamingResponse(
                    event_type="step.delta",
                    index=0,
                    delta={"type": "text", "text": delta_text},
                )

        # Handle ResponseCreatedEvent or ResponseInProgressEvent
        if isinstance(responses_chunk, (ResponseCreatedEvent, ResponseInProgressEvent)):
            if not self.sent_interaction_start:
                self.sent_interaction_start = True
                response_id = (
                    getattr(responses_chunk.response, "id", None)
                    if hasattr(responses_chunk, "response")
                    else None
                ) or f"interaction_{id(self)}"
                event_type = (
                    "interaction.start" if use_legacy else "interaction.created"
                )
                return InteractionsAPIStreamingResponse(
                    event_type=event_type,
                    id=response_id,
                    object="interaction",
                    status="in_progress",
                    model=self.model,
                )

        # Handle ResponseCompletedEvent
        if isinstance(responses_chunk, ResponseCompletedEvent):
            self.finished = True
            response = responses_chunk.response
            response_id = getattr(response, "id", None) or f"interaction_{id(self)}"

            if use_legacy:
                return InteractionsAPIStreamingResponse(
                    event_type="interaction.complete",
                    id=response_id,
                    object="interaction",
                    status="completed",
                    model=self.model,
                    outputs=[{"type": "text", "text": self.collected_text}],
                )
            else:
                return InteractionsAPIStreamingResponse(
                    event_type="interaction.completed",
                    id=response_id,
                    object="interaction",
                    status="completed",
                    model=self.model,
                    steps=[
                        {
                            "type": "model_output",
                            "content": [{"type": "text", "text": self.collected_text}],
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
        # Check for a pending interaction.complete/completed event BEFORE the
        # finished check — otherwise the buffered completion event (which
        # carries the full text) would be dropped after `self.finished` is set.
        if hasattr(self, "_pending_interaction_complete"):
            pending: InteractionsAPIStreamingResponse = getattr(
                self, "_pending_interaction_complete"
            )
            delattr(self, "_pending_interaction_complete")
            return pending

        if self.finished:
            raise StopIteration

        # Use a loop instead of recursion to avoid stack overflow
        sync_iterator = cast(
            SyncResponsesAPIStreamingIterator, self.responses_stream_iterator
        )
        while True:
            try:
                # Get next chunk from responses API stream
                chunk = next(sync_iterator)

                # Transform chunk (chunk is already a ResponsesAPIStreamingResponse)
                transformed = self._transform_responses_chunk_to_interactions_chunk(
                    chunk
                )

                if transformed:
                    completion_event_type = (
                        "interaction.complete"
                        if self._use_legacy
                        else "interaction.completed"
                    )
                    stop_event_type = (
                        "content.stop" if self._use_legacy else "step.stop"
                    )
                    # If content was started, send the stop event before the completion event.
                    if (
                        self.finished
                        and self.sent_content_start
                        and transformed.event_type == completion_event_type
                    ):
                        stop_kwargs: Dict[str, Any] = {
                            "event_type": stop_event_type,
                            "index": 0,
                        }
                        if self._use_legacy:
                            stop_kwargs["id"] = transformed.id
                            stop_kwargs["object"] = "content"
                            stop_kwargs["delta"] = {
                                "type": "text",
                                "text": self.collected_text,
                            }
                        stop_chunk = InteractionsAPIStreamingResponse(**stop_kwargs)
                        self._pending_interaction_complete = transformed
                        return stop_chunk
                    return transformed

                # If no transformation, continue to next chunk (loop continues)

            except StopIteration:
                self.finished = True

                # Send final stop event if content was started
                if self.sent_content_start:
                    stop_event_type = (
                        "content.stop" if self._use_legacy else "step.stop"
                    )
                    stop_kwargs = {
                        "event_type": stop_event_type,
                        "index": 0,
                    }
                    if self._use_legacy:
                        stop_kwargs["object"] = "content"
                        stop_kwargs["delta"] = {
                            "type": "text",
                            "text": self.collected_text,
                        }
                    return InteractionsAPIStreamingResponse(**stop_kwargs)

                raise StopIteration

    def __aiter__(self) -> AsyncIterator[InteractionsAPIStreamingResponse]:
        """Async iterator implementation."""
        return self

    async def __anext__(self) -> InteractionsAPIStreamingResponse:
        """Get next chunk in async mode."""
        # Check for a pending interaction.complete/completed event BEFORE the
        # finished check — otherwise the buffered completion event (which
        # carries the full text) would be dropped after `self.finished` is set.
        if hasattr(self, "_pending_interaction_complete"):
            pending: InteractionsAPIStreamingResponse = getattr(
                self, "_pending_interaction_complete"
            )
            delattr(self, "_pending_interaction_complete")
            return pending

        if self.finished:
            raise StopAsyncIteration

        # Use a loop instead of recursion to avoid stack overflow
        async_iterator = cast(
            ResponsesAPIStreamingIterator, self.responses_stream_iterator
        )
        while True:
            try:
                # Get next chunk from responses API stream
                chunk = await async_iterator.__anext__()

                # Transform chunk (chunk is already a ResponsesAPIStreamingResponse)
                transformed = self._transform_responses_chunk_to_interactions_chunk(
                    chunk
                )

                if transformed:
                    completion_event_type = (
                        "interaction.complete"
                        if self._use_legacy
                        else "interaction.completed"
                    )
                    stop_event_type = (
                        "content.stop" if self._use_legacy else "step.stop"
                    )
                    # If content was started, send the stop event before the completion event.
                    if (
                        self.finished
                        and self.sent_content_start
                        and transformed.event_type == completion_event_type
                    ):
                        stop_kwargs_async: Dict[str, Any] = {
                            "event_type": stop_event_type,
                            "index": 0,
                        }
                        if self._use_legacy:
                            stop_kwargs_async["id"] = transformed.id
                            stop_kwargs_async["object"] = "content"
                            stop_kwargs_async["delta"] = {
                                "type": "text",
                                "text": self.collected_text,
                            }
                        stop_chunk = InteractionsAPIStreamingResponse(
                            **stop_kwargs_async
                        )
                        self._pending_interaction_complete = transformed
                        return stop_chunk
                    return transformed

                # If no transformation, continue to next chunk (loop continues)

            except StopAsyncIteration:
                self.finished = True

                # Send final stop event if content was started
                if self.sent_content_start:
                    stop_event_type = (
                        "content.stop" if self._use_legacy else "step.stop"
                    )
                    stop_kwargs_async = {
                        "event_type": stop_event_type,
                        "index": 0,
                    }
                    if self._use_legacy:
                        stop_kwargs_async["object"] = "content"
                        stop_kwargs_async["delta"] = {
                            "type": "text",
                            "text": self.collected_text,
                        }
                    return InteractionsAPIStreamingResponse(**stop_kwargs_async)

                raise StopAsyncIteration

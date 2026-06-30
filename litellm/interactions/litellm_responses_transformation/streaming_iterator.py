"""
Streaming iterator for transforming Responses API stream to Interactions API stream.
"""

from collections import deque
from typing import (
    Any,
    AsyncIterator,
    Deque,
    Dict,
    Iterator,
    List,
    Optional,
    cast,
)

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
        interaction.created -> step.start -> step.delta ... -> step.stop -> interaction.completed
    - Legacy schema (use_legacy_interactions_schema=True, remove after June 8 2026):
        interaction.start -> content.start -> content.delta ... -> content.stop -> interaction.complete
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
        # Buffer of events that have been derived from upstream chunks but not
        # yet returned to the caller. A single Responses API chunk may expand
        # into multiple Interactions API events (e.g. the first text delta
        # produces interaction.created + step.start + step.delta), and the
        # terminal sequence on stream end may also span multiple events
        # (step.stop + interaction.completed).
        self._pending_events: Deque[InteractionsAPIStreamingResponse] = deque()
        # Tracks whether we've already emitted a terminal completion event so
        # the StopIteration fallback path doesn't double-emit.
        self._sent_completion_event = False
        # ID resolved from the first upstream chunk (item_id on a text delta or
        # response.id on response.created). Persisted so the EOF terminal
        # events stay correlated with the start events delivered earlier.
        self._interaction_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Event builders
    # ------------------------------------------------------------------

    def _build_interaction_start_event(self, interaction_id: str) -> InteractionsAPIStreamingResponse:
        event_type = "interaction.start" if self._use_legacy else "interaction.created"
        return InteractionsAPIStreamingResponse(
            event_type=event_type,
            id=interaction_id,
            object="interaction",
            status="in_progress",
            model=self.model,
        )

    def _build_content_start_event(self, interaction_id: str) -> InteractionsAPIStreamingResponse:
        if self._use_legacy:
            return InteractionsAPIStreamingResponse(
                event_type="content.start",
                id=interaction_id,
                object="content",
                delta={"type": "text", "text": ""},
            )
        return InteractionsAPIStreamingResponse(
            event_type="step.start",
            index=0,
            step={"type": "model_output", "content": []},
        )

    def _build_text_delta_event(self, interaction_id: str, delta_text: str) -> InteractionsAPIStreamingResponse:
        if self._use_legacy:
            return InteractionsAPIStreamingResponse(
                event_type="content.delta",
                id=interaction_id,
                object="content",
                delta={"type": "text", "text": delta_text},
            )
        return InteractionsAPIStreamingResponse(
            event_type="step.delta",
            index=0,
            delta={"type": "text", "text": delta_text},
        )

    def _build_content_stop_event(self, interaction_id: Optional[str]) -> InteractionsAPIStreamingResponse:
        if self._use_legacy:
            return InteractionsAPIStreamingResponse(
                event_type="content.stop",
                id=interaction_id,
                object="content",
                delta={"type": "text", "text": self.collected_text},
            )
        return InteractionsAPIStreamingResponse(
            event_type="step.stop",
            index=0,
        )

    def _build_completion_event(self, response_id: str) -> InteractionsAPIStreamingResponse:
        if self._use_legacy:
            return InteractionsAPIStreamingResponse(
                event_type="interaction.complete",
                id=response_id,
                object="interaction",
                status="completed",
                model=self.model,
                outputs=[{"type": "text", "text": self.collected_text}],
            )
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

    # ------------------------------------------------------------------
    # Per-chunk transform (returns a list of events to enqueue)
    # ------------------------------------------------------------------

    def _events_for_chunk(
        self, responses_chunk: ResponsesAPIStreamingResponse
    ) -> List[InteractionsAPIStreamingResponse]:
        """
        Translate a single upstream Responses API chunk into the list of
        Interactions API events it should produce.

        Returning a list (rather than a single event) lets a chunk emit any
        synthetic start events that haven't been sent yet *together with* the
        actual delta event, so we never silently drop the chunk's payload.
        """
        if not responses_chunk:
            return []

        # Text delta: emit any missing start events, then the delta itself.
        if isinstance(responses_chunk, OutputTextDeltaEvent):
            delta_text = responses_chunk.delta if isinstance(responses_chunk.delta, str) else ""
            self.collected_text += delta_text
            interaction_id = getattr(responses_chunk, "item_id", None) or f"interaction_{id(self)}"
            if self._interaction_id is None:
                self._interaction_id = interaction_id

            events: List[InteractionsAPIStreamingResponse] = []
            if not self.sent_interaction_start:
                self.sent_interaction_start = True
                events.append(self._build_interaction_start_event(interaction_id))
            if not self.sent_content_start:
                self.sent_content_start = True
                events.append(self._build_content_start_event(interaction_id))
            events.append(self._build_text_delta_event(interaction_id, delta_text))
            return events

        # Response created / in-progress: synthesize interaction start if we
        # haven't already sent one.
        if isinstance(responses_chunk, (ResponseCreatedEvent, ResponseInProgressEvent)):
            if not self.sent_interaction_start:
                self.sent_interaction_start = True
                response_id = (
                    getattr(responses_chunk.response, "id", None) if hasattr(responses_chunk, "response") else None
                ) or f"interaction_{id(self)}"
                if self._interaction_id is None:
                    self._interaction_id = response_id
                return [self._build_interaction_start_event(response_id)]
            return []

        # Response completed: emit step.stop (if content was started) followed
        # by the terminal completion event. Prefer the interaction id already
        # established by earlier events so consumers can correlate the start
        # and completion events by id (response.id may differ from the item_id
        # used to derive the initial id when the stream starts directly with a
        # text delta).
        if isinstance(responses_chunk, ResponseCompletedEvent):
            self.finished = True
            response = responses_chunk.response
            response_id = self._interaction_id or getattr(response, "id", None) or f"interaction_{id(self)}"

            terminal: List[InteractionsAPIStreamingResponse] = []
            if self.sent_content_start:
                terminal.append(self._build_content_stop_event(response_id))
            terminal.append(self._build_completion_event(response_id))
            self._sent_completion_event = True
            return terminal

        return []

    def _build_terminal_events_on_eof(
        self,
    ) -> List[InteractionsAPIStreamingResponse]:
        """
        Build the events to flush when the upstream stream ends without a
        ResponseCompletedEvent. Ensures consumers always observe a terminal
        interaction.completed/interaction.complete carrying the full text.
        """
        if self._sent_completion_event:
            return []

        fallback_id = self._interaction_id or f"interaction_{id(self)}"
        terminal: List[InteractionsAPIStreamingResponse] = []
        if self.sent_content_start:
            terminal.append(self._build_content_stop_event(fallback_id))
        if self.sent_interaction_start or self.collected_text:
            terminal.append(self._build_completion_event(fallback_id))
            self._sent_completion_event = True
        return terminal

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[InteractionsAPIStreamingResponse]:
        return self

    def __next__(self) -> InteractionsAPIStreamingResponse:
        if self._pending_events:
            return self._pending_events.popleft()

        if self.finished:
            raise StopIteration

        sync_iterator = cast(SyncResponsesAPIStreamingIterator, self.responses_stream_iterator)
        while True:
            try:
                chunk = next(sync_iterator)
            except StopIteration:
                self.finished = True
                self._pending_events.extend(self._build_terminal_events_on_eof())
                if self._pending_events:
                    return self._pending_events.popleft()
                raise

            events = self._events_for_chunk(chunk)
            if events:
                self._pending_events.extend(events)
                return self._pending_events.popleft()

    def __aiter__(self) -> AsyncIterator[InteractionsAPIStreamingResponse]:
        return self

    async def __anext__(self) -> InteractionsAPIStreamingResponse:
        if self._pending_events:
            return self._pending_events.popleft()

        if self.finished:
            raise StopAsyncIteration

        async_iterator = cast(ResponsesAPIStreamingIterator, self.responses_stream_iterator)
        while True:
            try:
                chunk = await async_iterator.__anext__()
            except StopAsyncIteration:
                self.finished = True
                self._pending_events.extend(self._build_terminal_events_on_eof())
                if self._pending_events:
                    return self._pending_events.popleft()
                raise

            events = self._events_for_chunk(chunk)
            if events:
                self._pending_events.extend(events)
                return self._pending_events.popleft()

    # ------------------------------------------------------------------
    # Backwards-compatible single-chunk transform (used by tests and any
    # external callers that drove the iterator chunk-by-chunk pre-fix).
    # ------------------------------------------------------------------

    def _transform_responses_chunk_to_interactions_chunk(
        self,
        responses_chunk: ResponsesAPIStreamingResponse,
    ) -> Optional[InteractionsAPIStreamingResponse]:
        """
        Compatibility shim: returns the *first* event produced for this chunk
        and queues any remaining events on ``self._pending_events`` so they
        are surfaced on subsequent calls/iterations.

        Prefer ``_events_for_chunk`` in new code.
        """
        events = self._events_for_chunk(responses_chunk)
        if not events:
            return None
        first = events[0]
        if len(events) > 1:
            self._pending_events.extend(events[1:])
        return first

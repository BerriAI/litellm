"""
xAI Grok Voice realtime event normalizer.

xAI's Grok Voice realtime API is structurally OpenAI-compatible but ships
several wire-format quirks that cause strict GA clients (e.g. pipecat's
``OpenAIRealtimeLLMService``) to crash before they can process tool calls:

  - ``ping`` keepalive events (unknown to GA clients)
  - ``usage: {}`` on ``response.created`` / ``response.done``
  - ``role: "tool"`` on ``conversation.item.added`` function_call items
  - Missing ``output_index`` / ``content_index`` on streaming response events
  - Missing ``part`` on ``response.content_part.done``

``XAIRealtimeNormalizer`` is plugged into ``RealTimeStreaming`` at handler
construction time (see ``handler.py``) so all normalization is isolated here
and ``RealTimeStreaming`` stays provider-agnostic.
"""

from typing import Any, Optional


class XAIRealtimeNormalizer:
    """Per-session normalizer that fixes xAI Grok Voice wire-format quirks."""

    # ---------------------------------------------------------------------------
    # Event-type sets used by the index-injection logic
    # ---------------------------------------------------------------------------
    _EVENTS_NEEDING_OUTPUT_INDEX = frozenset(
        [
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.output_audio.delta",
            "response.output_audio.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ]
    )
    _EVENTS_NEEDING_CONTENT_INDEX = frozenset(
        [
            "response.content_part.added",
            "response.content_part.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.output_audio.delta",
            "response.output_audio.done",
        ]
    )

    def __init__(self) -> None:
        # Cache content-part objects keyed by (response_id, item_id, content_index)
        # so that ``response.content_part.done`` events missing ``part`` can be
        # back-filled from earlier ``content_part.added`` / delta-done events.
        self._content_part_by_key: dict[tuple, dict[str, Any]] = {}

    # ---------------------------------------------------------------------------
    # Public interface consumed by RealTimeStreaming
    # ---------------------------------------------------------------------------

    def should_drop(self, event: object) -> bool:
        """Return True for provider-specific keepalives unknown to GA clients."""
        return isinstance(event, dict) and event.get("type") == "ping"

    def normalize(self, event: dict) -> dict:
        """Apply all xAI normalization passes in order."""
        event = self._normalize_content_part_events(event)
        event_type = event.get("type") or ""
        event = self._normalize_conversation_item_added(event, event_type)
        event = self._inject_missing_indices(event, event_type)
        event = self._normalize_response_usage_event(event, event_type)
        return event

    def patch_outgoing_session(self, session: dict) -> dict:
        """Patch a client ``session.update`` payload before forwarding to xAI.

        Unlike OpenAI, xAI does not default ``turn_detection.create_response``
        to ``True`` for ``server_vad``. Clients such as Pipecat omit the field,
        which leaves VAD detecting speech but never auto-creating a response.
        Only fill the default when the client did not set ``create_response``.
        """
        session = dict(session)
        self._default_server_vad_create_response(session)
        return session

    @staticmethod
    def _default_server_vad_create_response(session: dict) -> None:
        turn_detection = session.get("turn_detection")
        if isinstance(turn_detection, dict):
            XAIRealtimeNormalizer._ensure_server_vad_create_response(turn_detection)

        audio = session.get("audio")
        if isinstance(audio, dict):
            audio_input = audio.get("input")
            if isinstance(audio_input, dict):
                nested_td = audio_input.get("turn_detection")
                if isinstance(nested_td, dict):
                    XAIRealtimeNormalizer._ensure_server_vad_create_response(nested_td)

    @staticmethod
    def _ensure_server_vad_create_response(turn_detection: dict) -> None:
        if (
            turn_detection.get("type") == "server_vad"
            and "create_response" not in turn_detection
        ):
            turn_detection["create_response"] = True

    # ---------------------------------------------------------------------------
    # Pass 1: content-part caching and back-fill
    # ---------------------------------------------------------------------------

    @staticmethod
    def _content_part_key(event: dict) -> tuple:
        return (
            event.get("response_id"),
            event.get("item_id"),
            event.get("content_index", 0),
        )

    def _remember_content_part(self, event: dict) -> None:
        part = event.get("part")
        if isinstance(part, dict):
            self._content_part_by_key[self._content_part_key(event)] = part

    def _update_content_part_field(
        self, event: dict, *, part_type: str, field: str, value: object
    ) -> None:
        if value is None:
            return
        key = self._content_part_key(event)
        existing = self._content_part_by_key.get(key)
        if not isinstance(existing, dict):
            updated = {"type": part_type, field: value}
        else:
            updated = {
                **existing,
                "type": existing.get("type", part_type),
                field: value,
            }
        self._content_part_by_key[key] = updated

    def _resolve_content_part(self, event: dict) -> dict[str, Any]:
        part = event.get("part")
        if isinstance(part, dict):
            return part
        cached = self._content_part_by_key.get(self._content_part_key(event))
        if isinstance(cached, dict):
            return cached
        return {"type": "audio", "transcript": ""}

    def _normalize_content_part_events(self, event: dict) -> dict:
        event_type = event.get("type")

        if event_type == "response.content_part.added":
            self._remember_content_part(event)
            if not isinstance(event.get("part"), dict):
                return {**event, "part": self._resolve_content_part(event)}
            return event

        if event_type == "response.output_text.done":
            self._update_content_part_field(
                event, part_type="text", field="text", value=event.get("text")
            )
            return event

        if event_type == "response.output_audio_transcript.done":
            self._update_content_part_field(
                event,
                part_type="audio",
                field="transcript",
                value=event.get("transcript"),
            )
            return event

        if event_type == "response.content_part.done":
            self._remember_content_part(event)
            if not isinstance(event.get("part"), dict):
                return {**event, "part": self._resolve_content_part(event)}
            return event

        return event

    # ---------------------------------------------------------------------------
    # Pass 2: conversation.item.added role normalisation
    # ---------------------------------------------------------------------------

    @staticmethod
    def _normalize_conversation_item_added(event: dict, event_type: str) -> dict:
        """Map ``role: "tool"`` → ``role: "assistant"`` on function_call items.

        xAI uses ``role: "tool"`` which is not in the GA-allowed set
        ("user" | "assistant" | "system").
        """
        if event_type != "conversation.item.added":
            return event
        item = event.get("item")
        if not isinstance(item, dict):
            return event
        if item.get("role") == "tool":
            return {**event, "item": {**item, "role": "assistant"}}
        return event

    # ---------------------------------------------------------------------------
    # Pass 3: inject missing output_index / content_index
    # ---------------------------------------------------------------------------

    def _inject_missing_indices(self, event: dict, event_type: str) -> dict:
        """Inject ``output_index`` / ``content_index`` defaults when absent.

        xAI omits both fields on every streaming response event; pydantic GA
        clients require them as non-optional ints.  Defaulting to 0 is correct
        for single-turn single-item responses and harmless for well-formed events.
        """
        needs_output = event_type in self._EVENTS_NEEDING_OUTPUT_INDEX
        needs_content = event_type in self._EVENTS_NEEDING_CONTENT_INDEX
        if not needs_output and not needs_content:
            return event
        patch: dict[str, Any] = {}
        if needs_output and "output_index" not in event:
            patch["output_index"] = 0
        if needs_content and "content_index" not in event:
            patch["content_index"] = 0
        if not patch:
            return event
        return {**event, **patch}

    # ---------------------------------------------------------------------------
    # Pass 4: response usage normalisation
    # ---------------------------------------------------------------------------

    @staticmethod
    def _default_ga_usage() -> dict[str, Any]:
        default_details: dict[str, Any] = {
            "cached_tokens": 0,
            "text_tokens": 0,
            "audio_tokens": 0,
        }
        return {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "input_token_details": default_details.copy(),
            "output_token_details": default_details.copy(),
        }

    @staticmethod
    def _normalize_usage(
        usage: object, *, empty_as_null: bool
    ) -> Optional[dict[str, Any]]:
        """Coerce a usage object into the full OpenAI GA shape.

        ``empty_as_null=True`` for ``response.created`` (usage optional).
        ``empty_as_null=False`` for ``response.done`` (e2e tests assert non-null).
        """
        if not isinstance(usage, dict):
            return None
        if not usage:
            return None if empty_as_null else XAIRealtimeNormalizer._default_ga_usage()
        default_details: dict[str, Any] = {
            "cached_tokens": 0,
            "text_tokens": 0,
            "audio_tokens": 0,
        }
        normalized: dict[str, Any] = {
            "total_tokens": usage.get("total_tokens", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "input_token_details": default_details.copy(),
            "output_token_details": default_details.copy(),
        }
        for key in ("input_token_details", "output_token_details"):
            details = usage.get(key)
            if isinstance(details, dict):
                normalized[key] = {**default_details, **details}
        return normalized

    def _normalize_response_usage_event(self, event: dict, event_type: str) -> dict:
        if event_type not in ("response.created", "response.done"):
            return event
        response = event.get("response")
        if not isinstance(response, dict) or "usage" not in response:
            return event
        normalized_usage = self._normalize_usage(
            response.get("usage"),
            empty_as_null=event_type == "response.created",
        )
        if normalized_usage is response.get("usage"):
            return event
        return {**event, "response": {**response, "usage": normalized_usage}}

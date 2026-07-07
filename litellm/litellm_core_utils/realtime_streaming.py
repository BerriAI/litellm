import asyncio
import concurrent.futures
import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Union, cast

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.base_llm.realtime.transformation import BaseRealtimeConfig
from litellm.types.llms.openai import (
    OpenAIRealtimeEvents,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseDelta,
    OpenAIRealtimeStreamResponseBaseObject,
    OpenAIRealtimeStreamSessionEvents,
)
from litellm.types.realtime import ALL_DELTA_TYPES

from .litellm_logging import Logging as LiteLLMLogging

if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

    CLIENT_CONNECTION_CLASS = ClientConnection
else:
    CLIENT_CONNECTION_CLASS = Any

# Create a thread pool with a maximum of 10 threads
executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)


class RealtimeEventNormalizer(Protocol):
    def should_drop(self, event: object) -> bool: ...
    def normalize(self, event: dict) -> dict: ...
    def patch_outgoing_session(self, session: dict) -> dict: ...


DefaultLoggedRealTimeEventTypes = [
    "session.created",
    "response.create",
    "response.done",
    "conversation.item.added",  # GA
    "conversation.item.done",  # GA
]


class RealTimeStreaming:
    def __init__(
        self,
        websocket: Any,
        backend_ws: CLIENT_CONNECTION_CLASS,
        logging_obj: LiteLLMLogging,
        provider_config: Optional[BaseRealtimeConfig] = None,
        model: str = "",
        user_api_key_dict: Optional[Any] = None,
        request_data: Optional[Dict] = None,
        backend_uses_beta_protocol: Optional[bool] = None,
        force_transcription_model: Optional[str] = None,
        event_normalizer: Optional[RealtimeEventNormalizer] = None,
    ):
        self.websocket = websocket
        self.backend_ws = backend_ws
        self.logging_obj = logging_obj
        self.messages: List[OpenAIRealtimeEvents] = []
        self.input_message: Dict = {}
        self.input_messages: List[Dict[str, str]] = []
        self.session_tools: List[Dict] = []
        self.tool_calls: List[Dict] = []

        # Detect whether the client is explicitly opting into the beta protocol.
        self._client_wants_beta = self._detect_beta_header(websocket)
        self._backend_uses_beta_protocol = (
            self._client_wants_beta if backend_uses_beta_protocol is None else backend_uses_beta_protocol
        )

        _logged_real_time_event_types = litellm.logged_real_time_event_types

        if _logged_real_time_event_types is None:
            _logged_real_time_event_types = DefaultLoggedRealTimeEventTypes
        self.logged_real_time_event_types = _logged_real_time_event_types
        self.provider_config = provider_config
        self.model = model
        self.current_delta_chunks: Optional[List[OpenAIRealtimeResponseDelta]] = None
        self.current_output_item_id: Optional[str] = None
        self.current_response_id: Optional[str] = None
        self.current_conversation_id: Optional[str] = None
        self.current_item_chunks: Optional[List[OpenAIRealtimeOutputItemDone]] = None
        self.current_delta_type: Optional[ALL_DELTA_TYPES] = None
        self.session_configuration_request: Optional[str] = None
        self.user_api_key_dict = user_api_key_dict
        self.request_data: Dict = request_data or {}
        # Violation counter for end_session_after_n_fails support
        self._violation_count: int = 0
        # When a text message is blocked, hold the guardrail reason so the next
        # response.create can be rewritten to include the failure context.
        self._pending_guardrail_message: Optional[str] = None
        # Track whether session.created has already been sent to the client
        # (e.g. synthetic event in deferred setup mode).
        self._session_created_sent_to_client: bool = False
        # Track whether we have already sent the guardrail turn-detection update
        # that disables provider auto-response for transcription guardrails.
        self._guardrail_turn_detection_update_sent: bool = False
        # Deferred Gemini Live setup: Pipecat may stream audio before session.update.
        # Buffer client audio until the backend acknowledges setup (setupComplete).
        self._backend_setup_complete: bool = provider_config is None or provider_config.requires_session_configuration()
        self._flushing_pending_messages_until_setup: bool = False
        self._pending_messages_until_setup: List[str] = []
        self._pending_messages_byte_total: int = 0
        # Gemini Live rejects a follow-up BidiGenerateContentSetup once any
        # content (realtimeInput / clientContent / toolResponse) has been sent.
        self._content_sent_after_setup: bool = False
        # Whether this is a transcription-only session (session.type == "transcription",
        # e.g. gpt-realtime-whisper). Such sessions must not be sent response.create and
        # their input_audio_transcription.completed usage drives duration-based cost.
        self._force_transcription_model = force_transcription_model
        self._is_transcription_session: bool = force_transcription_model is not None
        # Optional per-provider GA event normalizer (e.g. XAIRealtimeNormalizer).
        self._event_normalizer = event_normalizer

    # Per-connection caps for pre-setup audio frames (message count + total bytes).
    _MAX_BUFFERED_MESSAGES: int = 200
    _MAX_BUFFERED_BYTES: int = 10 * 1024 * 1024  # 10 MB

    _SESSION_EVENT_TYPES = frozenset(["session.created", "session.updated"])
    _CLIENT_AUDIO_BUFFER_TYPES = frozenset(
        [
            "input_audio_buffer.append",
            "input_audio_buffer.commit",
            "input_audio_buffer.clear",
            "input_audio_buffer.end",
        ]
    )
    _CLIENT_AUDIO_BUFFER_COMMIT_TYPES = frozenset(["input_audio_buffer.commit", "input_audio_buffer.end"])
    _AUDIO_FORMAT_MAP: Dict[str, Dict[str, Any]] = {
        "pcm16": {"type": "audio/pcm", "rate": 24000},
        "g711_ulaw": {"type": "audio/G711-ulaw", "rate": 8000},
        "g711_alaw": {"type": "audio/G711-alaw", "rate": 8000},
    }
    # GA name → beta name (when client WebSocket includes OpenAI-Beta: realtime=v1)
    _GA_TO_BETA_EVENT_TYPES: Dict[str, str] = {
        "conversation.item.added": "conversation.item.created",
        "response.output_text.delta": "response.text.delta",
        "response.output_audio.delta": "response.audio.delta",
        "response.output_audio_transcript.delta": "response.audio_transcript.delta",
        "response.output_text.done": "response.text.done",
        "response.output_audio.done": "response.audio.done",
        "response.output_audio_transcript.done": "response.audio_transcript.done",
    }
    _GA_TO_BETA_CONTENT_TYPES: Dict[str, str] = {
        "output_text": "text",
        "output_audio": "audio",
    }

    def _should_store_message(
        self,
        message_obj: Union[dict, OpenAIRealtimeEvents],
    ) -> bool:
        _msg_type = message_obj["type"] if "type" in message_obj else None
        if self.logged_real_time_event_types == "*":
            return True
        if _msg_type and _msg_type in self.logged_real_time_event_types:
            return True
        return False

    def store_message(self, message: Union[str, bytes, dict, OpenAIRealtimeEvents]):
        """Store message in list"""
        if isinstance(message, bytes):
            message = message.decode("utf-8")
        if isinstance(message, dict):
            # TypedDict union members do not narrow to plain dict for mypy.
            message_obj: Dict[str, Any] = cast(Dict[str, Any], message)
        else:
            message_obj = cast(Dict[str, Any], json.loads(cast(str, message)))
        self._collect_tool_calls_from_response_done(cast(dict, message_obj))
        if not self._should_store_message(message_obj):
            return
        try:
            event_type = message_obj.get("type", "")
            if event_type in self._SESSION_EVENT_TYPES:
                typed_obj: OpenAIRealtimeEvents = OpenAIRealtimeStreamSessionEvents(**message_obj)  # type: ignore
            else:
                # Catch-all base object so unknown/new event names never raise.
                typed_obj = OpenAIRealtimeStreamResponseBaseObject(**message_obj)  # type: ignore
        except Exception as e:
            verbose_logger.debug(f"Error parsing message for logging: {e}")
            self.messages.append(message_obj)  # type: ignore[arg-type]
            return
        self.messages.append(typed_obj)

    def _collect_user_input_from_client_event(self, message: Union[str, dict]) -> None:
        """Extract user text content from client WebSocket events for spend logging."""
        try:
            if isinstance(message, str):
                msg_obj = json.loads(message)
            elif isinstance(message, dict):
                msg_obj = message
            else:
                return

            msg_type = msg_obj.get("type", "")

            if msg_type == "conversation.item.create":
                item = msg_obj.get("item", {})
                if item.get("role") == "user":
                    content_list = item.get("content", [])
                    for content in content_list:
                        if isinstance(content, dict) and content.get("type") == "input_text":
                            text = content.get("text", "")
                            if text:
                                self.input_messages.append({"role": "user", "content": text})
            elif msg_type == "session.update":
                session = msg_obj.get("session", {})
                instructions = session.get("instructions", "")
                if instructions:
                    self.input_messages.append({"role": "system", "content": instructions})
                tools = session.get("tools")
                if tools and isinstance(tools, list):
                    self.session_tools = tools
                # GA: session.type is required; log it for traceability but no action needed
                verbose_logger.debug(f"Realtime session.type: {session.get('type')}")
                if session.get("type") == "transcription":
                    self._is_transcription_session = True
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass

    def _collect_user_input_from_backend_event(self, event_obj: Union[dict, OpenAIRealtimeEvents]) -> None:
        """Extract user voice transcription from backend events for spend logging."""
        try:
            event_type = event_obj.get("type", "")
            if event_type == "conversation.item.input_audio_transcription.completed":
                transcript = cast(str, event_obj.get("transcript", ""))
                if transcript:
                    self.input_messages.append({"role": "user", "content": transcript})
        except (AttributeError, TypeError):
            pass

    def _detect_transcription_session_from_backend(self, event_obj: Union[dict, OpenAIRealtimeEvents]) -> None:
        """Flag transcription-only sessions from backend session events."""
        try:
            event_type = event_obj.get("type", "")
            if event_type in (
                "transcription_session.created",
                "transcription_session.updated",
            ):
                self._is_transcription_session = True
            elif event_type in ("session.created", "session.updated"):
                session = cast(dict, event_obj).get("session", {}) or {}
                if session.get("type") == "transcription":
                    self._is_transcription_session = True
        except (AttributeError, TypeError):
            pass

    def _capture_transcription_usage(self, event_obj: Union[dict, OpenAIRealtimeEvents]) -> None:
        """
        Append a usage-only transcription completed event to the logged results so
        the cost calculator can bill it by audio duration. The default logged event
        types exclude this event, so it is captured here directly for transcription
        sessions rather than widening logging for every realtime session. Only the
        type and usage are kept — the transcript is already captured separately in
        input_messages, so it is not duplicated into the response log here.
        """
        try:
            usage = event_obj.get("usage")
            if usage is None:
                return
            # If this event type is already captured by store_message (e.g. the user
            # logs all realtime events), don't append a second copy.
            if self._should_store_message(event_obj):
                return
            self.messages.append(
                cast(
                    OpenAIRealtimeEvents,
                    {
                        "type": "conversation.item.input_audio_transcription.completed",
                        "usage": usage,
                    },
                )
            )
        except (AttributeError, TypeError):
            pass

    def _collect_tool_calls_from_response_done(self, event_obj: Union[dict, OpenAIRealtimeEvents]) -> None:
        """Extract function_call items from response.done events for spend logging."""
        try:
            if event_obj.get("type") != "response.done":
                return
            response = cast(Dict[str, Any], event_obj.get("response", {}))
            for item in response.get("output", []):
                if item.get("type") == "function_call":
                    self.tool_calls.append(
                        {
                            "id": item.get("call_id", ""),
                            "type": "function",
                            "function": {
                                "name": item.get("name", ""),
                                "arguments": item.get("arguments", "{}"),
                            },
                        }
                    )
        except (AttributeError, TypeError):
            pass

    def store_input(self, message: Union[str, dict]):
        """Store input message"""
        self.input_message = message if isinstance(message, dict) else {}
        self._collect_user_input_from_client_event(message)
        if self.logging_obj:
            self.logging_obj.pre_call(input=message, api_key="")

    async def log_messages(self):
        """Log messages in list"""
        if self.logging_obj:
            if self.input_messages:
                self.logging_obj.model_call_details["messages"] = self.input_messages
            if self.session_tools or self.tool_calls:
                self.logging_obj.model_call_details["realtime_tools"] = self.session_tools
                self.logging_obj.model_call_details["realtime_tool_calls"] = self.tool_calls
            ## ASYNC LOGGING
            # Route through the bounded logging worker (per-coroutine timeout +
            # concurrency cap) instead of a bare create_task, so a slow callback
            # can't leave suspended tasks pinning each call's response in memory.
            GLOBAL_LOGGING_WORKER.ensure_initialized_and_enqueue(self.logging_obj.async_success_handler(self.messages))
            ## SYNC LOGGING
            executor.submit(self.logging_obj.success_handler(self.messages))

    async def _send_to_backend(self, message: str) -> bool:
        """Send a message to the backend WebSocket.

        If a provider_config is set the message is first passed through
        transform_realtime_request so that provider-specific translation
        (e.g. dropping session.update for Vertex AI) is applied even for
        guardrail-injected messages.

        Returns True if at least one message was actually delivered to the
        backend, False if the provider transformation produced no output and
        the message was effectively dropped.
        """
        message = self._enforce_transcription_session_model(message)
        if self.provider_config:
            transformed = self.provider_config.transform_realtime_request(
                message, self.model, self.session_configuration_request
            )
            sent = False
            for msg in transformed:
                try:
                    msg_obj = json.loads(msg)
                except (json.JSONDecodeError, TypeError):
                    msg_obj = None
                if isinstance(msg_obj, dict) and self.provider_config.is_setup_message(msg_obj):
                    if self._content_sent_after_setup:
                        verbose_logger.debug("Dropping follow-up setup after content was already sent to backend")
                        continue
                    msg = self._maybe_inject_guardrail_auto_response_disable(msg)
                    await self.backend_ws.send(msg)  # type: ignore[union-attr, attr-defined]
                    self._cache_session_configuration_request(msg)
                    sent = True
                else:
                    is_content_message = isinstance(msg_obj, dict) and self.provider_config.is_content_message(msg_obj)
                    # Send first, then mutate state, so a failed send leaves both
                    # ``session_configuration_request`` and
                    # ``_content_sent_after_setup`` untouched. Caching or marking
                    # content before send would leave the session believing the
                    # backend received a setup/content frame it never got, causing
                    # subsequent client session.update messages to be dropped.
                    await self.backend_ws.send(msg)  # type: ignore[union-attr, attr-defined]
                    self._cache_session_configuration_request(msg)
                    if is_content_message:
                        self._content_sent_after_setup = True
                    sent = True
            return sent
        await self.backend_ws.send(message)  # type: ignore[union-attr, attr-defined]
        return True

    def _enforce_transcription_session_model(self, message: str) -> str:
        """Force client transcription session updates to the authorized model.

        `/v1/realtime?intent=transcription` may intentionally omit `model` from
        the upstream URL for Azure compatibility, but the proxy still authorizes
        a resolved LiteLLM model before opening the backend websocket. If a
        client later sends a transcription `session.update`, any model embedded
        in that update must be rewritten to the same authorized model instead of
        allowing a post-auth model/deployment switch.

        Normal realtime sessions keep their independent nested transcription
        model behavior because `_force_transcription_model` is only set for
        transcription-intent websocket routes.
        """
        if self._force_transcription_model is None:
            return message

        try:
            message_obj = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return message

        if message_obj.get("type") not in (
            "session.update",
            "transcription_session.update",
        ):
            return message

        session = message_obj.get("session")
        if not isinstance(session, dict):
            return message

        if session.get("type") == "transcription":
            self._is_transcription_session = True

        authorized_model = self._force_transcription_model
        changed = False

        transcription = session.get("input_audio_transcription")
        if isinstance(transcription, dict) and transcription.get("model") != authorized_model:
            session["input_audio_transcription"] = {
                **transcription,
                "model": authorized_model,
            }
            changed = True

        audio = session.get("audio")
        if isinstance(audio, dict):
            audio_input = audio.get("input")
            if isinstance(audio_input, dict):
                nested_transcription = audio_input.get("transcription")
                if isinstance(nested_transcription, dict) and nested_transcription.get("model") != authorized_model:
                    session["audio"] = {
                        **audio,
                        "input": {
                            **audio_input,
                            "transcription": {
                                **nested_transcription,
                                "model": authorized_model,
                            },
                        },
                    }
                    changed = True

        if not changed:
            return message
        return json.dumps(message_obj)

    def _uses_deferred_backend_setup(self) -> bool:
        """True when setup is deferred until the client's first session.update."""
        if self.provider_config is None:
            return False
        return not self.provider_config.requires_session_configuration()

    @staticmethod
    def _collapse_buffered_audio_messages(messages: List[str]) -> List[str]:
        """Apply ``input_audio_buffer.clear`` semantics before replaying buffered frames.

        During deferred Gemini Live setup, ``clear`` is buffered alongside appends.
        On flush each append becomes a provider ``realtimeInput``; ``clear`` must
        drop preceding uncommitted appends instead of being forwarded as a no-op.
        """
        collapsed: List[str] = []
        pending_appends: List[str] = []

        for message in messages:
            try:
                msg_type = json.loads(message).get("type")
            except (json.JSONDecodeError, TypeError):
                collapsed.extend(pending_appends)
                pending_appends = []
                collapsed.append(message)
                continue

            if msg_type == "input_audio_buffer.append":
                pending_appends.append(message)
            elif msg_type == "input_audio_buffer.clear":
                pending_appends = []
            elif msg_type in RealTimeStreaming._CLIENT_AUDIO_BUFFER_COMMIT_TYPES:
                collapsed.extend(pending_appends)
                pending_appends = []
                collapsed.append(message)
            else:
                collapsed.extend(pending_appends)
                pending_appends = []
                collapsed.append(message)

        collapsed.extend(pending_appends)
        return collapsed

    def _sync_pending_messages_byte_total(self) -> None:
        self._pending_messages_byte_total = sum(
            len(message.encode("utf-8")) for message in self._pending_messages_until_setup
        )

    def _should_buffer_client_message_until_setup(self, message: str) -> bool:
        if not self._uses_deferred_backend_setup():
            return False
        if self._backend_setup_complete and not self._flushing_pending_messages_until_setup:
            return False
        try:
            msg_obj = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return False
        return msg_obj.get("type") in RealTimeStreaming._CLIENT_AUDIO_BUFFER_TYPES

    def _buffer_pending_message_until_setup(self, message: str) -> None:
        try:
            msg_type = json.loads(message).get("type")
        except (json.JSONDecodeError, TypeError):
            msg_type = None

        if msg_type == "input_audio_buffer.clear":
            self._pending_messages_until_setup = self._collapse_buffered_audio_messages(
                self._pending_messages_until_setup + [message]
            )
            self._sync_pending_messages_byte_total()
            return

        msg_bytes = len(message.encode("utf-8"))
        if (
            len(self._pending_messages_until_setup) < RealTimeStreaming._MAX_BUFFERED_MESSAGES
            and self._pending_messages_byte_total + msg_bytes <= RealTimeStreaming._MAX_BUFFERED_BYTES
        ):
            self._pending_messages_until_setup.append(message)
            self._pending_messages_byte_total += msg_bytes
        else:
            verbose_logger.warning(
                "Pre-setup buffer full (%d messages / %d bytes); dropping frame",
                len(self._pending_messages_until_setup),
                self._pending_messages_byte_total,
            )

    async def _flush_pending_messages_until_setup(self) -> bool:
        pending = self._collapse_buffered_audio_messages(self._pending_messages_until_setup)
        self._pending_messages_until_setup = []
        self._pending_messages_byte_total = 0
        for idx, message in enumerate(pending):
            try:
                await self._send_to_backend(message)
            except Exception as e:
                unsent = pending[idx:]
                self._pending_messages_until_setup = unsent + self._pending_messages_until_setup
                self._pending_messages_byte_total = sum(
                    len(msg.encode("utf-8")) for msg in self._pending_messages_until_setup
                )
                verbose_logger.debug(
                    "Failed to flush buffered client message after setup: %s (%d buffered message(s) retained)",
                    e,
                    len(unsent),
                )
                return False
        return True

    def _should_drop_event_from_client(self, event: object) -> bool:
        """Return True for provider-specific events that must not reach GA clients."""
        if self._event_normalizer is not None:
            return self._event_normalizer.should_drop(event)
        return False

    def _normalize_event_for_ga_client(self, event: dict) -> dict:
        """Apply per-provider GA normalization before forwarding to clients."""
        if self._event_normalizer is not None:
            return self._event_normalizer.normalize(event)
        return event

    def _event_to_client_json(self, event: dict) -> str:
        return json.dumps(self._normalize_event_for_ga_client(event))

    async def _send_event_to_client(self, event: Any, event_str: str) -> bool:
        if self._should_drop_event_from_client(event):
            return False
        if isinstance(event, dict):
            event = self._normalize_event_for_ga_client(event)
            event_str = json.dumps(event)
        if self._client_wants_beta and isinstance(event, dict):
            try:
                translated = self._translate_event_to_beta(event)
                if translated is None:
                    return False
                await self.websocket.send_text(json.dumps(translated))
                return True
            except Exception as e:
                verbose_logger.warning(
                    "Failed to translate %s to beta protocol, forwarding untranslated event to client: %s",
                    event.get("type"),
                    e,
                )
        await self.websocket.send_text(event_str)
        return True

    def _cache_session_configuration_request(self, transformed_message: str) -> None:
        """Store setup payload once sent to backend.

        Updates the cached setup on every successful setup send so follow-up
        ``session.update`` messages (which produce a merged setup with new
        ``generationConfig`` / ``systemInstruction`` / etc.) are reflected in
        the cache used by downstream readers (``transform_session_created_event``,
        ``return_new_content_delta_events`` modality lookup, ...).
        """
        try:
            message_obj = json.loads(transformed_message)
            if "setup" in message_obj:
                self.session_configuration_request = transformed_message
        except (json.JSONDecodeError, TypeError):
            return

    def _make_disable_auto_response_message(self) -> str:
        """Return a session.update that disables VAD auto-response."""
        turn_detection: Dict[str, Any] = {
            "type": "server_vad",
            "create_response": False,
        }
        if self._backend_uses_beta_protocol:
            session: Dict[str, Any] = {"turn_detection": turn_detection}
        else:
            session = {
                "type": "realtime",
                "audio": {"input": {"turn_detection": turn_detection}},
            }
        return json.dumps({"type": "session.update", "session": session})

    async def _maybe_send_guardrail_turn_detection_update(self) -> None:
        """Disable provider auto-response once when transcription guardrails are enabled."""
        if self._guardrail_turn_detection_update_sent:
            return
        if not self._has_audio_transcription_guardrails():
            return
        sent = await self._send_to_backend(self._make_disable_auto_response_message())
        # Only mark as sent when the provider transformation actually delivered
        # the update to the backend. Otherwise (e.g. Gemini drops session.update
        # after the initial setup), leave the flag unset so future opportunities
        # — such as a duplicate session.created — can retry.
        if sent:
            self._guardrail_turn_detection_update_sent = True

    def _maybe_inject_guardrail_auto_response_disable(self, setup_message: str) -> str:
        """Fold the transcription-guardrail auto-response disable into the setup.

        Gemini/Vertex Live reject a second ``setup`` (1007), so the guardrail's
        ``automaticActivityDetection.disabled=true`` cannot be delivered as a
        follow-up session.update; it must live in the one-and-only setup, or a
        ``realtime_input_transcription`` guardrail is bypassed (the model
        auto-responds before the proxy can gate the turn). Applies only to the
        bidi ``setup`` shape; OpenAI sessions accept follow-up updates and so are
        left untouched (handled by ``_maybe_send_guardrail_turn_detection_update``).
        """
        if self._guardrail_turn_detection_update_sent:
            return setup_message
        if not self._has_audio_transcription_guardrails():
            return setup_message
        try:
            obj = json.loads(setup_message)
        except (json.JSONDecodeError, TypeError):
            return setup_message
        setup = obj.get("setup") if isinstance(obj, dict) else None
        if not isinstance(setup, dict):
            return setup_message
        automatic = setup.setdefault("realtimeInputConfig", {}).setdefault("automaticActivityDetection", {})
        automatic["disabled"] = True
        self._guardrail_turn_detection_update_sent = True
        verbose_logger.debug(
            "Realtime: folded automaticActivityDetection.disabled=true into setup for transcription-guardrail gating"
        )
        return json.dumps(obj)

    def _has_realtime_guardrails_for_event_hooks(
        self,
        event_hooks: List[Any],
    ) -> bool:
        """Return True if any callback would run for one of ``event_hooks``."""
        from litellm.integrations.custom_guardrail import CustomGuardrail

        return any(
            isinstance(cb, CustomGuardrail)
            and any(
                cb.should_run_guardrail(
                    data=self.request_data,
                    event_type=et,
                )
                for et in event_hooks
            )
            for cb in litellm.callbacks
        )

    def _has_realtime_guardrails(self) -> bool:
        """Return True if any callback is registered for realtime guardrail event types."""
        from litellm.types.guardrails import GuardrailEventHooks

        return self._has_realtime_guardrails_for_event_hooks(
            [
                GuardrailEventHooks.realtime_input_transcription,
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ]
        )

    def _has_audio_transcription_guardrails(self) -> bool:
        """Return True when a guardrail is configured for the audio/VAD transcript path.

        Only ``realtime_input_transcription`` hooks disable ``server_vad`` auto-response.
        ``pre_call`` / ``post_call`` guardrails (e.g. Model Armor on chat completions)
        must not override ``turn_detection.create_response`` on realtime sessions.
        """
        from litellm.types.guardrails import GuardrailEventHooks

        return self._has_realtime_guardrails_for_event_hooks([GuardrailEventHooks.realtime_input_transcription])

    async def run_realtime_guardrails(
        self,
        transcript: str,
        item_id: Optional[str] = None,
        pre_block_backend_message: Optional[str] = None,
        event_hooks: Optional[List[Any]] = None,
    ) -> bool:
        """
        Run registered guardrails on realtime text (transcript, user message, tool output).

        Returns True if blocked (synthetic warning already sent to client).
        Returns False if clean (caller should send response.create to the backend).

        ``pre_block_backend_message`` (if provided) is sent to the backend
        BEFORE any of the guardrail's own backend messages when a block is
        triggered. This is needed for protocol contracts that require a
        specific message to be sent first — e.g. Gemini Live requires a
        matching ``toolResponse`` immediately after a ``toolCall`` before any
        other client messages can be accepted.

        ``event_hooks`` selects which guardrail modes to evaluate. Audio/VAD
        transcript completion uses ``realtime_input_transcription`` only;
        typed user messages and tool outputs use ``pre_call``.
        """
        from litellm.integrations.custom_guardrail import CustomGuardrail
        from litellm.types.guardrails import GuardrailEventHooks

        if event_hooks is None:
            event_hooks = [GuardrailEventHooks.realtime_input_transcription]
        _realtime_event_types = event_hooks
        _check_data = {**self.request_data, "transcript": transcript}
        _already_run: set = set()

        for callback in litellm.callbacks:
            if not isinstance(callback, CustomGuardrail):
                continue
            if id(callback) in _already_run:
                continue
            if not any(callback.should_run_guardrail(data=_check_data, event_type=et) for et in _realtime_event_types):
                continue
            _already_run.add(id(callback))
            try:
                await callback.apply_guardrail(
                    inputs={"texts": [transcript], "images": []},
                    request_data={"user_api_key_dict": self.user_api_key_dict},
                    input_type="request",
                )
            except Exception as e:
                # Re-raise unexpected errors (no status_code/detail = programming bug, not a block).
                # HTTPException and guardrail-raised exceptions have a status_code or detail attr.
                is_guardrail_block = hasattr(e, "status_code") or isinstance(e, ValueError)
                if not is_guardrail_block:
                    verbose_logger.exception(
                        "[realtime guardrail] unexpected error in apply_guardrail: %s",
                        e,
                    )
                    raise
                # Extract the human-readable error from the detail dict (HTTPException)
                # or fall back to str(e) for plain ValueError.
                detail = getattr(e, "detail", None)
                if isinstance(detail, dict):
                    safe_msg = detail.get("error") or str(e)
                elif detail is not None:
                    safe_msg = str(detail)
                else:
                    safe_msg = str(e) or "I'm sorry, that request was blocked by the content filter."

                # Use realtime_violation_message if configured; fall back to guardrail error text.
                error_msg = getattr(callback, "realtime_violation_message", None) or safe_msg

                # Deliver any caller-supplied backend message FIRST so that
                # protocol contracts requiring a specific ordering (e.g.
                # Gemini Live's mandatory ``toolResponse`` after a
                # ``toolCall``) are honored before the guardrail's own
                # clientContent / cancel messages are sent.
                if pre_block_backend_message is not None:
                    await self._send_to_backend(pre_block_backend_message)
                # Cancel any in-progress LLM response (e.g. VAD auto-response).
                await self._send_to_backend(json.dumps({"type": "response.cancel"}))
                # Send the policy violation hint (shows as small gray status text in UI).
                await self.websocket.send_text(
                    json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "type": "guardrail_violation",
                                "message": error_msg,
                                "code": "content_policy_violation",
                            },
                        }
                    )
                )
                # Ask the LLM to voice the exact guardrail message so the
                # user hears it as audio in voice sessions (not just text).
                guardrail_prompt = (
                    f"Say exactly the following message to the user, word for word, "
                    f"do not add anything else: {error_msg}"
                )
                await self._send_to_backend(
                    json.dumps(
                        {
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": guardrail_prompt}],
                            },
                        }
                    )
                )
                await self._send_to_backend(json.dumps({"type": "response.create"}))

                self._violation_count += 1
                end_session_after: Optional[int] = getattr(callback, "end_session_after_n_fails", None)
                should_end = getattr(callback, "on_violation", None) == "end_session" or (
                    end_session_after is not None and self._violation_count >= end_session_after
                )
                if should_end:
                    verbose_logger.warning(
                        "[realtime guardrail] ending session after violation %d",
                        self._violation_count,
                    )
                    await self.backend_ws.close()  # type: ignore[union-attr, attr-defined]

                verbose_logger.warning(
                    "[realtime guardrail] BLOCKED transcript (violation %d): %r",
                    self._violation_count,
                    transcript[:80],
                )
                return True
        return False

    async def _handle_provider_config_message(self, raw_response) -> None:
        """Process a backend message when a provider_config is set (transformed path)."""
        returned_object = self.provider_config.transform_realtime_response(  # type: ignore[union-attr]
            raw_response,
            self.model,
            self.logging_obj,
            realtime_response_transform_input={
                "session_configuration_request": self.session_configuration_request,
                "current_output_item_id": self.current_output_item_id,
                "current_response_id": self.current_response_id,
                "current_delta_chunks": self.current_delta_chunks,
                "current_conversation_id": self.current_conversation_id,
                "current_item_chunks": self.current_item_chunks,
                "current_delta_type": self.current_delta_type,
            },
        )

        transformed_response = returned_object["response"]
        self.current_output_item_id = returned_object["current_output_item_id"]
        self.current_response_id = returned_object["current_response_id"]
        self.current_delta_chunks = returned_object["current_delta_chunks"]
        self.current_conversation_id = returned_object["current_conversation_id"]
        self.current_item_chunks = returned_object["current_item_chunks"]
        self.current_delta_type = returned_object["current_delta_type"]
        self.session_configuration_request = returned_object["session_configuration_request"]
        events = transformed_response if isinstance(transformed_response, list) else [transformed_response]
        for event in events:
            if self._should_drop_event_from_client(event):
                continue
            is_session_created_event = isinstance(event, dict) and event.get("type") == "session.created"
            if is_session_created_event:
                if self._uses_deferred_backend_setup() and not self._backend_setup_complete:
                    self._backend_setup_complete = True
                    self._flushing_pending_messages_until_setup = True
                    try:
                        while self._pending_messages_until_setup:
                            flushed = await self._flush_pending_messages_until_setup()
                            if not flushed:
                                break
                    finally:
                        self._flushing_pending_messages_until_setup = False
                if self._session_created_sent_to_client:
                    # A synthetic session.created (with placeholder defaults) was
                    # already forwarded to the client when we connected.  The
                    # provider's real session.created (e.g. emitted from Gemini
                    # `setupComplete`) carries the authoritative modalities/model
                    # from the client's session.update.  Re-emit it as
                    # `session.updated` so the client learns the corrected
                    # configuration without seeing two `session.created` events.
                    event = {**event, "type": "session.updated"}
                else:
                    self._session_created_sent_to_client = True
            event_str = json.dumps(event)
            ## For audio/VAD guardrail path: forward the (possibly retyped)
            ## session.created first, then invoke the one-time guardrail
            ## turn-detection update.  ``_maybe_send_guardrail_turn_detection_update``
            ## is idempotent (gated by ``_guardrail_turn_detection_update_sent``),
            ## so duplicate session.created events — including those emitted
            ## after a synthetic session.created from ``llm_http_handler`` in
            ## deferred-setup mode — still get a single chance to inject the
            ## update if a prior attempt was dropped by the provider transform.
            if is_session_created_event and self._has_audio_transcription_guardrails():
                self.store_message(event_str)
                await self._send_event_to_client(event, event_str)
                await self._maybe_send_guardrail_turn_detection_update()
                continue
            ## GUARDRAIL: run on transcription events in provider_config path too
            if isinstance(event, dict) and event.get("type") == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "")
                self._collect_user_input_from_backend_event(cast(dict, event))
                self.store_message(event_str)
                await self._send_event_to_client(event, event_str)
                blocked = await self.run_realtime_guardrails(
                    cast(str, transcript),
                    item_id=cast(Optional[str], event.get("item_id")),
                )
                if not blocked:
                    await self._send_to_backend(json.dumps({"type": "response.create"}))
                continue
            ## LOGGING
            self.store_message(event_str)
            await self._send_event_to_client(event, event_str)

    @staticmethod
    def _parse_backend_event(raw_response: str) -> Optional[dict]:
        """Parse a backend frame once. Returns None for non-JSON or non-object frames."""
        try:
            event = json.loads(raw_response)
        except (json.JSONDecodeError, TypeError):
            return None
        return event if isinstance(event, dict) else None

    async def _handle_raw_backend_message(self, event_obj: dict, raw_response: str) -> bool:
        """Process a backend message without provider_config (raw path).

        Returns True if the caller should skip the default store+forward (i.e. continue the loop).
        """
        event_type = event_obj.get("type")

        self._detect_transcription_session_from_backend(event_obj)

        # Send session.created to the client FIRST so it stays in sync, then inject
        # the disable-auto-response session.update; otherwise a backend error could
        # reach the client before it sees session.created.
        if event_type == "session.created" and self._has_audio_transcription_guardrails():
            self.store_message(event_obj)
            await self.websocket.send_text(self._event_to_client_json(event_obj))
            await self._send_to_backend(self._make_disable_auto_response_message())
            return True

        if event_type == "conversation.item.input_audio_transcription.completed":
            transcript = event_obj.get("transcript", "")
            self._collect_user_input_from_backend_event(event_obj)
            self.store_message(event_obj)
            await self.websocket.send_text(self._event_to_client_json(event_obj))

            # Transcription-only sessions (e.g. gpt-realtime-whisper) have no
            # assistant turn: capture audio-duration usage for cost and never
            # trigger response.create.
            if self._is_transcription_session:
                self._capture_transcription_usage(event_obj)
                return True

            blocked = await self.run_realtime_guardrails(
                transcript,
                item_id=event_obj.get("item_id"),
            )
            if not blocked:
                await self._send_to_backend(json.dumps({"type": "response.create"}))
            return True
        return False

    async def backend_to_client_send_messages(self):
        import websockets

        try:
            while True:
                try:
                    raw_response = await self.backend_ws.recv(  # type: ignore[union-attr]
                        decode=False
                    )
                except TypeError:
                    raw_response = await self.backend_ws.recv()  # type: ignore[union-attr, assignment]

                if isinstance(raw_response, bytes):
                    try:
                        raw_response = raw_response.decode("utf-8")
                    except UnicodeDecodeError:
                        verbose_logger.warning("Received non-UTF-8 binary frame from backend, skipping.")
                        continue

                if self.provider_config:
                    try:
                        await self._handle_provider_config_message(raw_response)
                    except Exception as e:
                        verbose_logger.exception(f"Error processing backend message, skipping: {e}")
                        continue
                else:
                    event = self._parse_backend_event(raw_response)
                    if event is None:
                        await self.websocket.send_text(raw_response)
                        continue

                    if self._should_drop_event_from_client(event):
                        continue

                    if await self._handle_raw_backend_message(event, raw_response):
                        continue

                    event = self._normalize_event_for_ga_client(event)
                    self.store_message(event)

                    if not self._client_wants_beta:
                        await self.websocket.send_text(json.dumps(event))
                        continue

                    translated = self._translate_event_to_beta(event)
                    if translated is None:
                        continue
                    await self.websocket.send_text(json.dumps(translated))

        except websockets.exceptions.ConnectionClosed as e:  # type: ignore
            verbose_logger.exception(f"Connection closed in backend to client send messages - {e}")
        except Exception as e:
            verbose_logger.exception(f"Error in backend to client send messages: {e}")
        finally:
            await self.log_messages()

    @staticmethod
    def _detect_beta_header(websocket: Any) -> bool:
        """Return True if the client sent 'OpenAI-Beta: realtime=v1'.

        Checks the raw ASGI scope headers so it works for both FastAPI WebSocket
        objects and any test doubles that expose a .scope dict.
        """
        try:
            headers = websocket.scope.get("headers", [])
            for name, value in headers:
                if isinstance(name, bytes):
                    name = name.decode("latin-1")
                if isinstance(value, bytes):
                    value = value.decode("latin-1")
                if name.lower() == "openai-beta" and "realtime=v1" in value.lower():
                    return True
        except Exception:
            pass
        return False

    @staticmethod
    def _remap_beta_session_to_ga(session: dict) -> dict:
        """
        Convert a beta-style session.update payload to the GA nested schema.

        Beta → GA field mappings
        ─────────────────────────────────────────────────────────────────────
        session.type                    (inject "realtime" if absent)
        session.modalities              → session.output_modalities
        session.voice                   → session.audio.output.voice
        session.input_audio_format      → session.audio.input.format  (with type/rate)
        session.output_audio_format     → session.audio.output.format (with type/rate)
        session.turn_detection          → session.audio.input.turn_detection
        session.input_audio_transcription → session.audio.input.transcription
        ─────────────────────────────────────────────────────────────────────
        Fields not in the mapping (instructions, tools, etc.) are passed through.
        GA clients that already use the nested shape are unaffected.
        """
        # Work on a shallow copy so we don't mutate the caller's dict
        session = dict(session)

        # 1. Ensure session.type is present
        if "type" not in session:
            session["type"] = "realtime"

        # 2. Rename modalities → output_modalities and normalise combinations.
        # Beta allowed ["audio", "text"] together; GA only supports ["audio"] or
        # ["text"] as single-element lists. When both are present we prefer
        # ["audio"] because audio mode already delivers transcripts via events.
        if "modalities" in session:
            mods = session.pop("modalities")
            if "output_modalities" not in session:
                mods_set = {m.lower() for m in (mods or [])}
                if "audio" in mods_set:
                    session["output_modalities"] = ["audio"]
                elif "text" in mods_set:
                    session["output_modalities"] = ["text"]

        # 3-7. Lift flat audio fields into the nested audio object
        audio: Dict[str, Any] = {}
        inp: Dict[str, Any] = {}
        out: Dict[str, Any] = {}

        # voice → audio.output.voice
        if "voice" in session:
            out["voice"] = session.pop("voice")

        # input_audio_format → audio.input.format
        if "input_audio_format" in session:
            raw = session.pop("input_audio_format")
            inp["format"] = RealTimeStreaming._AUDIO_FORMAT_MAP.get(raw, raw) if isinstance(raw, str) else raw

        # output_audio_format → audio.output.format
        if "output_audio_format" in session:
            raw = session.pop("output_audio_format")
            out["format"] = RealTimeStreaming._AUDIO_FORMAT_MAP.get(raw, raw) if isinstance(raw, str) else raw

        # turn_detection → audio.input.turn_detection
        if "turn_detection" in session:
            inp["turn_detection"] = session.pop("turn_detection")

        # input_audio_transcription → audio.input.transcription
        if "input_audio_transcription" in session:
            inp["transcription"] = session.pop("input_audio_transcription")

        if inp:
            audio["input"] = inp
        if out:
            audio["output"] = out

        if audio:
            # Merge with any existing GA-style `audio` block the client already set,
            # letting the remapped values take precedence within each sub-key.
            existing = session.get("audio") or {}
            for sub_key, sub_val in audio.items():
                if sub_key in existing and isinstance(existing[sub_key], dict) and isinstance(sub_val, dict):
                    existing[sub_key] = {**existing[sub_key], **sub_val}
                else:
                    existing[sub_key] = sub_val
            session["audio"] = existing

        return session

    @staticmethod
    def _translate_event_to_beta(event: dict) -> Optional[dict]:
        """Translate a single GA event dict to its beta equivalent.

        Returns None when the event must be dropped (the GA-only
        conversation.item.done has no beta counterpart). Returns the original
        event object unchanged when no translation applies; otherwise returns a
        translated copy.
        """
        event_type = event.get("type", "")

        if event_type == "conversation.item.done":
            return None

        renamed_type = RealTimeStreaming._GA_TO_BETA_EVENT_TYPES.get(event_type)
        has_item = isinstance(event.get("item"), dict)
        response = event.get("response")
        has_response_output = isinstance(response, dict) and isinstance(response.get("output"), list)
        if renamed_type is None and not has_item and not has_response_output:
            return event

        translated = dict(event)
        if renamed_type is not None:
            translated["type"] = renamed_type
        if has_item:
            translated["item"] = RealTimeStreaming._translate_item_content_types(dict(translated["item"]))
        if has_response_output:
            resp = dict(translated["response"])
            resp["output"] = [
                (RealTimeStreaming._translate_item_content_types(dict(o)) if isinstance(o, dict) else o)
                for o in resp["output"]
            ]
            translated["response"] = resp

        return translated

    @staticmethod
    def _translate_item_content_types(item: dict) -> dict:
        """Replace GA content type names with beta names inside a single item."""
        if "content" not in item or not isinstance(item["content"], list):
            return item
        new_content = []
        for block in item["content"]:
            if isinstance(block, dict) and block.get("type") in RealTimeStreaming._GA_TO_BETA_CONTENT_TYPES:
                block = dict(block)
                block["type"] = RealTimeStreaming._GA_TO_BETA_CONTENT_TYPES[block["type"]]
            new_content.append(block)
        item["content"] = new_content
        return item

    async def client_ack_messages(self):
        try:
            while True:
                message = await self.websocket.receive_text()

                ## GUARDRAIL: intercept conversation.item.create for text-based injection.
                guardrail_turn_detection_injected = False
                msg_type: Optional[str] = None
                try:
                    from litellm.types.guardrails import GuardrailEventHooks

                    msg_obj = json.loads(message)
                    msg_type = msg_obj.get("type")

                    if msg_type == "conversation.item.create":
                        # Check user text messages for prompt injection
                        item = msg_obj.get("item", {})
                        # Check function_call_output first so a client cannot
                        # bypass the tool-result guardrail by also setting
                        # role="user" on a function_call_output item.
                        if item.get("type") == "function_call_output":
                            # Tool results are client-controlled and fed to the
                            # model; check them with the same guardrail used for
                            # user text so an attacker cannot smuggle blocked
                            # content into a function_call_output.
                            output = item.get("output", "")
                            output_text = output if isinstance(output, str) else json.dumps(output)
                            if output_text:
                                # Build the sanitized function_call_output up
                                # front so we can hand it to the guardrail
                                # runner as the pre-block message. Providers
                                # that pair every toolCall with a toolResponse
                                # (e.g. Gemini/Vertex Live) require the
                                # toolResponse to arrive BEFORE any other
                                # client message — otherwise the guardrail's
                                # own clientContent would violate the
                                # pending-tool-call protocol contract and the
                                # backend could close the connection before
                                # the sanitized response ever lands. Dropping
                                # the blocked item outright would similarly
                                # leave such providers waiting indefinitely.
                                # The sanitized payload carries no blocked
                                # content — only a generic policy marker.
                                sanitized_msg = json.dumps(
                                    {
                                        **msg_obj,
                                        "item": {
                                            **item,
                                            "output": json.dumps(
                                                {
                                                    "error": "Tool output blocked by content policy",
                                                }
                                            ),
                                        },
                                    }
                                )
                                blocked = await self.run_realtime_guardrails(
                                    output_text,
                                    pre_block_backend_message=sanitized_msg,
                                    event_hooks=[GuardrailEventHooks.pre_call],
                                )
                                if blocked:
                                    # ``_pending_guardrail_message`` is
                                    # intentionally NOT set here. That flag
                                    # exists to swallow the reflexive
                                    # ``response.create`` an OpenAI client
                                    # sends immediately after a user text
                                    # message. In a tool-calling flow the
                                    # client may not send a ``response.create``
                                    # at all (e.g. Gemini SDKs auto-respond),
                                    # so leaving the flag set would
                                    # incorrectly drop an unrelated
                                    # ``response.create`` from a later
                                    # interaction turn.
                                    continue
                        elif item.get("role") == "user":
                            content_list = item.get("content", [])
                            texts = [
                                c.get("text", "")
                                for c in content_list
                                if isinstance(c, dict) and c.get("type") == "input_text"
                            ]
                            combined_text = " ".join(texts)
                            if combined_text:
                                blocked = await self.run_realtime_guardrails(
                                    combined_text,
                                    event_hooks=[GuardrailEventHooks.pre_call],
                                )
                                if blocked:
                                    # Store the guardrail reason so the next response.create
                                    # (sent automatically by the client) is rewritten to
                                    # include it as response instructions.
                                    self._pending_guardrail_message = combined_text
                                    continue  # don't forward the original blocked message

                    if msg_type == "response.create" and self._pending_guardrail_message:
                        # The guardrail already sent the synthetic AI bubble — drop this
                        # response.create so OpenAI doesn't generate an additional response.
                        self._pending_guardrail_message = None
                        continue

                    ## GUARDRAIL: Inject turn_detection into first session.update
                    # if needed. Done BEFORE the GA remap so the injected
                    # ``create_response`` rides along with any client-provided
                    # turn_detection fields (e.g. silence_duration_ms) into the
                    # nested ``audio.input.turn_detection`` path produced by the
                    # remap. Doing this after the remap would create a separate
                    # minimal root-level ``turn_detection`` and silently drop
                    # the client's nested settings.
                    if (
                        msg_type == "session.update"
                        and self.session_configuration_request is None
                        and not self._guardrail_turn_detection_update_sent
                        and self._has_audio_transcription_guardrails()
                    ):
                        session = msg_obj.setdefault("session", {})
                        if isinstance(session, dict):
                            existing_td = session.get("turn_detection")
                            if not isinstance(existing_td, dict):
                                existing_td = {}
                            existing_td["create_response"] = False
                            session["turn_detection"] = existing_td
                            message = json.dumps(msg_obj)
                            guardrail_turn_detection_injected = True
                            verbose_logger.debug(
                                "Injected turn_detection into first session.update for audio transcription guardrails"
                            )

                    ## GUARDRAIL: Force ``create_response`` to False in any
                    # client-provided ``turn_detection`` so a later
                    # ``session.update`` cannot re-enable VAD auto-response
                    # and bypass the transcription guardrail after the
                    # initial disable. Covers both the flat beta key and the
                    # nested GA ``audio.input.turn_detection`` shape, since
                    # the GA remap below also accepts either form. Skipped
                    # when the injection block above already ran for this
                    # message, to avoid redundant double-serialization.
                    if (
                        msg_type == "session.update"
                        and not guardrail_turn_detection_injected
                        and self._has_audio_transcription_guardrails()
                    ):
                        session = msg_obj.get("session")
                        if isinstance(session, dict):
                            td_overridden = False
                            flat_td = session.get("turn_detection")
                            flat_td_present = flat_td is not None
                            if flat_td_present:
                                if not isinstance(flat_td, dict):
                                    flat_td = {}
                                if flat_td.get("create_response") is not False:
                                    flat_td["create_response"] = False
                                    session["turn_detection"] = flat_td
                                    td_overridden = True
                            nested_td_present = False
                            audio = session.get("audio")
                            if isinstance(audio, dict):
                                audio_input = audio.get("input")
                                if isinstance(audio_input, dict):
                                    nested_td = audio_input.get("turn_detection")
                                    if nested_td is not None:
                                        nested_td_present = True
                                        if not isinstance(nested_td, dict):
                                            nested_td = {}
                                        if nested_td.get("create_response") is not False:
                                            nested_td["create_response"] = False
                                            audio_input["turn_detection"] = nested_td
                                            td_overridden = True
                            # Symmetric with the first-update injection block:
                            # if the client omitted turn_detection entirely on
                            # a subsequent session.update, still inject the
                            # ``create_response: False`` override so the
                            # transcription guardrail cannot be re-enabled by
                            # any downstream merge that drops the original
                            # disable.
                            if not flat_td_present and not nested_td_present:
                                session["turn_detection"] = {"create_response": False}
                                td_overridden = True
                            if td_overridden:
                                message = json.dumps(msg_obj)

                    # GA compatibility: remap beta-style session fields only when
                    # the upstream is in GA mode. Beta upstreams expect the flat
                    # session shape unchanged.
                    if msg_type == "session.update" and not self._backend_uses_beta_protocol:
                        session = msg_obj.get("session", {})
                        if isinstance(session, dict):
                            session = self._remap_beta_session_to_ga(session)
                            msg_obj["session"] = session
                            message = json.dumps(msg_obj)

                    if msg_type == "session.update" and self._event_normalizer:
                        session = msg_obj.get("session")
                        if isinstance(session, dict):
                            msg_obj["session"] = self._event_normalizer.patch_outgoing_session(session)
                            message = json.dumps(msg_obj)

                except (json.JSONDecodeError, AttributeError):
                    pass

                ## LOGGING
                # Log after any in-place modifications (GA remap, guardrail
                # turn_detection injection) so audit logs reflect what we
                # actually forward to the backend.
                self.store_input(message=message)

                if self._should_buffer_client_message_until_setup(message):
                    self._buffer_pending_message_until_setup(message)
                    continue

                if self._pending_messages_until_setup:
                    should_send_setup_before_buffered_messages = (
                        not self._backend_setup_complete
                        and not self._flushing_pending_messages_until_setup
                        and msg_type == "session.update"
                    )
                    if not should_send_setup_before_buffered_messages:
                        self._buffer_pending_message_until_setup(message)
                        if self._backend_setup_complete and not self._flushing_pending_messages_until_setup:
                            await self._flush_pending_messages_until_setup()
                        continue

                if self._flushing_pending_messages_until_setup:
                    self._buffer_pending_message_until_setup(message)
                    continue

                ## FORWARD TO BACKEND
                # Only mark the guardrail turn_detection update as sent after the
                # backend actually accepted the message. Setting the flag earlier
                # would permanently disable the injection if ``_send_to_backend``
                # raised — neither this loop nor
                # ``_maybe_send_guardrail_turn_detection_update`` would retry.
                sent = await self._send_to_backend(message)
                if guardrail_turn_detection_injected and sent:
                    self._guardrail_turn_detection_update_sent = True

        except Exception as e:
            verbose_logger.debug(f"Error in client ack messages: {e}")

    async def bidirectional_forward(self):
        forward_task = asyncio.create_task(self.backend_to_client_send_messages())
        try:
            await self.client_ack_messages()
        except self.websocket.exceptions.ConnectionClosed:  # type: ignore
            verbose_logger.debug("Connection closed")
            forward_task.cancel()
        finally:
            if not forward_task.done():
                forward_task.cancel()
                try:
                    await forward_task
                except asyncio.CancelledError:
                    pass


def client_sent_openai_beta_realtime_header(websocket: Any) -> bool:
    """True when the client WebSocket includes ``OpenAI-Beta: realtime=v1``."""
    return RealTimeStreaming._detect_beta_header(websocket)

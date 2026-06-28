import base64
import binascii
import json
import math
import struct
from typing import TYPE_CHECKING, Any, cast

from litellm._uuid import uuid
from litellm.llms.base_llm.realtime.transformation import (
    BaseRealtimeConfig,
    RealtimeMessage,
)
from litellm.llms.volcengine.common_utils import VolcEngineError
from litellm.llms.volcengine.realtime.protocol import (
    EV_ASR_ENDED,
    EV_ASR_INFO,
    EV_ASR_RESPONSE,
    EV_CHAT_RESPONSE,
    EV_CONFIG_UPDATED,
    EV_CONNECTION_FAILED,
    EV_CONNECTION_STARTED,
    EV_DIALOG_COMMON_ERROR,
    EV_FINISH_CONNECTION,
    EV_SAY_HELLO,
    EV_SESSION_FAILED,
    EV_SESSION_FINISHED,
    EV_SESSION_STARTED,
    EV_START_CONNECTION,
    EV_START_SESSION,
    EV_TASK_REQUEST,
    EV_TTS_ENDED,
    EV_UPDATE_CONFIG,
    MSG_AUDIO_SERVER,
    MSG_ERROR,
    RealtimeFrame,
    decode_realtime_frame,
    encode_audio_event,
    encode_json_event,
    parse_json_payload,
    parse_payload,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    OpenAIRealtimeContentPartDone,
    OpenAIRealtimeDoneEvent,
    OpenAIRealtimeEvents,
    OpenAIRealtimeOutputItemDone,
    OpenAIRealtimeResponseAudioDone,
    OpenAIRealtimeResponseContentPartAdded,
    OpenAIRealtimeResponseDelta,
    OpenAIRealtimeResponseDoneObject,
    OpenAIRealtimeStreamResponseOutputItem,
    OpenAIRealtimeStreamResponseOutputItemAdded,
    OpenAIRealtimeStreamSession,
    OpenAIRealtimeStreamSessionEvents,
)
from litellm.types.realtime import (
    RealtimeResponseTransformInput,
    RealtimeResponseTypedDict,
)
from litellm.utils import get_empty_usage

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


VOLCENGINE_REALTIME_DEFAULT_API_BASE = (
    "wss://openspeech.bytedance.com/api/v3/realtime/dialogue"
)
VOLCENGINE_REALTIME_DEFAULT_RESOURCE_ID = "volc.speech.dialog"
VOLCENGINE_REALTIME_DEFAULT_APP_KEY = "PlgvMymc7f3tQnJ6"
VOLCENGINE_REALTIME_DEFAULT_MODEL_VERSION = "1.2.1.1"
VOLCENGINE_REALTIME_DEFAULT_BOT_NAME = "助手"
VOLCENGINE_REALTIME_DEFAULT_SPEAKER = "zh_female_vv_jupiter_bigtts"
VOLCENGINE_REALTIME_DEFAULT_END_SMOOTH_WINDOW_MS = 1500
VOLCENGINE_REALTIME_INPUT_SAMPLE_RATE_HZ = 16000
VOLCENGINE_REALTIME_OUTPUT_SAMPLE_RATE_HZ = 24000
VOLCENGINE_REALTIME_STARTUP_SANITIZE_MS = 250
VOLCENGINE_REALTIME_FADE_IN_MS = 20
VOLCENGINE_REALTIME_STARTUP_SILENCE_THRESHOLD = 64
VOLCENGINE_REALTIME_SENTINEL_THRESHOLD = 32760
OPENAI_REALTIME_VOICE_NAMES = {
    "alloy",
    "ash",
    "ballad",
    "cedar",
    "coral",
    "echo",
    "fable",
    "marin",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
}


class VolcEngineRealtimeConfig(BaseRealtimeConfig):
    def __init__(self) -> None:
        super().__init__()
        self._session_id = str(uuid.uuid4())
        self._session_started = False
        self._latest_session: dict[str, Any] = {}
        self._input_sample_rate_hz: int | None = None
        self._current_user_item_id: str | None = None
        self._current_user_transcript = ""
        self._current_assistant_transcript = ""
        self._pending_response_metadata: dict[str, Any] | None = None
        self._response_audio_startup_samples_seen = 0
        self._response_audio_fade_samples_applied = 0
        self._response_audio_started = False

    def validate_environment(
        self, headers: dict, model: str, api_key: str | None = None
    ) -> dict:
        resolved_headers: dict[str, Any] = {
            key: value
            for key, value in (headers or {}).items()
            if key.lower().startswith("x-api-")
        }
        _setdefault_header(
            resolved_headers, "X-Api-Resource-Id", pick_realtime_resource_id(model)
        )
        _setdefault_header(resolved_headers, "X-Api-App-Key", _get_realtime_app_key())
        _setdefault_header(resolved_headers, "X-Api-Connect-Id", str(uuid.uuid4()))

        if _has_volcengine_auth_headers(resolved_headers):
            return resolved_headers

        app_id, access_key = _get_app_id_access_key(api_key)
        if app_id and access_key:
            resolved_headers["X-Api-App-ID"] = app_id
            resolved_headers["X-Api-Access-Key"] = access_key
            return resolved_headers

        volcengine_api_key = get_secret_str("VOLCENGINE_API_KEY")
        specific_speech_key = get_secret_str(
            "VOLCENGINE_REALTIME_API_KEY"
        ) or get_secret_str("VOLCENGINE_SPEECH_KEY")
        speech_key = api_key
        if specific_speech_key and (not api_key or api_key == volcengine_api_key):
            speech_key = specific_speech_key
        if speech_key and speech_key.strip():
            resolved_headers["X-Api-Key"] = speech_key.strip()
            return resolved_headers

        raise VolcEngineError(
            status_code=401,
            message=(
                "Volcengine Realtime credentials are required. Set "
                "VOLCENGINE_REALTIME_API_KEY / VOLCENGINE_SPEECH_KEY, or set "
                "legacy VOLCENGINE_REALTIME_APP_ID / "
                "VOLCENGINE_REALTIME_ACCESS_KEY credentials."
            ),
        )

    def get_complete_url(
        self, api_base: str | None, model: str, api_key: str | None = None
    ) -> str:
        if api_base and api_base.startswith(("ws://", "wss://")):
            return api_base
        return VOLCENGINE_REALTIME_DEFAULT_API_BASE

    def requires_session_configuration(self) -> bool:
        return True

    def session_configuration_request(self, model: str) -> RealtimeMessage | None:
        return encode_json_event(event=EV_START_CONNECTION, payload={})

    def transform_realtime_request(
        self,
        message: str,
        model: str,
        session_configuration_request: RealtimeMessage | None = None,
    ) -> list[RealtimeMessage]:
        try:
            message_obj = json.loads(message)
        except (json.JSONDecodeError, TypeError):
            return []

        message_type = message_obj.get("type")
        if message_type == "session.update":
            session = message_obj.get("session") or {}
            if isinstance(session, dict):
                self._latest_session = _deep_merge(self._latest_session, session)
                self._input_sample_rate_hz = _extract_input_sample_rate_hz(
                    self._latest_session
                )
            else:
                session = {}
            if self._session_started:
                update_payload = _update_config_payload(
                    model=model,
                    latest_session=self._latest_session,
                    updated_session=session,
                )
                if not update_payload:
                    return []
                return cast(
                    list[RealtimeMessage],
                    [
                        encode_json_event(
                            event=EV_UPDATE_CONFIG,
                            session_id=self._session_id,
                            payload=update_payload,
                        )
                    ],
                )
            return self._start_session_frame(model=model, session=self._latest_session)

        if message_type == "input_audio_buffer.append":
            frames: list[RealtimeMessage] = []
            if not self._session_started:
                frames.extend(
                    self._start_session_frame(
                        model=model, session=self._latest_session or {}
                    )
                )
            audio = message_obj.get("audio")
            if not isinstance(audio, str) or not audio:
                return frames
            try:
                audio_bytes = base64.b64decode(audio)
            except (binascii.Error, ValueError):
                return frames
            audio_bytes = _normalise_input_audio(
                audio_bytes, self._input_sample_rate_hz
            )
            frames.append(
                encode_audio_event(
                    event=EV_TASK_REQUEST,
                    session_id=self._session_id,
                    payload=audio_bytes,
                )
            )
            return frames

        if message_type == "response.create":
            self._pending_response_metadata = _extract_response_create_metadata(
                message_obj
            )
            instructions = _extract_response_create_instructions(message_obj)
            if not instructions:
                return []
            frames: list[RealtimeMessage] = []
            if not self._session_started:
                frames.extend(
                    self._start_session_frame(
                        model=model, session=self._latest_session or {}
                    )
                )
            frames.append(
                encode_json_event(
                    event=EV_SAY_HELLO,
                    session_id=self._session_id,
                    payload={"content": instructions},
                )
            )
            return frames

        if message_type == "session.close":
            return [
                encode_json_event(
                    event=EV_FINISH_CONNECTION,
                    payload={},
                )
            ]

        return []

    def transform_session_created_event(
        self,
        model: str,
        logging_session_id: str,
        session_configuration_request: RealtimeMessage | None = None,
    ) -> OpenAIRealtimeStreamSessionEvents:
        return self._session_event("session.created", model)

    def transform_realtime_response(
        self,
        message: str | bytes,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        realtime_response_transform_input: RealtimeResponseTransformInput,
    ) -> RealtimeResponseTypedDict:
        current_output_item_id = realtime_response_transform_input[
            "current_output_item_id"
        ]
        current_response_id = realtime_response_transform_input["current_response_id"]
        current_delta_chunks = realtime_response_transform_input["current_delta_chunks"]
        current_conversation_id = realtime_response_transform_input[
            "current_conversation_id"
        ]
        current_item_chunks = realtime_response_transform_input["current_item_chunks"]
        current_delta_type = realtime_response_transform_input["current_delta_type"]
        session_configuration_request = realtime_response_transform_input[
            "session_configuration_request"
        ]

        if isinstance(message, str):
            message = message.encode("utf-8")
        frame = decode_realtime_frame(bytes(message))
        returned_message: list[OpenAIRealtimeEvents] = []

        if frame.message_type == MSG_ERROR:
            returned_message.append(
                _error_event(
                    code=str(frame.error_code or "volcengine_realtime_error"),
                    message=_safe_payload_detail(frame),
                )
            )
        elif frame.message_type == MSG_AUDIO_SERVER and frame.payload:
            (
                audio_events,
                current_output_item_id,
                current_response_id,
                current_conversation_id,
            ) = self._audio_delta_events(
                payload=parse_payload(frame),
                current_output_item_id=current_output_item_id,
                current_response_id=current_response_id,
                current_conversation_id=current_conversation_id,
            )
            current_delta_type = "audio"
            returned_message.extend(audio_events)
        else:
            (
                returned_message,
                current_output_item_id,
                current_response_id,
                current_conversation_id,
                current_delta_type,
            ) = self._non_audio_frame_events(
                frame=frame,
                model=model,
                current_output_item_id=current_output_item_id,
                current_response_id=current_response_id,
                current_conversation_id=current_conversation_id,
                current_delta_type=current_delta_type,
            )

        if returned_message:
            current_delta_chunks = _update_delta_chunks(
                returned_message=returned_message,
                current_delta_chunks=current_delta_chunks,
            )
            current_item_chunks = _update_item_chunks(returned_message)

        return {
            "response": returned_message,
            "current_output_item_id": current_output_item_id,
            "current_response_id": current_response_id,
            "current_delta_chunks": current_delta_chunks,
            "current_conversation_id": current_conversation_id,
            "current_item_chunks": current_item_chunks,
            "current_delta_type": current_delta_type,
            "session_configuration_request": session_configuration_request,
        }

    def _non_audio_frame_events(
        self,
        frame: RealtimeFrame,
        model: str,
        current_output_item_id: str | None,
        current_response_id: str | None,
        current_conversation_id: str | None,
        current_delta_type: str | None,
    ) -> tuple[
        list[OpenAIRealtimeEvents],
        str | None,
        str | None,
        str | None,
        str | None,
    ]:
        if frame.event == EV_CONNECTION_STARTED:
            return (
                [self._session_event("session.created", model)],
                current_output_item_id,
                current_response_id,
                current_conversation_id,
                current_delta_type,
            )
        if frame.event in {EV_SESSION_STARTED, EV_CONFIG_UPDATED}:
            return (
                [self._session_event("session.updated", model)],
                current_output_item_id,
                current_response_id,
                current_conversation_id,
                current_delta_type,
            )
        if frame.event in {
            EV_CONNECTION_FAILED,
            EV_SESSION_FAILED,
            EV_DIALOG_COMMON_ERROR,
        }:
            return (
                [
                    _error_event(
                        code=f"volcengine_event_{frame.event}",
                        message=_safe_payload_detail(frame),
                    )
                ],
                current_output_item_id,
                current_response_id,
                current_conversation_id,
                current_delta_type,
            )
        if frame.event in {EV_TTS_ENDED, EV_SESSION_FINISHED}:
            if current_output_item_id and current_response_id:
                done_events = self._audio_done_events(
                    current_output_item_id=current_output_item_id,
                    current_response_id=current_response_id,
                    current_conversation_id=current_conversation_id,
                    transcript=self._current_assistant_transcript,
                )
                self._current_assistant_transcript = ""
                return done_events, None, None, current_conversation_id, None
            return (
                [],
                current_output_item_id,
                current_response_id,
                current_conversation_id,
                current_delta_type,
            )
        if frame.event in {EV_ASR_INFO, EV_ASR_RESPONSE, EV_ASR_ENDED}:
            return (
                self._asr_events(frame),
                current_output_item_id,
                current_response_id,
                current_conversation_id,
                current_delta_type,
            )
        if frame.event == EV_CHAT_RESPONSE:
            assistant_text = _chat_response_text(frame)
            if not assistant_text:
                return (
                    [],
                    current_output_item_id,
                    current_response_id,
                    current_conversation_id,
                    current_delta_type,
                )
            self._current_assistant_transcript += assistant_text
            (
                transcript_events,
                current_output_item_id,
                current_response_id,
                current_conversation_id,
            ) = self._audio_transcript_delta_events(
                text=assistant_text,
                current_output_item_id=current_output_item_id,
                current_response_id=current_response_id,
                current_conversation_id=current_conversation_id,
            )
            return (
                transcript_events,
                current_output_item_id,
                current_response_id,
                current_conversation_id,
                "audio",
            )
        return (
            [],
            current_output_item_id,
            current_response_id,
            current_conversation_id,
            current_delta_type,
        )

    def _asr_events(self, frame: RealtimeFrame) -> list[OpenAIRealtimeEvents]:
        if frame.event == EV_ASR_INFO:
            if self._current_user_item_id is None:
                self._current_user_item_id = f"item_{uuid.uuid4()}"
            return [_input_speech_started_event(self._current_user_item_id)]
        if frame.event == EV_ASR_RESPONSE:
            final_text = _asr_final_text(frame)
            if final_text:
                self._current_user_transcript += final_text
            return []
        if frame.event == EV_ASR_ENDED:
            current_transcript = self._current_user_transcript
            self._current_user_transcript = ""
            current_item_id = self._current_user_item_id
            self._current_user_item_id = None
            if current_transcript:
                return [
                    _input_transcript_event(
                        current_transcript,
                        item_id=current_item_id,
                    )
                ]
        return []

    def _start_session_frame(self, model: str, session: Any) -> list[RealtimeMessage]:
        if not isinstance(session, dict):
            session = {}
        self._session_started = True
        return [
            encode_json_event(
                event=EV_START_SESSION,
                session_id=self._session_id,
                payload=_start_session_payload(model=model, session=session),
            )
        ]

    def _session_event(
        self, event_type: str, model: str
    ) -> OpenAIRealtimeStreamSessionEvents:
        session = OpenAIRealtimeStreamSession(
            id=self._session_id,
            modalities=["audio"],
            model=model,
            input_audio_format="pcm16",
            output_audio_format="pcm16",
        )
        instructions = self._latest_session.get("instructions")
        if isinstance(instructions, str):
            session["instructions"] = instructions
        return OpenAIRealtimeStreamSessionEvents(
            type=cast(Any, event_type),
            session=session,
            event_id=f"event_{uuid.uuid4()}",
        )

    def _audio_delta_events(
        self,
        payload: bytes,
        current_output_item_id: str | None,
        current_response_id: str | None,
        current_conversation_id: str | None,
    ) -> tuple[list[OpenAIRealtimeEvents], str, str, str]:
        events: list[OpenAIRealtimeEvents] = []
        current_response_id = current_response_id or f"resp_{uuid.uuid4()}"
        current_conversation_id = current_conversation_id or f"conv_{uuid.uuid4()}"

        if current_output_item_id is None:
            self._reset_audio_startup_filter()
            current_output_item_id = f"item_{uuid.uuid4()}"
            events.extend(
                _new_audio_response_events(
                    response_id=current_response_id,
                    output_item_id=current_output_item_id,
                    conversation_id=current_conversation_id,
                    metadata=self._consume_pending_response_metadata(),
                )
            )

        payload = self._smooth_response_startup_audio(payload)
        events.append(
            OpenAIRealtimeResponseDelta(
                type="response.output_audio.delta",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
                delta=base64.b64encode(payload).decode("ascii"),
            )
        )
        return (
            events,
            current_output_item_id,
            current_response_id,
            current_conversation_id,
        )

    def _audio_done_events(
        self,
        current_output_item_id: str,
        current_response_id: str,
        current_conversation_id: str | None,
        transcript: str = "",
    ) -> list[OpenAIRealtimeEvents]:
        done_event = OpenAIRealtimeResponseAudioDone(
            type="response.output_audio.done",
            content_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=current_output_item_id,
            output_index=0,
            response_id=current_response_id,
        )
        part_done = OpenAIRealtimeContentPartDone(
            type="response.content_part.done",
            content_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=current_output_item_id,
            output_index=0,
            part={"type": "output_audio", "transcript": transcript},
            response_id=current_response_id,
        )
        output_item = OpenAIRealtimeOutputItemDone(
            type="response.output_item.done",
            event_id=f"event_{uuid.uuid4()}",
            output_index=0,
            response_id=current_response_id,
            item=_assistant_audio_item(
                item_id=current_output_item_id,
                status="completed",
                transcript=transcript,
            ),
        )
        response_done = OpenAIRealtimeDoneEvent(
            type="response.done",
            event_id=f"event_{uuid.uuid4()}",
            response=OpenAIRealtimeResponseDoneObject(
                object="realtime.response",
                id=current_response_id,
                status="completed",
                output=[output_item["item"]],
                conversation_id=current_conversation_id or f"conv_{uuid.uuid4()}",
                modalities=["audio"],
                usage=_empty_usage_dict(),
            ),
        )
        return [done_event, part_done, output_item, response_done]

    def _audio_transcript_delta_events(
        self,
        text: str,
        current_output_item_id: str | None,
        current_response_id: str | None,
        current_conversation_id: str | None,
    ) -> tuple[list[OpenAIRealtimeEvents], str, str, str]:
        events: list[OpenAIRealtimeEvents] = []
        current_response_id = current_response_id or f"resp_{uuid.uuid4()}"
        current_conversation_id = current_conversation_id or f"conv_{uuid.uuid4()}"

        if current_output_item_id is None:
            self._reset_audio_startup_filter()
            current_output_item_id = f"item_{uuid.uuid4()}"
            events.extend(
                _new_audio_response_events(
                    response_id=current_response_id,
                    output_item_id=current_output_item_id,
                    conversation_id=current_conversation_id,
                    metadata=self._consume_pending_response_metadata(),
                )
            )

        events.append(
            OpenAIRealtimeResponseDelta(
                type="response.output_audio_transcript.delta",
                content_index=0,
                event_id=f"event_{uuid.uuid4()}",
                item_id=current_output_item_id,
                output_index=0,
                response_id=current_response_id,
                delta=text,
            )
        )
        return (
            events,
            current_output_item_id,
            current_response_id,
            current_conversation_id,
        )

    def _reset_audio_startup_filter(self) -> None:
        self._response_audio_startup_samples_seen = 0
        self._response_audio_fade_samples_applied = 0
        self._response_audio_started = False

    def _consume_pending_response_metadata(self) -> dict[str, Any] | None:
        metadata = self._pending_response_metadata
        self._pending_response_metadata = None
        return metadata

    def _smooth_response_startup_audio(self, payload: bytes) -> bytes:
        if len(payload) < 2:
            return payload

        sample_count = len(payload) // 2
        samples = list(struct.unpack(f"<{sample_count}h", payload[: sample_count * 2]))
        startup_samples = (
            VOLCENGINE_REALTIME_OUTPUT_SAMPLE_RATE_HZ
            * VOLCENGINE_REALTIME_STARTUP_SANITIZE_MS
            // 1000
        )
        fade_samples = (
            VOLCENGINE_REALTIME_OUTPUT_SAMPLE_RATE_HZ
            * VOLCENGINE_REALTIME_FADE_IN_MS
            // 1000
        )

        for index, sample in enumerate(samples):
            if self._response_audio_startup_samples_seen < startup_samples:
                if abs(sample) >= VOLCENGINE_REALTIME_SENTINEL_THRESHOLD:
                    sample = 0

            if not self._response_audio_started:
                if abs(sample) <= VOLCENGINE_REALTIME_STARTUP_SILENCE_THRESHOLD:
                    sample = 0
                else:
                    self._response_audio_started = True

            if (
                self._response_audio_started
                and self._response_audio_fade_samples_applied < fade_samples
            ):
                gain = (self._response_audio_fade_samples_applied + 1) / fade_samples
                sample = round(sample * gain)
                self._response_audio_fade_samples_applied += 1

            samples[index] = max(-32768, min(32767, sample))
            self._response_audio_startup_samples_seen += 1

        smoothed = struct.pack(f"<{sample_count}h", *samples)
        return smoothed + payload[sample_count * 2 :]


def pick_realtime_resource_id(model_name: str | None) -> str:
    model = _normalize_model_name(model_name)
    if model == VOLCENGINE_REALTIME_DEFAULT_RESOURCE_ID:
        return model
    return VOLCENGINE_REALTIME_DEFAULT_RESOURCE_ID


def _extract_response_create_instructions(message_obj: dict[str, Any]) -> str | None:
    response = message_obj.get("response")
    if isinstance(response, dict):
        instructions = response.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            return instructions.strip()

    instructions = message_obj.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        return instructions.strip()
    return None


def _extract_response_create_metadata(
    message_obj: dict[str, Any],
) -> dict[str, Any] | None:
    response = message_obj.get("response")
    if isinstance(response, dict):
        metadata = response.get("metadata")
        if isinstance(metadata, dict):
            return dict(metadata)

    metadata = message_obj.get("metadata")
    if isinstance(metadata, dict):
        return dict(metadata)
    return None


def _empty_usage_dict() -> dict[str, int]:
    usage = get_empty_usage()
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
    }


def _start_session_payload(model: str, session: dict[str, Any]) -> dict[str, Any]:
    volcengine_overrides = session.get("volcengine")
    if not isinstance(volcengine_overrides, dict):
        volcengine_overrides = {}

    payload: dict[str, Any] = {
        "tts": {
            "speaker": _extract_tts_speaker(session)
            or VOLCENGINE_REALTIME_DEFAULT_SPEAKER,
            "audio_config": {
                "channel": 1,
                "format": "pcm_s16le",
                "sample_rate": VOLCENGINE_REALTIME_OUTPUT_SAMPLE_RATE_HZ,
            },
            "extra": {},
        },
        "asr": {
            "audio_info": {
                "format": "pcm",
                "sample_rate": VOLCENGINE_REALTIME_INPUT_SAMPLE_RATE_HZ,
                "channel": 1,
            },
            "extra": {
                "end_smooth_window_ms": VOLCENGINE_REALTIME_DEFAULT_END_SMOOTH_WINDOW_MS
            },
        },
        "dialog": {
            "bot_name": volcengine_overrides.get("bot_name")
            or VOLCENGINE_REALTIME_DEFAULT_BOT_NAME,
            "system_role": session.get("instructions") or "",
            "speaking_style": volcengine_overrides.get("speaking_style") or "",
            "extra": {
                "strict_audit": False,
                "model": _pick_realtime_model_version(model),
            },
        },
    }
    if isinstance(volcengine_overrides.get("tts"), dict):
        payload["tts"] = _deep_merge(payload["tts"], volcengine_overrides["tts"])
    if isinstance(volcengine_overrides.get("asr"), dict):
        payload["asr"] = _deep_merge(payload["asr"], volcengine_overrides["asr"])
    if isinstance(volcengine_overrides.get("dialog"), dict):
        payload["dialog"] = _deep_merge(
            payload["dialog"], volcengine_overrides["dialog"]
        )
    return payload


def _update_config_payload(
    model: str, latest_session: dict[str, Any], updated_session: dict[str, Any]
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    full_config = _start_session_payload(model=model, session=latest_session)

    if _session_update_affects_dialog_config(updated_session):
        payload["dialog"] = _dialog_update_config(full_config["dialog"])
    if _session_update_affects_tts_config(updated_session):
        payload["tts"] = _tts_update_config(full_config["tts"])

    return payload


def _session_update_affects_dialog_config(session: dict[str, Any]) -> bool:
    if "instructions" in session:
        return True
    volcengine = session.get("volcengine")
    if not isinstance(volcengine, dict):
        return False
    return any(key in volcengine for key in ("bot_name", "speaking_style", "dialog"))


def _session_update_affects_tts_config(session: dict[str, Any]) -> bool:
    if "voice" in session:
        return True
    audio = session.get("audio")
    if isinstance(audio, dict):
        output = audio.get("output")
        if isinstance(output, dict) and "voice" in output:
            return True
    volcengine = session.get("volcengine")
    return isinstance(volcengine, dict) and "tts" in volcengine


def _dialog_update_config(dialog: dict[str, Any]) -> dict[str, Any]:
    update = {
        "bot_name": dialog.get("bot_name") or VOLCENGINE_REALTIME_DEFAULT_BOT_NAME,
        "system_role": dialog.get("system_role") or "",
        "speaking_style": dialog.get("speaking_style") or "",
    }
    for key in ("dialog_id", "character_manifest", "location"):
        if key in dialog:
            update[key] = dialog[key]
    return update


def _tts_update_config(tts: dict[str, Any]) -> dict[str, Any]:
    update: dict[str, Any] = {
        "speaker": tts.get("speaker") or VOLCENGINE_REALTIME_DEFAULT_SPEAKER
    }
    audio_config = tts.get("audio_config")
    if isinstance(audio_config, dict):
        mutable_audio_config = {
            key: audio_config[key]
            for key in ("speech_rate", "loudness_rate")
            if key in audio_config
        }
        if mutable_audio_config:
            update["audio_config"] = mutable_audio_config
    extra = tts.get("extra")
    if isinstance(extra, dict) and extra:
        update["extra"] = extra
    return update


def _extract_tts_speaker(session: dict[str, Any]) -> str | None:
    legacy_voice = session.get("voice")
    if isinstance(legacy_voice, str):
        speaker = _normalise_volcengine_speaker(legacy_voice)
        if speaker:
            return speaker

    audio = session.get("audio")
    if not isinstance(audio, dict):
        return None
    output = audio.get("output")
    if not isinstance(output, dict):
        return None
    voice = output.get("voice")
    if isinstance(voice, str):
        return _normalise_volcengine_speaker(voice)
    return None


def _normalise_volcengine_speaker(voice: str) -> str | None:
    speaker = voice.strip()
    if not speaker:
        return None
    if speaker.lower() in OPENAI_REALTIME_VOICE_NAMES:
        return None
    return speaker


def _extract_input_sample_rate_hz(session: dict[str, Any]) -> int | None:
    audio = session.get("audio")
    if not isinstance(audio, dict):
        return None
    input_audio = audio.get("input")
    if not isinstance(input_audio, dict):
        return None
    audio_format = input_audio.get("format")
    if not isinstance(audio_format, dict):
        return None
    rate = audio_format.get("rate")
    if isinstance(rate, int):
        return rate
    if isinstance(rate, str) and rate.isdigit():
        return int(rate)
    return None


def _pick_realtime_model_version(model: str | None) -> str:
    normalized = _normalize_model_name(model)
    if normalized in {"1.2.1.1", "2.2.0.0"}:
        return normalized
    return VOLCENGINE_REALTIME_DEFAULT_MODEL_VERSION


def _normalise_input_audio(audio_bytes: bytes, source_rate_hz: int | None) -> bytes:
    if source_rate_hz in (None, VOLCENGINE_REALTIME_INPUT_SAMPLE_RATE_HZ):
        return audio_bytes
    return _resample_pcm16_mono(
        audio_bytes,
        source_rate_hz=source_rate_hz,
        target_rate_hz=VOLCENGINE_REALTIME_INPUT_SAMPLE_RATE_HZ,
    )


def _resample_pcm16_mono(
    audio_bytes: bytes, source_rate_hz: int, target_rate_hz: int
) -> bytes:
    if source_rate_hz <= 0 or target_rate_hz <= 0:
        return audio_bytes
    sample_count = len(audio_bytes) // 2
    if sample_count == 0:
        return b""
    if len(audio_bytes) % 2:
        audio_bytes = audio_bytes[: sample_count * 2]

    source_samples = struct.unpack(f"<{sample_count}h", audio_bytes)
    target_sample_count = max(1, round(sample_count * target_rate_hz / source_rate_hz))
    if target_sample_count == sample_count:
        return audio_bytes

    output_samples: list[int] = []
    for target_index in range(target_sample_count):
        source_position = target_index * source_rate_hz / target_rate_hz
        left_index = int(math.floor(source_position))
        right_index = min(left_index + 1, sample_count - 1)
        fraction = source_position - left_index
        sample = round(
            source_samples[left_index]
            + (source_samples[right_index] - source_samples[left_index]) * fraction
        )
        output_samples.append(max(-32768, min(32767, int(sample))))
    return struct.pack(f"<{len(output_samples)}h", *output_samples)


def _new_audio_response_events(
    response_id: str,
    output_item_id: str,
    conversation_id: str,
    metadata: dict[str, Any] | None = None,
) -> list[OpenAIRealtimeEvents]:
    response: dict[str, Any] = {
        "object": "realtime.response",
        "id": response_id,
        "status": "in_progress",
        "status_details": None,
        "output": [],
        "conversation_id": conversation_id,
        "modalities": ["audio"],
    }
    if metadata is not None:
        response["metadata"] = metadata

    return [
        cast(
            OpenAIRealtimeEvents,
            {
                "type": "response.created",
                "event_id": f"event_{uuid.uuid4()}",
                "response": response,
            },
        ),
        OpenAIRealtimeStreamResponseOutputItemAdded(
            type="response.output_item.added",
            event_id=f"event_{uuid.uuid4()}",
            response_id=response_id,
            output_index=0,
            item=_assistant_audio_item(item_id=output_item_id, status="in_progress"),
        ),
        cast(
            OpenAIRealtimeEvents,
            {
                "type": "conversation.item.added",
                "event_id": f"event_{uuid.uuid4()}",
                "previous_item_id": None,
                "item": _assistant_audio_item(
                    item_id=output_item_id, status="in_progress"
                ),
            },
        ),
        OpenAIRealtimeResponseContentPartAdded(
            type="response.content_part.added",
            content_index=0,
            output_index=0,
            event_id=f"event_{uuid.uuid4()}",
            item_id=output_item_id,
            part={"type": "output_audio", "transcript": ""},
            response_id=response_id,
        ),
    ]


def _assistant_audio_item(
    item_id: str, status: str, transcript: str = ""
) -> OpenAIRealtimeStreamResponseOutputItem:
    return OpenAIRealtimeStreamResponseOutputItem(
        id=item_id,
        object="realtime.item",
        type="message",
        status=cast(Any, status),
        role="assistant",
        content=[{"type": "output_audio", "transcript": transcript}],
    )


def _error_event(code: str, message: str) -> OpenAIRealtimeEvents:
    return cast(
        OpenAIRealtimeEvents,
        {
            "type": "error",
            "event_id": f"event_{uuid.uuid4()}",
            "error": {
                "type": "server_error",
                "code": code,
                "message": message,
            },
        },
    )


def _update_delta_chunks(
    returned_message: list[OpenAIRealtimeEvents],
    current_delta_chunks: list[OpenAIRealtimeResponseDelta] | None,
) -> list[OpenAIRealtimeResponseDelta] | None:
    for event in returned_message:
        if event.get("type") == "response.output_audio.delta":
            if current_delta_chunks is None:
                current_delta_chunks = []
            current_delta_chunks.append(cast(OpenAIRealtimeResponseDelta, event))
    return current_delta_chunks


def _update_item_chunks(
    returned_message: list[OpenAIRealtimeEvents],
) -> list[OpenAIRealtimeOutputItemDone] | None:
    items = [
        cast(OpenAIRealtimeOutputItemDone, event)
        for event in returned_message
        if event.get("type") == "response.output_item.done"
    ]
    return items or None


def _get_app_id_access_key(
    api_key: str | None,
) -> tuple[str | None, str | None]:
    if api_key and ":" in api_key:
        app_id, access_key = api_key.split(":", 1)
        return app_id.strip() or None, access_key.strip() or None
    app_id = get_secret_str("VOLCENGINE_REALTIME_APP_ID") or get_secret_str(
        "DOUBAO_APP_ID"
    )
    access_key = get_secret_str("VOLCENGINE_REALTIME_ACCESS_KEY") or get_secret_str(
        "DOUBAO_ACCESS_KEY"
    )
    return app_id, access_key


def _has_volcengine_auth_headers(headers: dict[str, Any]) -> bool:
    normalized = {key.lower() for key in headers}
    return "x-api-key" in normalized or (
        "x-api-app-id" in normalized
        and "x-api-access-key" in normalized
        and "x-api-app-key" in normalized
    )


def _setdefault_header(headers: dict[str, Any], key: str, value: str) -> None:
    if not any(existing.lower() == key.lower() for existing in headers):
        headers[key] = value


def _get_realtime_app_key() -> str:
    return (
        get_secret_str("VOLCENGINE_REALTIME_APP_KEY")
        or get_secret_str("DOUBAO_APP_KEY")
        or VOLCENGINE_REALTIME_DEFAULT_APP_KEY
    )


def _safe_payload_detail(frame: Any) -> str:
    try:
        payload = parse_json_payload(frame)
        return str(payload.get("message") or payload.get("error") or payload)
    except (EOFError, OSError, UnicodeDecodeError, ValueError, VolcEngineError):
        return "unparseable error payload"


def _asr_final_text(frame: Any) -> str | None:
    try:
        payload = parse_json_payload(frame)
    except (EOFError, OSError, UnicodeDecodeError, ValueError, VolcEngineError):
        return None
    results = payload.get("results")
    if not isinstance(results, list):
        return None
    parts: list[str] = []
    for result in results:
        if not isinstance(result, dict):
            continue
        if result.get("is_interim") is True:
            continue
        text = result.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    transcript = "".join(parts).strip()
    return transcript or None


def _chat_response_text(frame: Any) -> str | None:
    try:
        payload = parse_json_payload(frame)
    except (EOFError, OSError, UnicodeDecodeError, ValueError, VolcEngineError):
        return None
    content = payload.get("content")
    if isinstance(content, str):
        content = content.strip()
        if content:
            return content
    return None


def _input_speech_started_event(item_id: str) -> OpenAIRealtimeEvents:
    return cast(
        OpenAIRealtimeEvents,
        {
            "type": "input_audio_buffer.speech_started",
            "event_id": f"event_{uuid.uuid4()}",
            "audio_start_ms": 0,
            "item_id": item_id,
        },
    )


def _input_transcript_event(
    transcript: str, item_id: str | None = None
) -> OpenAIRealtimeEvents:
    return cast(
        OpenAIRealtimeEvents,
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "event_id": f"event_{uuid.uuid4()}",
            "transcript": transcript,
            "item_id": item_id or f"item_{uuid.uuid4()}",
            "content_index": 0,
        },
    )


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    pending: list[tuple[dict[str, Any], dict[str, Any]]] = [(merged, override)]
    while pending:
        target, source = pending.pop()
        for key, value in source.items():
            existing = target.get(key)
            if isinstance(existing, dict) and isinstance(value, dict):
                child = dict(existing)
                target[key] = child
                pending.append((child, value))
            else:
                target[key] = value
    return merged


def _normalize_model_name(model_name: str | None) -> str:
    model = (model_name or "").lower().strip()
    if model.startswith("volcengine/"):
        model = model.split("/", 1)[1]
    return model

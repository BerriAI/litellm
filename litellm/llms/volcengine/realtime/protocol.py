import gzip
import json
from dataclasses import dataclass
from typing import Any

from litellm.llms.volcengine.common_utils import VolcEngineError

MSG_FULL_CLIENT = 0b0001
MSG_AUDIO_CLIENT = 0b0010
MSG_FULL_SERVER = 0b1001
MSG_AUDIO_SERVER = 0b1011
MSG_ERROR = 0b1111

FLAG_EVENT = 0b0100
FLAG_ERROR = 0b1111

SER_RAW = 0b0000
SER_JSON = 0b0001

COMP_NONE = 0b0000
COMP_GZIP = 0b0001

EV_START_CONNECTION = 1
EV_FINISH_CONNECTION = 2
EV_CONNECTION_STARTED = 50
EV_CONNECTION_FAILED = 51
EV_CONNECTION_FINISHED = 52

EV_START_SESSION = 100
EV_FINISH_SESSION = 102
EV_SESSION_STARTED = 150
EV_SESSION_FINISHED = 152
EV_SESSION_FAILED = 153

EV_TASK_REQUEST = 200
EV_UPDATE_CONFIG = 201
EV_CONFIG_UPDATED = 251
EV_SAY_HELLO = 300
EV_TTS_SENTENCE_START = 350
EV_TTS_SENTENCE_END = 351
EV_TTS_RESPONSE = 352
EV_TTS_ENDED = 359
EV_ASR_INFO = 450
EV_CLEAR_AUDIO = EV_ASR_INFO
EV_ASR_RESPONSE = 451
EV_ASR_ENDED = 459
EV_CHAT_RESPONSE = 550
EV_CHAT_ENDED = 559
EV_DIALOG_COMMON_ERROR = 599


@dataclass
class RealtimeFrame:
    message_type: int
    flags: int
    serialization: int
    compression: int
    payload: bytes
    event: int | None = None
    session_id: str | None = None
    error_code: int | None = None


def encode_json_event(
    *,
    event: int,
    payload: dict[str, Any],
    session_id: str | None = None,
    compression: int = COMP_NONE,
) -> bytes:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    if compression == COMP_GZIP:
        body = gzip.compress(body)
    return encode_event_frame(
        message_type=MSG_FULL_CLIENT,
        flags=FLAG_EVENT,
        serialization=SER_JSON,
        compression=compression,
        event=event,
        session_id=session_id,
        payload=body,
    )


def encode_audio_event(*, event: int, payload: bytes, session_id: str, compression: int = COMP_NONE) -> bytes:
    body = gzip.compress(payload) if compression == COMP_GZIP else payload
    return encode_event_frame(
        message_type=MSG_AUDIO_CLIENT,
        flags=FLAG_EVENT,
        serialization=SER_RAW,
        compression=compression,
        event=event,
        session_id=session_id,
        payload=body,
    )


def encode_event_frame(
    *,
    message_type: int,
    flags: int,
    serialization: int,
    compression: int,
    event: int,
    payload: bytes,
    session_id: str | None = None,
) -> bytes:
    parts = [
        bytes(
            [
                0x11,
                ((message_type & 0x0F) << 4) | (flags & 0x0F),
                ((serialization & 0x0F) << 4) | (compression & 0x0F),
                0x00,
            ]
        ),
        int(event).to_bytes(4, "big", signed=False),
    ]
    if session_id is not None:
        sid = session_id.encode("utf-8")
        parts.append(len(sid).to_bytes(4, "big", signed=False))
        parts.append(sid)
    parts.append(len(payload).to_bytes(4, "big", signed=False))
    parts.append(payload)
    return b"".join(parts)


def decode_realtime_frame(data: bytes) -> RealtimeFrame:
    if len(data) < 4:
        raise VolcEngineError(status_code=502, message="Volcengine realtime frame is too short.")
    b0, b1, b2 = data[0], data[1], data[2]
    if ((b0 >> 4) & 0x0F) != 1 or (b0 & 0x0F) != 1:
        raise VolcEngineError(status_code=502, message="Volcengine realtime frame has invalid header.")

    message_type = (b1 >> 4) & 0x0F
    flags = b1 & 0x0F
    serialization = (b2 >> 4) & 0x0F
    compression = b2 & 0x0F
    offset = 4

    def read_u32() -> int:
        nonlocal offset
        if offset + 4 > len(data):
            raise VolcEngineError(status_code=502, message="Volcengine realtime frame is truncated.")
        value = int.from_bytes(data[offset : offset + 4], "big", signed=False)
        offset += 4
        return value

    def read_bytes(size: int) -> bytes:
        nonlocal offset
        if offset + size > len(data):
            raise VolcEngineError(status_code=502, message="Volcengine realtime frame is truncated.")
        value = data[offset : offset + size]
        offset += size
        return value

    error_code = None
    event = None
    session_id = None
    if message_type == MSG_ERROR or flags == FLAG_ERROR:
        error_code = read_u32()
    if flags & FLAG_EVENT:
        event = read_u32()

    if event is not None and event >= EV_START_SESSION and offset + 4 <= len(data):
        sid_size = int.from_bytes(data[offset : offset + 4], "big", signed=False)
        if 0 < sid_size <= 256 and offset + 4 + sid_size + 4 <= len(data):
            offset += 4
            session_id = read_bytes(sid_size).decode("utf-8")

    payload = b""
    if offset + 4 <= len(data):
        payload_size = read_u32()
        if payload_size > 0:
            payload = read_bytes(payload_size)

    return RealtimeFrame(
        message_type=message_type,
        flags=flags,
        serialization=serialization,
        compression=compression,
        payload=payload,
        event=event,
        session_id=session_id,
        error_code=error_code,
    )


def parse_payload(frame: RealtimeFrame) -> bytes:
    payload = frame.payload
    if payload and frame.compression == COMP_GZIP:
        payload = gzip.decompress(payload)
    return payload


def parse_json_payload(frame: RealtimeFrame) -> dict[str, Any]:
    payload = parse_payload(frame)
    if not payload:
        return {}
    return json.loads(payload.decode("utf-8"))

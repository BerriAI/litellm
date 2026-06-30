import gzip
import json
from dataclasses import dataclass
from typing import Any

from litellm.llms.volcengine.common_utils import VolcEngineError

MSG_FULL_CLIENT = 0b0001
MSG_FULL_SERVER = 0b1001
MSG_AUDIO_SERVER = 0b1011
MSG_ERROR = 0b1111

FLAG_EVENT = 0b0100
FLAG_ERROR = 0b1111

SER_JSON = 0b0001
COMP_NONE = 0b0000
COMP_GZIP = 0b0001

EV_START_CONNECTION = 1
EV_FINISH_CONNECTION = 2
EV_START_SESSION = 100
EV_FINISH_SESSION = 102
EV_TASK_REQUEST = 200
EV_CONNECTION_STARTED = 50
EV_CONNECTION_FAILED = 51
EV_CONNECTION_FINISHED = 52
EV_SESSION_STARTED = 150
EV_SESSION_FINISHED = 152
EV_SESSION_FAILED = 153
EV_TTS_RESPONSE = 352


@dataclass
class TtsFrame:
    message_type: int
    flags: int
    serialization: int
    compression: int
    payload: bytes
    event: int | None = None
    session_id: str | None = None
    error_code: int | None = None


def encode_json_event(*, event: int, payload: dict[str, Any], session_id: str | None = None) -> bytes:
    return encode_event_frame(
        message_type=MSG_FULL_CLIENT,
        flags=FLAG_EVENT,
        serialization=SER_JSON,
        compression=COMP_NONE,
        event=event,
        session_id=session_id,
        payload=json.dumps(payload).encode("utf-8"),
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


def decode_event_frame(data: bytes) -> TtsFrame:
    if len(data) < 4:
        raise VolcEngineError(status_code=502, message="Volcengine TTS frame is too short.")
    b0, b1, b2 = data[0], data[1], data[2]
    if ((b0 >> 4) & 0x0F) != 1 or (b0 & 0x0F) != 1:
        raise VolcEngineError(status_code=502, message="Volcengine TTS frame has invalid header.")

    message_type = (b1 >> 4) & 0x0F
    flags = b1 & 0x0F
    serialization = (b2 >> 4) & 0x0F
    compression = b2 & 0x0F
    offset = 4

    def read_u32() -> int:
        nonlocal offset
        if offset + 4 > len(data):
            raise VolcEngineError(status_code=502, message="Volcengine TTS frame is truncated.")
        value = int.from_bytes(data[offset : offset + 4], "big", signed=False)
        offset += 4
        return value

    def read_bytes(size: int) -> bytes:
        nonlocal offset
        if offset + size > len(data):
            raise VolcEngineError(status_code=502, message="Volcengine TTS frame is truncated.")
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

    if event is not None and event >= 100 and offset + 4 <= len(data):
        sid_size = int.from_bytes(data[offset : offset + 4], "big", signed=False)
        if 0 < sid_size <= 256 and offset + 4 + sid_size + 4 <= len(data):
            offset += 4
            session_id = read_bytes(sid_size).decode("utf-8")

    payload = b""
    if offset + 4 <= len(data):
        payload_size = read_u32()
        if payload_size > 0:
            payload = read_bytes(payload_size)

    return TtsFrame(
        message_type=message_type,
        flags=flags,
        serialization=serialization,
        compression=compression,
        payload=payload,
        event=event,
        session_id=session_id,
        error_code=error_code,
    )


def parse_json_payload(frame: TtsFrame) -> dict[str, Any]:
    payload = frame.payload
    if payload and frame.compression == COMP_GZIP:
        payload = gzip.decompress(payload)
    if not payload:
        return {}
    return json.loads(payload.decode("utf-8"))

import gzip
import json
from dataclasses import dataclass
from typing import Any

from litellm.llms.volcengine.common_utils import VolcEngineError

HEADER_BYTE_0 = 0x11

MSG_FULL_CLIENT = 0b0001
MSG_AUDIO_CLIENT = 0b0010
MSG_FULL_SERVER = 0b1001
MSG_ERROR = 0b1111

FLAG_POS_SEQ = 0b0001
FLAG_NEG_SEQ = 0b0011

SER_RAW = 0b0000
SER_JSON = 0b0001
COMP_NONE = 0b0000
COMP_GZIP = 0b0001


@dataclass
class SaucFrame:
    message_type: int
    flags: int
    serialization: int
    compression: int
    payload: bytes
    sequence: int | None = None
    error_code: int | None = None


def encode_sauc_frame(
    *,
    message_type: int,
    flags: int,
    serialization: int,
    compression: int,
    payload: bytes,
    sequence: int | None = None,
) -> bytes:
    parts = [
        bytes(
            [
                HEADER_BYTE_0,
                ((message_type & 0x0F) << 4) | (flags & 0x0F),
                ((serialization & 0x0F) << 4) | (compression & 0x0F),
                0x00,
            ]
        )
    ]
    if flags & 0b0001:
        if sequence is None:
            raise VolcEngineError(
                status_code=500,
                message="SAUC frame sequence is required when sequence flag is set.",
            )
        parts.append(int(sequence).to_bytes(4, "big", signed=True))
    parts.append(len(payload).to_bytes(4, "big", signed=False))
    parts.append(payload)
    return b"".join(parts)


def encode_sauc_json_config(payload: dict[str, Any]) -> bytes:
    return encode_sauc_frame(
        message_type=MSG_FULL_CLIENT,
        flags=FLAG_POS_SEQ,
        serialization=SER_JSON,
        compression=COMP_GZIP,
        sequence=1,
        payload=gzip.compress(json.dumps(payload).encode("utf-8")),
    )


def encode_sauc_audio_chunk(pcm: bytes, sequence: int, last: bool) -> bytes:
    return encode_sauc_frame(
        message_type=MSG_AUDIO_CLIENT,
        flags=FLAG_NEG_SEQ if last else FLAG_POS_SEQ,
        serialization=SER_RAW,
        compression=COMP_GZIP,
        sequence=sequence,
        payload=gzip.compress(pcm),
    )


def decode_sauc_frame(data: bytes) -> SaucFrame:
    if len(data) < 4:
        raise VolcEngineError(status_code=502, message="SAUC frame is too short.")
    b0, b1, b2 = data[0], data[1], data[2]
    if ((b0 >> 4) & 0x0F) != 1 or (b0 & 0x0F) != 1:
        raise VolcEngineError(status_code=502, message="SAUC frame has invalid header.")

    message_type = (b1 >> 4) & 0x0F
    flags = b1 & 0x0F
    serialization = (b2 >> 4) & 0x0F
    compression = b2 & 0x0F
    offset = 4

    def read_u32() -> int:
        nonlocal offset
        if offset + 4 > len(data):
            raise VolcEngineError(status_code=502, message="SAUC frame is truncated.")
        value = int.from_bytes(data[offset : offset + 4], "big", signed=False)
        offset += 4
        return value

    def read_i32() -> int:
        nonlocal offset
        if offset + 4 > len(data):
            raise VolcEngineError(status_code=502, message="SAUC frame is truncated.")
        value = int.from_bytes(data[offset : offset + 4], "big", signed=True)
        offset += 4
        return value

    error_code = None
    sequence = None
    if message_type == MSG_ERROR:
        error_code = read_u32()
    elif flags & 0b0001:
        sequence = read_i32()

    payload = b""
    if offset + 4 <= len(data):
        payload_size = read_u32()
        if payload_size > 0:
            payload = data[offset : offset + payload_size]

    return SaucFrame(
        message_type=message_type,
        flags=flags,
        serialization=serialization,
        compression=compression,
        payload=payload,
        sequence=sequence,
        error_code=error_code,
    )


def parse_sauc_json_payload(frame: SaucFrame) -> dict[str, Any] | None:
    payload = frame.payload
    if payload and frame.compression == COMP_GZIP:
        payload = gzip.decompress(payload)
    if not payload:
        return None
    return json.loads(payload.decode("utf-8"))

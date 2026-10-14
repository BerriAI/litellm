"""AWS event-stream framing (application/vnd.amazon.eventstream), the wire format Bedrock streams
use. Each message is a prelude (total length, headers length), a prelude CRC, string headers and
the JSON payload, closed by a CRC of the whole message. The gateway decodes it with botocore's
EventStreamBuffer, so a framing mistake here shows up as a parse error on its side."""

from __future__ import annotations

import json
import struct
import zlib
from typing import Final

_STRING_HEADER: Final = 7


def _header(name: str, value: str) -> bytes:
    name_bytes: Final = name.encode()
    value_bytes: Final = value.encode()
    return (
        struct.pack(">B", len(name_bytes))
        + name_bytes
        + struct.pack(">BH", _STRING_HEADER, len(value_bytes))
        + value_bytes
    )


def encode_event(event_type: str, payload: dict[str, object]) -> bytes:
    headers: Final = (
        _header(":event-type", event_type)
        + _header(":content-type", "application/json")
        + _header(":message-type", "event")
    )
    body: Final = json.dumps(payload).encode()
    total_length: Final = 12 + len(headers) + len(body) + 4
    prelude: Final = struct.pack(">II", total_length, len(headers))
    prelude_crc: Final = struct.pack(">I", zlib.crc32(prelude) & 0xFFFFFFFF)
    message: Final = prelude + prelude_crc + headers + body
    return message + struct.pack(">I", zlib.crc32(message) & 0xFFFFFFFF)

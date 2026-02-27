"""
Image utilities for health checks.
"""

import io
import struct
import zlib


def _create_png(width: int = 256, height: int = 256) -> bytes:
    """Create a minimal white PNG image without external dependencies."""

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        return (
            struct.pack(">I", len(data))
            + c
            + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        )

    # IHDR: 8-bit RGB
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # IDAT: each row is filter byte(0) + RGB(255,255,255) * width
    raw_row = b"\x00" + b"\xff\xff\xff" * width
    raw = raw_row * height
    idat_data = zlib.compress(raw)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr_data)
    png += _chunk(b"IDAT", idat_data)
    png += _chunk(b"IEND", b"")
    return png


_TEST_PNG_BYTES: bytes = _create_png(256, 256)


def get_image_for_health_check() -> io.BytesIO:
    """Return a 256x256 PNG image as BytesIO for health check calls."""
    buf = io.BytesIO(_TEST_PNG_BYTES)
    buf.name = "health_check.png"
    return buf

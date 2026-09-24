import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import IO, Final, cast

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.token_counter import get_image_type
from litellm.types.llms.openai import FileTypes

_JPEG_SOF_MARKERS: Final = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_MAX_JPEG_SEGMENTS: Final = 1024
_MAX_JPEG_HEADER_OFFSET: Final = 16 * 1024 * 1024
_JPEG_FIRST_SEGMENT_OFFSET: Final = 2
_JPEG_SOF_PAYLOAD_SIZE: Final = 5
_JPEG_FILL_CHUNK: Final = 64 * 1024
_HEADER_READ_SIZE: Final = 32
_MIN_PNG_HEADER_SIZE: Final = 24
_MIN_GIF_HEADER_SIZE: Final = 10
_MIN_BMP_HEADER_SIZE: Final = 26
_MIN_WEBP_HEADER_SIZE: Final = 30


@dataclass(frozen=True, slots=True)
class ImageDimensions:
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


def _content_stream(image: FileTypes) -> IO[bytes] | None:
    content: Final = image[1] if isinstance(image, tuple) else image
    if isinstance(content, bytes):
        return BytesIO(content)
    if isinstance(content, (str, os.PathLike)):
        return None
    return content if content.seekable() else None


def _png_dimensions(head: bytes) -> ImageDimensions | None:
    if len(head) < _MIN_PNG_HEADER_SIZE:
        return None
    width, height = cast(tuple[int, int], struct.unpack(">II", head[16:24]))
    return ImageDimensions(width=width, height=height)


def _gif_dimensions(head: bytes) -> ImageDimensions | None:
    if len(head) < _MIN_GIF_HEADER_SIZE:
        return None
    width: Final = int.from_bytes(head[6:8], "little")
    height: Final = int.from_bytes(head[8:10], "little")
    return ImageDimensions(width=width, height=height)


def _bmp_dimensions(head: bytes) -> ImageDimensions | None:
    if len(head) < _MIN_BMP_HEADER_SIZE:
        return None
    if int.from_bytes(head[14:18], "little") == 12:
        core_width: Final = int.from_bytes(head[18:20], "little")
        core_height: Final = int.from_bytes(head[20:22], "little")
        return ImageDimensions(width=core_width, height=core_height)
    width: Final = int.from_bytes(head[18:22], "little", signed=True)
    height: Final = int.from_bytes(head[22:26], "little", signed=True)
    return ImageDimensions(width=width, height=abs(height))


def _webp_dimensions(head: bytes) -> ImageDimensions | None:
    if len(head) < _MIN_WEBP_HEADER_SIZE:
        return None
    match head[12:16]:  # pyright: ignore[reportMatchNotExhaustive]  # unknown fourccs fall through to the None below
        case b"VP8X":
            return ImageDimensions(
                width=int.from_bytes(head[24:27], "little") + 1,
                height=int.from_bytes(head[27:30], "little") + 1,
            )
        case b"VP8 ":
            width, height = cast(tuple[int, int], struct.unpack("<HH", head[26:30]))
            return ImageDimensions(width=width & 0x3FFF, height=height & 0x3FFF)
        case b"VP8L":
            bits: Final = int.from_bytes(head[21:25], "little")
            return ImageDimensions(width=(bits & 0x3FFF) + 1, height=((bits >> 14) & 0x3FFF) + 1)
    return None


def _jpeg_sof_dimensions(stream: IO[bytes], start: int, offset: int, segments_left: int) -> ImageDimensions | None:
    segment_offset = offset  # rebind-ok: advanced one JPEG segment per hop
    for _ in range(segments_left):
        if segment_offset > _MAX_JPEG_HEADER_OFFSET:
            return None
        stream.seek(start + segment_offset)
        marker = stream.read(4)
        if len(marker) < 4 or marker[0] != 0xFF:
            return None
        if marker[1] == 0xFF:
            stream.seek(start + segment_offset)
            fill = stream.read(_JPEG_FILL_CHUNK)
            run = len(fill) - len(fill.lstrip(b"\xff"))
            segment_offset += run - 1
            continue
        if marker[1] in _JPEG_SOF_MARKERS:
            sof = stream.read(_JPEG_SOF_PAYLOAD_SIZE)
            if len(sof) < _JPEG_SOF_PAYLOAD_SIZE:
                return None
            _precision, height, width = cast(tuple[int, int, int], struct.unpack(">BHH", sof))
            return ImageDimensions(width=width, height=height)
        segment_length = int.from_bytes(marker[2:4], "big")
        if segment_length < 2:
            return None
        segment_offset += 2 + segment_length
    return None


def _header_dimensions(stream: IO[bytes], position: int) -> ImageDimensions | None:
    stream.seek(position)
    head: Final = stream.read(_HEADER_READ_SIZE)
    match get_image_type(head):  # pyright: ignore[reportMatchNotExhaustive]  # unmatched types fall through to the None below
        case "png":
            return _png_dimensions(head)
        case "webp":
            return _webp_dimensions(head)
        case "jpeg":
            return _jpeg_sof_dimensions(stream, position, _JPEG_FIRST_SEGMENT_OFFSET, _MAX_JPEG_SEGMENTS)
        case "gif":
            return _gif_dimensions(head)
    if head[:2] == b"BM":
        return _bmp_dimensions(head)
    return None


def _measure_stream(stream: IO[bytes]) -> ImageDimensions | None:
    position: Final = stream.tell()
    try:
        dimensions: Final = _header_dimensions(stream, position)
    finally:
        stream.seek(position)
    if dimensions is None or dimensions.width <= 0 or dimensions.height <= 0:
        return None
    return dimensions


def read_image_dimensions(image: FileTypes) -> ImageDimensions | None:
    try:
        stream: Final = _content_stream(image)
        if stream is None:
            return None
        return _measure_stream(stream)
    except (OSError, ValueError, AttributeError, struct.error):
        return None


def total_reference_pixels(images: Sequence[FileTypes]) -> int | None:
    measured: Final = tuple(read_image_dimensions(image) for image in images)
    dimensions: Final = tuple(size for size in measured if size is not None)
    if len(dimensions) != len(measured):
        verbose_logger.debug(
            "Reference image %d could not be measured (path, non-seekable stream, or no readable PNG, JPEG, WebP, "
            "GIF or BMP header); billing generated pixels only",
            measured.index(None),
        )
        return None
    return sum(size.pixels for size in dimensions)

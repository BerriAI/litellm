import os
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import IO, Final, cast

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.token_counter import get_image_type
from litellm.types.llms.openai import FileTypes

_HEADER_READ_SIZE: Final = 32

_JPEG_SOF_MARKERS: Final = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_JPEG_MAX_SEGMENTS: Final = 1024
_JPEG_MAX_HEADER_OFFSET: Final = 16 * 1024 * 1024
_JPEG_FIRST_SEGMENT_OFFSET: Final = 2
_JPEG_SOF_PAYLOAD_SIZE: Final = 5
_JPEG_FILL_CHUNK: Final = 64 * 1024


@dataclass(frozen=True, slots=True)
class ImageDimensions:
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


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


def read_image_dimensions(image: FileTypes) -> ImageDimensions | None:
    try:
        content = image[1] if isinstance(image, tuple) else image
        if isinstance(content, (str, os.PathLike)):
            return None
        stream = BytesIO(content) if isinstance(content, bytes) else content
        if not stream.seekable():
            return None
        position = stream.tell()
        try:
            dimensions = _header_dimensions(stream, position)
        finally:
            stream.seek(position)
        if dimensions is None or dimensions.width <= 0 or dimensions.height <= 0:
            return None
        return dimensions
    except (OSError, ValueError, AttributeError, struct.error):
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
            return _jpeg_sof_dimensions(stream, position)
        case "gif":
            return _gif_dimensions(head)
    if head[:2] == b"BM":
        return _bmp_dimensions(head)
    return None


def _png_dimensions(head: bytes) -> ImageDimensions | None:
    if len(head) < 24:
        return None
    width, height = cast(tuple[int, int], struct.unpack(">II", head[16:24]))
    return ImageDimensions(width=width, height=height)


def _gif_dimensions(head: bytes) -> ImageDimensions | None:
    if len(head) < 10:
        return None
    width: Final = int.from_bytes(head[6:8], "little")
    height: Final = int.from_bytes(head[8:10], "little")
    return ImageDimensions(width=width, height=height)


def _bmp_dimensions(head: bytes) -> ImageDimensions | None:
    if len(head) < 26:
        return None
    if int.from_bytes(head[14:18], "little") == 12:
        core_width: Final = int.from_bytes(head[18:20], "little")
        core_height: Final = int.from_bytes(head[20:22], "little")
        return ImageDimensions(width=core_width, height=core_height)
    width: Final = int.from_bytes(head[18:22], "little", signed=True)
    height: Final = int.from_bytes(head[22:26], "little", signed=True)
    return ImageDimensions(width=width, height=abs(height))


def _webp_dimensions(head: bytes) -> ImageDimensions | None:
    if len(head) < 30:
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


def _jpeg_fill_run(stream: IO[bytes], start: int, offset: int) -> int:
    stream.seek(start + offset)
    fill: Final = stream.read(_JPEG_FILL_CHUNK)
    run: Final = len(fill) - len(fill.lstrip(b"\xff"))
    return run - 1


def _jpeg_sof_dimensions(stream: IO[bytes], start: int) -> ImageDimensions | None:
    offset = _JPEG_FIRST_SEGMENT_OFFSET  # rebind-ok: advanced one JPEG segment per hop
    for _ in range(_JPEG_MAX_SEGMENTS):
        if offset > _JPEG_MAX_HEADER_OFFSET:
            return None
        stream.seek(start + offset)
        marker = stream.read(4)
        if len(marker) < 4 or marker[0] != 0xFF:
            return None
        if marker[1] == 0xFF:
            offset += _jpeg_fill_run(stream, start, offset)
            continue
        if marker[1] in _JPEG_SOF_MARKERS:
            sof = stream.read(_JPEG_SOF_PAYLOAD_SIZE)
            if len(sof) < _JPEG_SOF_PAYLOAD_SIZE:
                return None
            height, width = cast(tuple[int, int], struct.unpack(">HH", sof[1:5]))
            return ImageDimensions(width=width, height=height)
        segment_length = int.from_bytes(marker[2:4], "big")
        if segment_length < 2:
            return None
        offset += 2 + segment_length
    return None

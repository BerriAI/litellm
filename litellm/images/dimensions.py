import base64
import os
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import IO, Final, cast

from httpx._types import RequestFiles  # pyright: ignore[reportPrivateImportUsage]  # same source the base transform classes use

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.token_counter import get_image_type
from litellm.types.llms.openai import FileTypes
from litellm.types.utils import ImageResponse

_HEADER_READ_SIZE: Final = 32
_IMAGE_SIGNATURE_BYTES: Final = 12
_DATA_URI_PREFIX: Final = "data:"
_EMBEDDED_IMAGE_MAX_DEPTH: Final = 4
_IMAGE_SIGNATURES: Final = frozenset({"png", "jpeg", "webp", "gif"})

_JPEG_SOF_MARKERS: Final = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_JPEG_MAX_SEGMENTS: Final = 1024
_JPEG_MAX_HEADER_OFFSET: Final = 16 * 1024 * 1024
_JPEG_FIRST_SEGMENT_OFFSET: Final = 2
_JPEG_SOF_PAYLOAD_SIZE: Final = 5
_JPEG_FILL_CHUNK: Final = 64 * 1024

_PNG_IHDR: Final = b"IHDR"
_WEBP_VP8_START_CODE: Final = b"\x9d\x01\x2a"
_WEBP_VP8L_SIGNATURE: Final = 0x2F
_BMP_CORE_DIB_SIZE: Final = 12
_BMP_KNOWN_DIB_SIZES: Final = frozenset({12, 40, 52, 56, 64, 108, 124})


@dataclass(frozen=True, slots=True)
class ImageDimensions:
    width: int
    height: int

    @property
    def pixels(self) -> int:
        return self.width * self.height


def total_reference_pixels(images: Sequence[FileTypes]) -> int | None:
    """All-or-nothing sum of measured reference pixels; never raises."""
    try:
        measured: Final = tuple(read_image_dimensions(image) for image in images)
    except Exception:  # noqa: BLE001  # a billing helper must fall back, never fail the request
        return None
    dimensions: Final = tuple(size for size in measured if size is not None)
    if len(dimensions) != len(measured):
        verbose_logger.debug(
            "Reference image %d could not be measured (path, non-seekable stream, or no readable PNG, JPEG, WebP, "
            "GIF or BMP header); billing generated pixels only",
            measured.index(None),
        )
        return None
    return sum(size.pixels for size in dimensions)


def uploaded_reference_pixels(
    files: RequestFiles | None,
    json_body: Mapping[str, object] | None,
) -> int | None:
    """Pixels across the image-bearing parts a request actually sends.

    Multipart requests carry image content under ``files``; JSON requests embed base64 inside
    ``json_body`` (e.g. FLUX ``input_image`` fields). Measuring the outgoing payload instead of the
    caller's arguments keeps billing aligned with what the provider meters when a transform filters
    images out or adds parts of its own (masks, single-image providers). All-or-nothing, ``0`` when
    nothing image-bearing is sent.
    """
    parts: Final = _file_parts(files) + tuple(_embedded_image_values(json_body or {}))
    if not parts:
        return 0
    return total_reference_pixels(parts)


def _file_parts(files: RequestFiles | None) -> tuple[FileTypes, ...]:
    if files is None:
        return ()
    if isinstance(files, Mapping):
        return tuple(files.values())
    return tuple(files)


def _file_content(part: FileTypes) -> object:
    """The payload bytes of an httpx file part: (field, (name, content, type, headers?)) unwraps twice."""
    unwrapped: Final = part[1] if isinstance(part, tuple) else part
    return unwrapped[1] if isinstance(unwrapped, tuple) else unwrapped


def read_image_dimensions(image: FileTypes) -> ImageDimensions | None:
    """Measure one image file's pixel dimensions; ``None`` when unmeasurable, never raises.

    Seekable content is measured from offset 0 (the bytes an upload sends) with the stream position
    restored afterward. Strings are read as embedded base64 or data-URI image content; a filesystem
    path does not decode to an image signature and still returns ``None``.
    """
    try:
        content: Final = cast(
            "IO[bytes] | bytes | str | os.PathLike[str] | None",
            _file_content(image),
        )
        if content is None:
            return None
        if isinstance(content, (str, os.PathLike)):
            embedded: Final = _decoded_image_bytes(os.fspath(content))
            if embedded is None:
                return None
            return _dimensions_from_stream(BytesIO(embedded))
        stream = BytesIO(content) if isinstance(content, (bytes, bytearray, memoryview)) else content
        return _dimensions_from_stream(stream)
    except Exception:  # noqa: BLE001  # an odd stream or malformed file must fall back, never raise
        return None


def with_reference_pixels(response: ImageResponse, reference_pixels: int | None) -> ImageResponse:
    if reference_pixels is not None:
        response.set_reference_pixels(reference_pixels)
    return response


def _dimensions_from_stream(stream: IO[bytes]) -> ImageDimensions | None:
    if not stream.seekable():
        return None
    position: Final = stream.tell()
    try:
        dimensions = _header_dimensions(stream, 0)
    finally:
        stream.seek(position)
    if dimensions is None or dimensions.width <= 0 or dimensions.height <= 0:
        return None
    return dimensions


def _decoded_image_bytes(text: str) -> bytes | None:
    """The image bytes a string carries, when it is base64 or data-URI encoded image content."""
    payload: Final = text.partition(",")[2] if text.startswith(_DATA_URI_PREFIX) else text
    if len(payload) < _IMAGE_SIGNATURE_BYTES:
        return None
    try:
        head: Final = base64.b64decode(payload[:64])
        if not _is_image_signature(head):
            return None
        return base64.b64decode(payload)
    except ValueError:  # binascii.Error subclasses ValueError; undecodable fields are not images
        return None


def _embedded_image_values(value: object, depth: int = 0) -> Iterator[bytes]:
    if depth > _EMBEDDED_IMAGE_MAX_DEPTH:
        return
    if isinstance(value, str):
        decoded: Final = _decoded_image_bytes(value)
        if decoded is not None:
            yield decoded
    elif isinstance(value, (bytes, bytearray, memoryview)):
        raw: Final = bytes(value)
        if _is_image_signature(raw[:_HEADER_READ_SIZE]):
            yield raw
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _embedded_image_values(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _embedded_image_values(item, depth + 1)


def _is_image_signature(head: bytes) -> bool:
    return len(head) >= _IMAGE_SIGNATURE_BYTES and (get_image_type(head) in _IMAGE_SIGNATURES or head[:2] == b"BM")


def _header_dimensions(stream: IO[bytes], position: int) -> ImageDimensions | None:
    stream.seek(position)
    head: Final = cast("bytes | None", stream.read(_HEADER_READ_SIZE))
    if head is None:
        return None
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
    if len(head) < 24 or head[12:16] != _PNG_IHDR:
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
    dib_size: Final = int.from_bytes(head[14:18], "little")
    if dib_size not in _BMP_KNOWN_DIB_SIZES:
        return None
    if dib_size == _BMP_CORE_DIB_SIZE:
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
            if head[23:26] != _WEBP_VP8_START_CODE:
                return None
            width, height = cast(tuple[int, int], struct.unpack("<HH", head[26:30]))
            return ImageDimensions(width=width & 0x3FFF, height=height & 0x3FFF)
        case b"VP8L":
            if head[20] != _WEBP_VP8L_SIGNATURE:
                return None
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

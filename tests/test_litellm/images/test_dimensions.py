import base64
import io
import struct
import zlib
from pathlib import Path
from typing import Any, Final, cast

import pytest
from PIL import Image

from litellm.images.dimensions import (
    _UNMEASURED_REFERENCE_PIXELS,
    ImageDimensions,
    read_image_dimensions,
    total_reference_pixels,
    uploaded_reference_pixels,
)

CAT_JPEG: Final = Path(__file__).parents[2] / "e2e" / "llm_translation" / "fixtures" / "cat.jpg"
JPEG_SOI: Final = b"\xff\xd8"
JPEG_APP1: Final = b"\xff\xe1"
JPEG_APP2: Final = b"\xff\xe2"
JPEG_SOF0: Final = b"\xff\xc0"
JPEG_MARKER_AND_LENGTH_SIZE: Final = 4


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + tag + payload + struct.pack(">I", zlib.crc32(tag + payload))


def _png(width: int, height: int) -> bytes:
    header: Final = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    rows: Final = b"".join(b"\x00" + bytes(width) for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )


def _pillow_bytes(width: int, height: int, mode: str, image_format: str, **save_options: object) -> bytes:
    buffer: Final = io.BytesIO()
    Image.new(mode, (width, height)).save(buffer, format=image_format, **save_options)
    return buffer.getvalue()


def _jpeg_app_segment(marker: bytes, payload_size: int) -> bytes:
    return marker + struct.pack(">H", payload_size + 2) + bytes(payload_size)


def _jpeg_with_leading_segments(jpeg: bytes, *segments: bytes) -> bytes:
    return JPEG_SOI + b"".join(segments) + jpeg[len(JPEG_SOI) :]


class _NonSeekableStream(io.BytesIO):
    def seekable(self) -> bool:
        return False


class _FailingReadStream(io.BytesIO):
    def read(self, size: int = -1) -> bytes:
        raise OSError("simulated disk read failure")


def _jpeg_sof_segment(marker: int, width: int, height: int) -> bytes:
    return bytes((0xFF, marker)) + struct.pack(">H", 17) + struct.pack(">BHH", 8, height, width)


def _jpeg_segment(marker: int, payload: bytes) -> bytes:
    return bytes((0xFF, marker)) + struct.pack(">H", len(payload) + 2) + payload


def _webp_chunk(fourcc: bytes, body: bytes) -> bytes:
    chunk: Final = fourcc + struct.pack("<I", len(body)) + body
    return b"RIFF" + struct.pack("<I", len(chunk) + 4) + b"WEBP" + chunk


def _fill_jpeg(fill_bytes: int, width: int = 320, height: int = 200) -> bytes:
    return JPEG_SOI + b"\xff" * fill_bytes + _jpeg_sof_segment(0xC0, width, height)


def _segments_jpeg(marker: int, count: int, payload_size: int = 4) -> bytes:
    return JPEG_SOI + _jpeg_segment(marker, bytes(payload_size)) * count + _jpeg_sof_segment(0xC0, 320, 200)


VP8_BODY: Final = bytes(3) + b"\x9d\x01\x2a" + struct.pack("<HH", 0xC000 | 500, 0xC000 | 250) + bytes(4)
VP8L_BODY: Final = b"\x2f" + (299 | (199 << 14)).to_bytes(4, "little") + bytes(5)
VP8X_BODY: Final = bytes(4) + (700 - 1).to_bytes(3, "little") + (350 - 1).to_bytes(3, "little")


def _gif(version: bytes, width: int, height: int) -> bytes:
    return version + struct.pack("<HH", width, height) + bytes(22)


def _bmp(width: int, height: int) -> bytes:
    return b"BM" + bytes(12) + struct.pack("<I", 40) + struct.pack("<ii", width, height) + bytes(6)


def _dims(width: int, height: int) -> ImageDimensions:
    return ImageDimensions(width=width, height=height)


@pytest.mark.parametrize(
    ("header", "expected"),
    (
        pytest.param(
            JPEG_SOI + b"\xff\xff" + _jpeg_sof_segment(0xC0, 320, 200),
            _dims(320, 200),
            id="jpeg-sof-after-0xff-padding",
        ),
        pytest.param(JPEG_SOI + bytes((0xFF, 0xE0, 0x00, 0x01)), None, id="jpeg-segment-length-below-2"),
        pytest.param(JPEG_SOI + bytes((0xFF, 0xC0, 0x00, 0x11, 0x08)), None, id="jpeg-truncated-sof-payload"),
        pytest.param(_segments_jpeg(0xE0, 63), _dims(320, 200), id="jpeg-sof-at-segment-64-boundary"),
        pytest.param(_segments_jpeg(0xE0, 65), _dims(320, 200), id="jpeg-65-segments-within-budget"),
        pytest.param(_segments_jpeg(0xE2, 300), _dims(320, 200), id="jpeg-300-app2-segments"),
        pytest.param(_segments_jpeg(0xE0, 1030), None, id="jpeg-more-than-1024-segments"),
        pytest.param(_fill_jpeg(1100), _dims(320, 200), id="jpeg-1100-fill-bytes-before-sof"),
        pytest.param(
            JPEG_SOI + b"\xff" + _jpeg_segment(0xE2, bytes(0xFEFE)) + _jpeg_sof_segment(0xC0, 640, 480),
            _dims(640, 480),
            id="jpeg-fill-before-large-segment",
        ),
        pytest.param(_fill_jpeg(200_000), _dims(320, 200), id="jpeg-200k-fill-bytes-before-sof"),
        pytest.param(_fill_jpeg(65535), _dims(320, 200), id="jpeg-65535-fill-bytes-before-sof"),
        pytest.param(_fill_jpeg(65536), _dims(320, 200), id="jpeg-65536-fill-bytes-before-sof"),
        pytest.param(_segments_jpeg(0xE0, 260, payload_size=65533), None, id="jpeg-sof-beyond-16mib"),
        pytest.param(JPEG_SOI + _jpeg_sof_segment(0xC2, 111, 55), _dims(111, 55), id="jpeg-sof2-progressive"),
        pytest.param(
            JPEG_SOI + _jpeg_segment(0xC4, bytes(8)) + _jpeg_sof_segment(0xC0, 400, 300),
            _dims(400, 300),
            id="jpeg-dht-skipped-before-sof0",
        ),
        pytest.param(_webp_chunk(b"VP8 ", VP8_BODY), _dims(500, 250), id="webp-vp8-lossy"),
        pytest.param(_webp_chunk(b"VP8L", VP8L_BODY), _dims(300, 200), id="webp-vp8l-lossless"),
        pytest.param(_webp_chunk(b"VP8X", VP8X_BODY), _dims(700, 350), id="webp-vp8x-extended"),
        pytest.param(_webp_chunk(b"VP8Z", bytes(10)), None, id="webp-unknown-fourcc"),
        pytest.param(_webp_chunk(b"VP8X", VP8X_BODY)[:29], None, id="webp-shorter-than-30-bytes"),
        pytest.param(
            _webp_chunk(b"VP8 ", bytes(3) + bytes(3) + struct.pack("<HH", 0xC000 | 500, 0xC000 | 250) + bytes(4)),
            None,
            id="webp-vp8-missing-start-code",
        ),
        pytest.param(
            _webp_chunk(b"VP8L", bytes(1) + (299 | (199 << 14)).to_bytes(4, "little") + bytes(5)),
            None,
            id="webp-vp8l-missing-signature",
        ),
        pytest.param(b"\x89PNG\r\n\x1a\n" + bytes(10), None, id="png-shorter-than-24-bytes"),
        pytest.param(
            b"\x89PNG\r\n\x1a\n"
            + _png_chunk(b"CgBI", struct.pack(">II", 4000, 3000) + bytes(8))
            + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1024, 1024, 8, 0, 0, 0, 0)),
            None,
            id="png-cgbi-first-chunk",
        ),
        pytest.param(_gif(b"GIF89a", 1024, 768), _dims(1024, 768), id="gif89a"),
        pytest.param(_gif(b"GIF87a", 320, 200), _dims(320, 200), id="gif87a"),
        pytest.param(b"GIF89a" + bytes(2), None, id="gif-shorter-than-10-bytes"),
        pytest.param(_bmp(1024, 768), _dims(1024, 768), id="bmp-positive-height"),
        pytest.param(_bmp(1024, -768), _dims(1024, 768), id="bmp-negative-height"),
        pytest.param(_bmp(-1024, 768), None, id="bmp-negative-width"),
        pytest.param(_bmp(0, 768), None, id="bmp-zero-width"),
        pytest.param(b"BM" + bytes(18), None, id="bmp-shorter-than-26-bytes"),
        pytest.param(
            b"BM" + bytes(12) + struct.pack("<I", 999) + struct.pack("<ii", 640, 480) + bytes(6),
            None,
            id="bmp-unknown-dib-size",
        ),
        pytest.param(
            b"BM" + bytes(12) + struct.pack("<I", 12) + struct.pack("<HH", 640, 480) + bytes(4),
            _dims(640, 480),
            id="bmp-coreheader",
        ),
        pytest.param(
            b"BM" + bytes(12) + struct.pack("<I", 108) + struct.pack("<ii", 640, 480) + bytes(6),
            _dims(640, 480),
            id="bmp-v4header",
        ),
        pytest.param(
            b"BM" + bytes(12) + struct.pack("<I", 124) + struct.pack("<ii", 640, 480) + bytes(6),
            _dims(640, 480),
            id="bmp-v5header",
        ),
        pytest.param(b"II*\x00" + bytes(28), None, id="unknown-format"),
    ),
)
def test_read_image_dimensions_parses_hand_built_headers(header: bytes, expected: ImageDimensions | None):
    assert read_image_dimensions(header) == expected


def test_read_image_dimensions_returns_none_and_restores_position_on_read_error():
    stream: Final = _FailingReadStream(bytes(64))
    stream.seek(5)

    assert read_image_dimensions(stream) is None
    assert stream.tell() == 5


@pytest.mark.parametrize("initial_position", (0, 3))
def test_read_image_dimensions_reads_png_jpeg_webp_headers_and_keeps_position(initial_position: int):
    pillow_jpeg: Final = _pillow_bytes(1024, 768, "RGB", "JPEG")
    large_app1_jpeg: Final = _jpeg_with_leading_segments(pillow_jpeg, _jpeg_app_segment(JPEG_APP1, 60_000))
    far_sof_jpeg: Final = _jpeg_with_leading_segments(
        pillow_jpeg, _jpeg_app_segment(JPEG_APP2, 40_000), _jpeg_app_segment(JPEG_APP2, 40_000)
    )
    assert far_sof_jpeg.index(JPEG_SOF0) > 70_000
    cases: Final = (
        (_png(1024, 1024), (1024, 1024)),
        (CAT_JPEG.read_bytes(), (512, 512)),
        (_pillow_bytes(640, 480, "RGB", "WEBP"), (640, 480)),
        (_pillow_bytes(640, 480, "RGB", "WEBP", lossless=True), (640, 480)),
        (_pillow_bytes(640, 480, "RGBA", "WEBP"), (640, 480)),
        (large_app1_jpeg, (1024, 768)),
        (far_sof_jpeg, (1024, 768)),
    )
    layouts: Final = tuple(image[12:16] for image, _ in cases[2:5])
    assert layouts == (b"VP8 ", b"VP8L", b"VP8X")

    for image, expected in cases:
        stream: Final = io.BytesIO(image)
        stream.seek(initial_position)
        # uploads rewind seekable parts to offset 0, so a mid-file cursor still measures the whole image
        assert read_image_dimensions(stream) == _dims(*expected), image[:16]
        assert stream.tell() == initial_position, image[:16]


def test_read_image_dimensions_returns_none_when_stream_content_is_not_an_image():
    stream: Final = io.BytesIO(bytes(5) + _png(1024, 1024))
    stream.seek(5)

    assert read_image_dimensions(stream) is None
    assert stream.tell() == 5


def test_read_image_dimensions_reads_base64_and_data_uri_strings():
    png_b64: Final = base64.b64encode(_png(512, 256)).decode()

    assert read_image_dimensions(png_b64) == _dims(512, 256)
    assert read_image_dimensions(f"data:image/png;base64,{png_b64}") == _dims(512, 256)
    assert read_image_dimensions(("ref.png", png_b64)) == _dims(512, 256)


@pytest.mark.parametrize(
    "text",
    (
        "/tmp/nonexistent.png",
        "",
        "the quick brown fox",
        base64.b64encode(b"not an image body").decode(),
        "data:text/plain;base64," + base64.b64encode(b"still not an image").decode(),
    ),
)
def test_read_image_dimensions_returns_none_for_non_image_strings(text: str):
    assert read_image_dimensions(text) is None


def test_read_image_dimensions_returns_none_on_truncated_jpeg_and_keeps_position():
    jpeg: Final = _pillow_bytes(1024, 768, "RGB", "JPEG")
    sof_offset: Final = jpeg.index(JPEG_SOF0)
    prefixes: Final = (32, 160, sof_offset, sof_offset + JPEG_MARKER_AND_LENGTH_SIZE)

    for prefix in prefixes:
        stream: Final = io.BytesIO(jpeg[:prefix])
        assert read_image_dimensions(stream) is None, prefix
        assert stream.tell() == 0, prefix


def test_read_image_dimensions_returns_none_for_unreadable_bytes_paths_tuples_and_non_seekable_streams(
    tmp_path: Path,
):
    png: Final = _png(64, 64)
    path: Final = tmp_path / "ref.png"
    path.write_bytes(png)
    non_seekable: Final = _NonSeekableStream(png)

    assert read_image_dimensions(io.BytesIO(b"image")) is None
    assert read_image_dimensions(b"II*\x00" + bytes(26)) is None
    assert read_image_dimensions(path) is None
    assert read_image_dimensions(("ref.png", path, "image/png", {})) is None
    assert read_image_dimensions(non_seekable) is None
    assert non_seekable.tell() == 0
    assert read_image_dimensions(("ref.png", png)) == _dims(64, 64)
    assert read_image_dimensions(("ref.png", io.BytesIO(png), "image/png")) == _dims(64, 64)


@pytest.mark.parametrize("sof_marker", (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF))
def test_read_image_dimensions_reads_every_jpeg_sof_marker(sof_marker: int):
    assert read_image_dimensions(JPEG_SOI + _jpeg_sof_segment(sof_marker, 320, 200)) == _dims(320, 200)


@pytest.mark.parametrize("skipped_marker", (0xC8, 0xCC))
def test_read_image_dimensions_skips_non_sof_c_range_markers(skipped_marker: int):
    jpeg: Final = JPEG_SOI + _jpeg_segment(skipped_marker, bytes(8)) + _jpeg_sof_segment(0xC0, 400, 300)

    assert read_image_dimensions(jpeg) == _dims(400, 300)


class _ReadOnlyObject:
    def read(self, size: int = -1) -> bytes:
        return b""


class _NonBlockingStream(io.BytesIO):
    def read(self, size: int = -1) -> bytes | None:
        return None


def test_read_image_dimensions_returns_none_when_read_returns_none():
    assert read_image_dimensions(cast(Any, _NonBlockingStream(_png(64, 64)))) is None


def test_read_image_dimensions_returns_none_for_short_and_none_tuples():
    assert read_image_dimensions(("ref.png",)) is None
    assert read_image_dimensions(("ref.png", None)) is None


def test_total_reference_pixels_bills_unmeasurable_input_as_one_megapixel():
    assert total_reference_pixels([("ref.png",)]) == _UNMEASURED_REFERENCE_PIXELS
    assert total_reference_pixels(cast(Any, 123)) == _UNMEASURED_REFERENCE_PIXELS


def test_total_reference_pixels_bills_closed_and_read_only_streams_as_one_megapixel():
    closed: Final = io.BytesIO(_png(64, 64))
    closed.close()

    assert total_reference_pixels([closed]) == _UNMEASURED_REFERENCE_PIXELS
    assert total_reference_pixels([cast(Any, _ReadOnlyObject())]) == _UNMEASURED_REFERENCE_PIXELS


def test_total_reference_pixels_bills_invalid_dimensions_as_one_megapixel():
    negative_width_bmp: Final = _bmp(-1024, 768)

    assert read_image_dimensions(negative_width_bmp) is None
    assert total_reference_pixels([negative_width_bmp, _png(1024, 1024)]) == _UNMEASURED_REFERENCE_PIXELS + 1024 * 1024


def test_total_reference_pixels_bills_each_unmeasurable_reference_as_one_megapixel():
    assert total_reference_pixels([_png(64, 64), CAT_JPEG.read_bytes()]) == 64 * 64 + 512 * 512
    assert total_reference_pixels([_png(64, 64), _gif(b"GIF89a", 100, 50)]) == 64 * 64 + 100 * 50
    assert total_reference_pixels([_png(64, 64), io.BytesIO(b"image"), CAT_JPEG.read_bytes()]) == (
        64 * 64 + _UNMEASURED_REFERENCE_PIXELS + 512 * 512
    )
    assert total_reference_pixels([]) == 0


def test_uploaded_reference_pixels_measures_the_uploaded_set_not_the_requested_set():
    png_b64: Final = base64.b64encode(_png(64, 64)).decode()
    jpeg_b64: Final = base64.b64encode(CAT_JPEG.read_bytes()).decode()

    # a transform that keeps only the first image (MAI) bills only what it sends
    assert uploaded_reference_pixels({"image[]": ("ref.png", _png(64, 64))}, {"prompt": "hi"}) == 64 * 64
    # FLUX.2 embeds each reference as a base64 field in the JSON body
    assert (
        uploaded_reference_pixels(
            [],
            {"model": "FLUX.2-flex", "prompt": "blend", "input_image": png_b64, "input_image_2": jpeg_b64, "n": 2},
        )
        == 64 * 64 + 512 * 512
    )
    # an extra file part a transform adds itself (a mask) is uploaded and metered too
    assert (
        uploaded_reference_pixels(
            {"image": ("edit.png", _png(64, 64)), "mask": ("mask.png", _png(10, 10))}, {"prompt": "cut"}
        )
        == 64 * 64 + 10 * 10
    )


def test_uploaded_reference_pixels_skips_non_image_fields_and_nested_bodies():
    png_b64: Final = base64.b64encode(_png(64, 64)).decode()
    jpeg_b64: Final = base64.b64encode(CAT_JPEG.read_bytes()).decode()

    assert (
        uploaded_reference_pixels(
            {"image": ("edit.png", _png(64, 64))},
            {"model": "m", "prompt": "hi", "size": "1024x1024", "extra": {"nested": jpeg_b64}, "refs": [png_b64]},
        )
        == 64 * 64 + 512 * 512 + 64 * 64
    )
    assert uploaded_reference_pixels(None, {"prompt": "hi", "model": "m"}) == 0
    assert uploaded_reference_pixels(None, None) == 0


def test_uploaded_reference_pixels_bills_each_unmeasurable_part_as_one_megapixel():
    png_b64: Final = base64.b64encode(_png(64, 64)).decode()
    truncated_png_b64: Final = base64.b64encode(_png(64, 64)[:16]).decode()

    assert uploaded_reference_pixels({"image": io.BytesIO(b"not an image")}, None) == _UNMEASURED_REFERENCE_PIXELS
    assert uploaded_reference_pixels(None, {"input_image": png_b64, "input_image_2": truncated_png_b64}) == (
        64 * 64 + _UNMEASURED_REFERENCE_PIXELS
    )

from __future__ import annotations

import base64
from io import BytesIO
from typing import Final, cast

from PIL import Image

from .media import dummy_image_url, structured_image_bytes, structured_image_data_uri


def test_dummy_image_url_encodes_text_and_dimensions() -> None:
    assert dummy_image_url("invoice 123", 24, width=320, height=80) == (
        "https://dummyjson.com/image/320x80/ffffff/000000?text=invoice%20123&fontSize=24"
    )


def test_structured_image_is_local_content_bearing_png() -> None:
    png: Final = structured_image_bytes()
    encoded: Final = structured_image_data_uri().partition(",")[2]
    image: Final = Image.open(BytesIO(png))
    colors: Final = cast(list[tuple[int, tuple[int, int, int]]], image.getcolors(maxcolors=2))

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert base64.b64decode(encoded, validate=True) == png
    assert image.size == (320, 80)
    assert {color for _, color in colors} == {(0, 0, 0), (255, 255, 255)}

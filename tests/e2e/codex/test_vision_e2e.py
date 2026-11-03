from __future__ import annotations

import random
import re
import struct
import zlib
from typing import Final

import pytest
from codex.harness import CodexHarness
from codex.setups import SETUPS, ModelSetup
from e2e_config import unique_marker

pytestmark = pytest.mark.e2e

PALETTE: Final = (
    ("red", (220, 20, 20)),
    ("green", (20, 160, 20)),
    ("blue", (20, 40, 220)),
    ("yellow", (240, 220, 20)),
    ("black", (0, 0, 0)),
)
STRIPES: Final = 3
STRIPE_SIDE: Final = 96
COLOR_NAME: Final = re.compile("|".join(name for name, _ in PALETTE))


def stripes_png(colors: tuple[tuple[int, int, int], ...]) -> bytes:
    row: Final = b"\x00" + b"".join(bytes(rgb) * STRIPE_SIDE for rgb in colors)
    header: Final = struct.pack(">IIBBBBB", STRIPE_SIDE * len(colors), STRIPE_SIDE, 8, 2, 0, 0, 0)
    return b"".join(
        (
            b"\x89PNG\r\n\x1a\n",
            _png_chunk(b"IHDR", header),
            _png_chunk(b"IDAT", zlib.compress(row * STRIPE_SIDE)),
            _png_chunk(b"IEND", b""),
        )
    )


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


@pytest.mark.parametrize("deployment", SETUPS)
class TestCodexVision:
    def test_names_the_colors_of_an_attached_image_without_the_shell(
        self, codex: CodexHarness, setup: ModelSetup
    ) -> None:
        stripes: Final = tuple(random.Random(unique_marker()).sample(PALETTE, STRIPES))
        image: Final = codex.home / "stripes.png"
        _ = image.write_bytes(stripes_png(tuple(rgb for _, rgb in stripes)))
        run: Final = codex.exec_with_flags(
            "The attached image is three vertical stripes of solid color. Without running any command, reply with"
            " the three colors from left to right, lowercase, separated by spaces.",
            model=setup.model,
            cwd=codex.new_workspace(files={}),
            flags=("--image", str(image)),
        )
        assert run.exit_code == 0, run
        assert run.commands == (), run
        assert tuple(match.group(0) for match in COLOR_NAME.finditer(run.answer.lower())) == tuple(
            name for name, _ in stripes
        ), run

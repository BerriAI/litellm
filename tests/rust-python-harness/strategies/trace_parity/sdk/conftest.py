from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest


@pytest.fixture(autouse=True)
def python_engine(monkeypatch: pytest.MonkeyPatch) -> Generator[None]:
    from litellm.rust_bridge import ocr as ocr_bridge

    previous: Final = ocr_bridge.rust_ocr_enabled()
    monkeypatch.setenv("LITELLM_RUST", "0")
    ocr_bridge.use_litellm_rust(False)
    try:
        yield
    finally:
        ocr_bridge.use_litellm_rust(previous)

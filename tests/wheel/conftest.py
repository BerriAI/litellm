from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "scripts"))  # test-quality-ok: the wheel inspector lives in .github/scripts, outside any importable package

from verify_linux_native_wheel import WheelFacts, WheelInspectionError, inspect_wheel  # noqa: E402  # CI helpers live outside Python packages


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--wheel", help="Release-built native wheel to inspect")


@pytest.fixture(scope="session")
def facts(request: pytest.FixtureRequest) -> WheelFacts:
    wheel: Final[object] = request.config.getoption("wheel")
    if not isinstance(wheel, str):
        pytest.fail("--wheel is required for release wheel checks")
    inspected: Final = inspect_wheel(Path(wheel))
    if isinstance(inspected, WheelInspectionError):
        pytest.fail(inspected.message)
    return inspected

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import pytest

from tests.test_litellm._fixture_recorder import fixture_id, recorded_fixtures

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
pytest_plugins: Final = ("tests.test_litellm.parity.pytest_plugin",)


def _fixture_directory() -> Path:
    configured: Final = os.environ.get(FIXTURE_DIR_ENV)
    return Path(configured).expanduser() if configured else Path(__file__).with_name(".fixtures")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "ocr_fixture" not in metafunc.fixturenames:
        return
    fixtures: Final = recorded_fixtures(_fixture_directory())
    if not fixtures:
        metafunc.parametrize(
            "ocr_fixture",
            (
                pytest.param(
                    None,
                    marks=pytest.mark.skip(reason=f"no recorded OCR fixtures in {_fixture_directory()}"),
                    id="no-recorded-fixtures",
                ),
            ),
        )
        return
    metafunc.parametrize("ocr_fixture", fixtures, ids=tuple(fixture_id(fixture) for fixture in fixtures))

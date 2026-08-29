from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from tests.test_litellm._fixture_recorder import fixture_id, recorded_fixtures

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"


def _fixture_directory() -> Path:
    if FIXTURE_DIR_ENV not in os.environ:
        return Path(__file__).with_name(".fixtures")
    configured: Final = os.environ[FIXTURE_DIR_ENV]
    if not configured:
        raise pytest.UsageError(f"{FIXTURE_DIR_ENV} is set but empty")
    return Path(configured).expanduser()


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "ocr_fixture" not in metafunc.fixturenames:
        return
    directory: Final = _fixture_directory()
    try:
        fixtures: Final = recorded_fixtures(directory)
    except (ValidationError, ValueError) as error:
        raise pytest.UsageError(
            f"Invalid OCR parity fixture bundle at {directory}. "
            "Each fixture must contain exactly `input` and `upstream_response`. "
            "Record fresh fixtures in an empty directory with: "
            f"`uv run python tests/test_litellm/ocr/generate_fixtures.py --fixture-dir {directory}`. "
            f"Validation details: {error}"
        ) from error
    if not fixtures:
        if FIXTURE_DIR_ENV in os.environ:
            raise pytest.UsageError(f"no recorded OCR fixtures in {directory}")
        metafunc.parametrize(
            "ocr_fixture",
            (
                pytest.param(
                    None,
                    marks=pytest.mark.skip(reason=f"no recorded OCR fixtures in {directory}"),
                    id="no-recorded-fixtures",
                ),
            ),
        )
        return
    metafunc.parametrize("ocr_fixture", fixtures, ids=tuple(fixture_id(fixture) for fixture in fixtures))

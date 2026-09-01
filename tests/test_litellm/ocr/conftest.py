from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.route_parity.fixtures.store import fixture_id, parametrize_recorded_fixtures
from tests.test_litellm.ocr.fixtures.models import OcrParityCase

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"


def _fixture_id(fixture: OcrParityCase) -> str:
    case_input: Final = fixture.litellm_input
    provider: Final = case_input.custom_llm_provider
    prefix: Final = f"{provider}/{case_input.model}" if provider else case_input.model
    return fixture_id(case_input, prefix)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    default_directory: Final = Path(__file__).with_name("fixtures") / "data"
    parametrize_recorded_fixtures(
        metafunc,
        fixture_name="ocr_fixture",
        case_type=OcrParityCase,
        env_var=FIXTURE_DIR_ENV,
        default_directory=default_directory,
        regeneration_command=(
            f"uv run python -m tests.test_litellm.ocr.fixtures.record --fixture-dir {default_directory}"
        ),
        id_builder=_fixture_id,
    )

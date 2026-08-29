from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.test_litellm._fixture_recorder import fixture_id, parametrize_recorded_fixtures
from tests.test_litellm.ocr.fixture_models import OcrParityCase

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"


def _fixture_id(fixture: OcrParityCase) -> str:
    case_input: Final = fixture.litellm_input
    model_id: Final = case_input.model.replace("/", "-")
    provider: Final = case_input.custom_llm_provider
    prefix: Final = f"{provider}-{model_id}" if provider and not model_id.startswith(f"{provider}-") else model_id
    return fixture_id(case_input, prefix)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    default_directory: Final = Path(__file__).with_name(".fixtures")
    parametrize_recorded_fixtures(
        metafunc,
        fixture_name="ocr_fixture",
        case_type=OcrParityCase,
        env_var=FIXTURE_DIR_ENV,
        default_directory=default_directory,
        regeneration_command=(
            "uv run python tests/test_litellm/ocr/generate_fixtures.py "
            f"--fixture-dir {default_directory}"
        ),
        id_builder=_fixture_id,
    )

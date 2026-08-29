from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Final

import pytest

from tests.test_litellm._json_fs_cache import JsonFileCache, canonical_json
from tests.test_litellm.ocr.fixture_models import OcrFixture

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
pytest_plugins: Final = ("tests.test_litellm.parity.pytest_plugin",)


def _fixture_directory() -> Path:
    configured: Final = os.environ.get(FIXTURE_DIR_ENV)
    return Path(configured).expanduser() if configured else Path(__file__).with_name(".fixtures")


def _recorded_fixtures() -> tuple[OcrFixture, ...]:
    raw_fixtures: Final = JsonFileCache(_fixture_directory()).values()
    return tuple(OcrFixture.model_validate(raw_fixture) for raw_fixture in raw_fixtures)


def _fixture_id(fixture: OcrFixture) -> str:
    raw_model: Final = fixture.request.sdk_kwargs.get("model")
    model: Final = raw_model if isinstance(raw_model, str) else "unknown-model"
    request_json: Final = canonical_json(fixture.request.provider_request.model_dump(mode="json"))
    digest: Final = hashlib.sha256(request_json.encode("utf-8")).hexdigest()[:8]
    return f"{fixture.request.provider}-{model.rsplit('/', 1)[-1]}-{digest}"


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "ocr_fixture" not in metafunc.fixturenames:
        return
    fixtures: Final = _recorded_fixtures()
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
    metafunc.parametrize("ocr_fixture", fixtures, ids=tuple(_fixture_id(fixture) for fixture in fixtures))

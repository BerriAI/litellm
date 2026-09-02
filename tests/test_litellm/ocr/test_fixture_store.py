from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Final, Protocol, cast

import pytest

from tests.route_parity.fixtures.cassette import deserialize_cassette
from tests.route_parity.fixtures.pytest_support import parametrize_recorded_fixtures
from tests.route_parity.fixtures.store import FixtureEnvelope, read_fixture, recorded_fixtures
from tests.test_litellm.ocr.conftest import ocr_fixture_id, ocr_fixture_marks
from tests.test_litellm.ocr.fixtures.migrate import migrate_fixture
from tests.test_litellm.ocr.fixtures.models import OcrParityCase


class _Parameter(Protocol):
    values: tuple[OcrParityCase, ...]
    marks: tuple[pytest.Mark, ...]


@dataclass(frozen=True, slots=True)
class _MetafuncSpy:
    fixturenames: tuple[str, ...]
    calls: Queue[tuple[object, ...]]

    def parametrize(self, *args: object, **_kwargs: object) -> None:
        self.calls.put(args)


def test_recorded_fixture_parametrization_applies_case_specific_marks() -> None:
    calls: Final[Queue[tuple[object, ...]]] = Queue()
    metafunc: Final = _MetafuncSpy(fixturenames=("ocr_fixture",), calls=calls)

    parametrize_recorded_fixtures(
        cast(pytest.Metafunc, metafunc),
        fixture_name="ocr_fixture",
        case_type=OcrParityCase,
        env_var="UNCONFIGURED_OCR_FIXTURE_TEST_DIRECTORY",
        default_directory=Path(__file__).with_name("fixtures") / "data",
        regeneration_command="unused",
        id_builder=ocr_fixture_id,
        marks_builder=ocr_fixture_marks,
    )

    parameters: Final = cast(tuple[_Parameter, ...], calls.get_nowait()[1])
    reducto_parameters: Final = tuple(
        parameter
        for parameter in parameters
        if parameter.values[0].litellm_input.contract in {"reducto_v3", "reducto_legacy"}
    )
    supported_parameters: Final = tuple(parameter for parameter in parameters if parameter not in reducto_parameters)

    assert reducto_parameters
    assert supported_parameters
    assert all(len(parameter.marks) == 1 for parameter in reducto_parameters)
    assert all(parameter.marks[0].name == "xfail" for parameter in reducto_parameters)
    assert all(parameter.marks[0].kwargs["strict"] is False for parameter in reducto_parameters)
    assert all(parameter.marks == () for parameter in supported_parameters)


def test_legacy_fixture_migration_preserves_responses_and_labels_reconstructed_requests(tmp_path: Path) -> None:
    case: Final = recorded_fixtures(Path(__file__).with_name("fixtures") / "data" / "mistral-ocr", OcrParityCase)[0]
    timestamp: Final = datetime(2020, 1, 1, tzinfo=timezone.utc)
    envelope: Final = FixtureEnvelope(
        schema_version=1,
        recorded_at=timestamp,
        case=case.model_dump(mode="json", exclude_unset=True),
    )
    legacy_path: Final = tmp_path / "legacy.json"
    legacy_path.write_text(envelope.model_dump_json())

    destination: Final = migrate_fixture(legacy_path)

    assert not legacy_path.exists()
    assert read_fixture(destination, OcrParityCase) == case
    cassette: Final = deserialize_cassette(destination.read_text())
    assert cassette.recorded_at == timestamp
    assert cassette.parity.request_source == "python_replay"
    assert len(cassette.interactions) == len(case.provider_responses)
    assert cassette.interactions[0].request.method == "POST"
    assert cassette.interactions[0].request.uri == "http://parity-provider.invalid/v1/ocr"
    assert "authorization" not in cassette.interactions[0].request.headers

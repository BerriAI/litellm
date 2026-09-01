from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Protocol, TypeVar, cast

import pytest
from pydantic import AwareDatetime, BaseModel, ConfigDict, TypeAdapter, ValidationError

FIXTURE_SCHEMA_VERSION: Final = 1
JSON_OBJECT: Final = TypeAdapter(dict[str, object])


class FixtureInput(Protocol):
    def canonical_input(self) -> dict[str, object]: ...


CaseT = TypeVar("CaseT", bound=BaseModel)


class FixtureEnvelope(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int
    recorded_at: AwareDatetime
    case: dict[str, object]


def canonical_json(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def fixture_cache_key(case_input: FixtureInput) -> dict[str, object]:
    return case_input.canonical_input()


def fixture_path(directory: Path, case_input: FixtureInput) -> Path:
    input_json: Final = canonical_json(fixture_cache_key(case_input))
    digest: Final = hashlib.sha256(input_json.encode("utf-8")).hexdigest()
    return directory / f"{digest}.json"


def load_fixture(directory: Path, case_input: FixtureInput, case_type: type[CaseT]) -> CaseT | None:
    path: Final = fixture_path(directory, case_input)
    if not path.is_file():
        return None
    return _load_fixture(JSON_OBJECT.validate_json(path.read_text(encoding="utf-8")), path, case_type)


def save_fixture(directory: Path, case_input: FixtureInput, case: BaseModel) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = fixture_path(directory, case_input)
    temporary_path: Final = path.with_suffix(".tmp")
    envelope: Final = FixtureEnvelope(
        schema_version=FIXTURE_SCHEMA_VERSION,
        recorded_at=datetime.now(timezone.utc),
        case=cast(dict[str, object], case.model_dump(mode="json", exclude_unset=True)),
    )
    serialized: Final = (
        json.dumps(envelope.model_dump(mode="json", exclude_unset=True), indent=2, sort_keys=True) + "\n"
    )
    temporary_path.write_text(serialized, encoding="utf-8")
    temporary_path.replace(path)
    return path


def _load_fixture(raw_fixture: dict[str, object], path: Path, case_type: type[CaseT]) -> CaseT:
    schema_version: Final = raw_fixture.get("schema_version")
    if schema_version != FIXTURE_SCHEMA_VERSION:
        raise ValueError(
            f"fixture {path} has schema_version {schema_version!r}, expected {FIXTURE_SCHEMA_VERSION}; "
            "delete it and regenerate the fixture bundle"
        )
    try:
        envelope: Final = FixtureEnvelope.model_validate(raw_fixture)
        return case_type.model_validate(envelope.case)
    except ValidationError as error:
        raise ValueError(f"invalid parity fixture {path} ({len(error.errors())} validation errors)") from error


def recorded_fixtures(directory: Path, case_type: type[CaseT]) -> tuple[CaseT, ...]:
    if not directory.is_dir():
        return ()
    paths: Final = tuple(sorted(directory.rglob("*.json")))
    return tuple(
        _load_fixture(JSON_OBJECT.validate_json(path.read_text(encoding="utf-8")), path, case_type) for path in paths
    )


def fixture_directory(configured: Path | None, env_value: str | None, default: Path) -> Path:
    return (configured or Path(env_value or default)).expanduser()


def fixture_id(case_input: FixtureInput, prefix: str) -> str:
    input_json: Final = canonical_json(case_input.canonical_input())
    digest: Final = hashlib.sha256(input_json.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{digest}"


def parametrize_recorded_fixtures(
    metafunc: pytest.Metafunc,
    *,
    fixture_name: str,
    case_type: type[CaseT],
    env_var: str,
    default_directory: Path,
    regeneration_command: str,
    id_builder: Callable[[CaseT], str],
) -> None:
    if fixture_name not in metafunc.fixturenames:
        return
    configured: Final = os.environ.get(env_var)
    if configured == "":
        raise pytest.UsageError(f"{env_var} is set but empty")
    directory: Final = Path(configured).expanduser() if configured is not None else default_directory
    try:
        fixtures: Final = recorded_fixtures(directory, case_type)
    except (ValidationError, ValueError) as error:
        raise pytest.UsageError(
            f"Invalid parity fixture bundle at {directory}. "
            "Each fixture must use the current versioned envelope. "
            f"Record fresh fixtures in an empty directory with: `{regeneration_command}`. "
            f"Validation details: {error}"
        ) from error
    if fixtures:
        metafunc.parametrize(fixture_name, fixtures, ids=tuple(id_builder(fixture) for fixture in fixtures))
        return
    if configured is not None:
        raise pytest.UsageError(f"no recorded fixtures in {directory}")
    metafunc.parametrize(
        fixture_name,
        (
            pytest.param(
                None,
                marks=pytest.mark.skip(reason=f"no recorded fixtures in {directory}"),
                id="no-recorded-fixtures",
            ),
        ),
    )

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, Protocol, TypeVar, cast

from pydantic import AwareDatetime, BaseModel, ConfigDict, TypeAdapter, ValidationError

from .cassette import deserialize_cassette, serialize_cassette
from .recording import RecordedInteraction

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
    return directory / f"{digest}.yaml"


def load_fixture(directory: Path, case_input: FixtureInput, case_type: type[CaseT]) -> CaseT | None:
    path: Final = fixture_path(directory, case_input)
    if path.is_file():
        return read_fixture(path, case_type)
    legacy_path: Final = path.with_suffix(".json")
    if not legacy_path.is_file():
        return None
    return read_fixture(legacy_path, case_type)


def save_fixture(
    directory: Path,
    case_input: FixtureInput,
    case: BaseModel,
    interactions: tuple[RecordedInteraction, ...],
    *,
    recorded_at: datetime | None = None,
    request_source: Literal["recorded", "python_replay"] = "recorded",
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = fixture_path(directory, case_input)
    serialized: Final = serialize_cassette(
        cast(dict[str, object], case.model_dump(mode="json", exclude_unset=True)),
        interactions,
        recorded_at or datetime.now(timezone.utc),
        request_source,
    )
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory, delete=False) as temporary:
        temporary_path: Final = Path(temporary.name)
        try:
            temporary.write(serialized)
            temporary.close()
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path


def read_fixture(path: Path, case_type: type[CaseT]) -> CaseT:
    contents: Final = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return _load_fixture(JSON_OBJECT.validate_json(contents), path, case_type)
    try:
        cassette: Final = deserialize_cassette(contents)
        return case_type.model_validate(cassette.case_data())
    except ValueError as error:
        raise ValueError(f"invalid parity cassette {path}") from error


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
    paths: Final = tuple(sorted((*directory.rglob("*.yaml"), *directory.rglob("*.json"))))
    return tuple(read_fixture(path, case_type) for path in paths)


def fixture_directory(configured: Path | None, env_value: str | None, default: Path) -> Path:
    return (configured or Path(env_value or default)).expanduser()


def fixture_id(case_input: FixtureInput, prefix: str) -> str:
    input_json: Final = canonical_json(case_input.canonical_input())
    digest: Final = hashlib.sha256(input_json.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{digest}"

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class Passed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["pass"] = "pass"


class Failed(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["fail"] = "fail"
    error: str = Field(min_length=1)


class NotApplicable(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["not_applicable"] = "not_applicable"
    reason: str = Field(min_length=1)


class NotTested(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["not_tested"] = "not_tested"


type Outcome = Annotated[Passed | Failed | NotApplicable | NotTested, Field(discriminator="status")]


class ResultRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_id: str
    provider: str
    model: str | None
    outcome: Outcome


RESULT_ROWS: Final = TypeAdapter(tuple[ResultRow, ...])


def read_rows(path: Path) -> tuple[ResultRow, ...]:
    return RESULT_ROWS.validate_json(path.read_bytes())


def write_rows(path: Path, rows: tuple[ResultRow, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _ = path.write_bytes(RESULT_ROWS.dump_json(sort_rows(rows), indent=2))


def sort_rows(rows: tuple[ResultRow, ...]) -> tuple[ResultRow, ...]:
    return tuple(sorted(rows, key=lambda row: (row.feature_id, row.provider, row.model or "")))


def results_path(client: str, environ: Mapping[str, str]) -> Path:
    return Path(environ.get(f"{client.upper()}_COMPAT_RESULTS_PATH", f"{client}-compat-results.json"))


def shard_dir(results: Path) -> Path:
    return results.with_name(f"{results.name}.shards")


def merged_rows(own: tuple[ResultRow, ...], shards: Path) -> tuple[ResultRow, ...]:
    shard_files: Final = tuple(sorted(shards.glob("*.json"))) if shards.is_dir() else ()
    return sort_rows((*own, *(row for shard in shard_files for row in read_rows(shard))))

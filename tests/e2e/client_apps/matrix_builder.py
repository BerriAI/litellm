from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Final, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictBool, TypeAdapter, model_validator

from client_apps.compat_results import (
    Failed,
    NotApplicable,
    NotTested,
    Outcome,
    Passed,
    ResultRow,
    read_rows,
)

CLIENT_APPS_DIR: Final = Path(__file__).resolve().parent
REGRESSION_EXIT_CODE: Final = 3


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ManifestFeature(_Frozen):
    id: str
    name: str


class KnownGap(_Frozen):
    feature: str
    provider: str
    reason: str = Field(min_length=1)
    strict: StrictBool = True


class Manifest(_Frozen):
    schema_version: str
    client: str
    package: str
    providers: tuple[str, ...]
    features: tuple[ManifestFeature, ...]
    known_gaps: tuple[KnownGap, ...] = ()

    @model_validator(mode="after")
    def _known_gaps_name_manifest_cells(self) -> Self:
        problems: Final = tuple(chain.from_iterable(self._gap_problems(gap) for gap in self.known_gaps))
        if problems:
            raise ValueError(f"known_gaps: {'; '.join(problems)}")
        cells: Final = tuple(f"{gap.feature}/{gap.provider}" for gap in self.known_gaps)
        repeated: Final = sorted(frozenset(cell for cell in cells if cells.count(cell) > 1))
        if repeated:
            raise ValueError(f"known_gaps lists a cell more than once: {', '.join(repeated)}")
        return self

    def _gap_problems(self, gap: KnownGap) -> tuple[str, ...]:
        feature_problem: Final = (
            () if any(feature.id == gap.feature for feature in self.features) else (f"unknown feature {gap.feature!r}",)
        )
        provider_problem: Final = () if gap.provider in self.providers else (f"unknown provider {gap.provider!r}",)
        return (*feature_problem, *provider_problem)

    def known_gap(self, feature_id: str, provider: str) -> KnownGap | None:
        return next(
            (gap for gap in self.known_gaps if gap.feature == feature_id and gap.provider == provider),
            None,
        )


class MatrixFeature(_Frozen):
    id: str
    name: str
    providers: Mapping[str, Outcome]


class Matrix(_Frozen):
    schema_version: str
    generated_at: str
    litellm_version: str
    client: str
    client_version: str
    providers: tuple[str, ...]
    features: tuple[MatrixFeature, ...]


class Regression(_Frozen):
    feature_id: str
    feature_name: str
    provider: str
    error: str


class _PackageJson(BaseModel):
    model_config = ConfigDict(frozen=True)

    dependencies: Mapping[str, str]


class BuildArgs(_Frozen):
    command: Literal["build"]
    manifest: Path
    results: Path
    out: Path
    litellm_version: str | None


class RegressionArgs(_Frozen):
    command: Literal["regressions"]
    old: Path
    new: Path


CLI_ARGS: Final = TypeAdapter[BuildArgs | RegressionArgs](
    Annotated[BuildArgs | RegressionArgs, Field(discriminator="command")]
)
REGRESSIONS: Final = TypeAdapter(tuple[Regression, ...])


def load_manifest(path: Path) -> Manifest:
    return Manifest.model_validate(yaml.safe_load(path.read_text()))


def pinned_client_version(package: str, package_json: Path = CLIENT_APPS_DIR / "package.json") -> str:
    return _PackageJson.model_validate_json(package_json.read_bytes()).dependencies[package]


def aggregate(rows: tuple[ResultRow, ...]) -> Outcome:
    tested: Final = tuple(row for row in rows if not isinstance(row.outcome, NotTested))
    failures: Final = tuple(_failure(row.model, row.outcome) for row in tested if isinstance(row.outcome, Failed))
    if failures:
        return Failed(error="; ".join(failures))
    if any(isinstance(row.outcome, Passed) for row in tested):
        return Passed()
    reasons: Final = tuple(row.outcome.reason for row in tested if isinstance(row.outcome, NotApplicable))
    if reasons:
        return NotApplicable(reason=reasons[0])
    return NotTested()


def _failure(model: str | None, outcome: Failed) -> str:
    return outcome.error if model is None else f"{model}: {outcome.error}"


def build_matrix(
    manifest: Manifest,
    rows: tuple[ResultRow, ...],
    *,
    litellm_version: str,
    client_version: str,
    generated_at: datetime,
) -> Matrix:
    return Matrix(
        schema_version=manifest.schema_version,
        generated_at=generated_at.isoformat(),
        litellm_version=litellm_version,
        client=manifest.client,
        client_version=client_version,
        providers=manifest.providers,
        features=tuple(
            MatrixFeature(
                id=feature.id,
                name=feature.name,
                providers=MappingProxyType(
                    {provider: aggregate(_cell_rows(rows, feature.id, provider)) for provider in manifest.providers}
                ),
            )
            for feature in manifest.features
        ),
    )


def _cell_rows(rows: tuple[ResultRow, ...], feature_id: str, provider: str) -> tuple[ResultRow, ...]:
    return tuple(row for row in rows if row.feature_id == feature_id and row.provider == provider)


def _cells(matrix: Matrix) -> Iterator[tuple[MatrixFeature, str, Outcome]]:
    for feature in matrix.features:
        for provider, outcome in feature.providers.items():
            yield feature, provider, outcome


def find_regressions(old: Matrix, new: Matrix) -> tuple[Regression, ...]:
    old_cells: Final = MappingProxyType({(feature.id, provider): outcome for feature, provider, outcome in _cells(old)})
    return tuple(
        sorted(
            (
                Regression(feature_id=feature.id, feature_name=feature.name, provider=provider, error=outcome.error)
                for feature, provider, outcome in _cells(new)
                if isinstance(outcome, Failed) and isinstance(old_cells.get((feature.id, provider)), Passed)
            ),
            key=lambda regression: (regression.feature_id, regression.provider),
        )
    )


def _build(args: BuildArgs) -> int:
    manifest: Final = load_manifest(args.manifest)
    matrix: Final = build_matrix(
        manifest,
        read_rows(args.results),
        litellm_version=args.litellm_version or version("litellm"),
        client_version=pinned_client_version(manifest.package),
        generated_at=datetime.now(timezone.utc),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    _ = args.out.write_text(matrix.model_dump_json(indent=2))
    return 0


def _regressions(args: RegressionArgs) -> int:
    if not args.old.exists():
        return 0
    regressions: Final = find_regressions(
        Matrix.model_validate_json(args.old.read_bytes()), Matrix.model_validate_json(args.new.read_bytes())
    )
    _ = sys.stdout.write(REGRESSIONS.dump_json(regressions, indent=2).decode() + "\n")
    return REGRESSION_EXIT_CODE if regressions else 0


def main(argv: tuple[str, ...]) -> int:
    parser: Final = ArgumentParser(prog="python -m client_apps.matrix_builder")
    commands: Final = parser.add_subparsers(dest="command", required=True)
    build: Final = commands.add_parser("build")
    _ = build.add_argument("--manifest", required=True)
    _ = build.add_argument("--results", required=True)
    _ = build.add_argument("--out", required=True)
    _ = build.add_argument("--litellm-version", dest="litellm_version", default=None)
    regressions: Final = commands.add_parser("regressions")
    _ = regressions.add_argument("--old", required=True)
    _ = regressions.add_argument("--new", required=True)
    match CLI_ARGS.validate_python(vars(parser.parse_args(argv))):
        case BuildArgs() as build_args:
            return _build(build_args)
        case RegressionArgs() as regression_args:
            return _regressions(regression_args)


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))

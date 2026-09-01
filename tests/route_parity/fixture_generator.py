from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Generic, Protocol, TypeVar, cast

from hypothesis.strategies import SearchStrategy
from pydantic import BaseModel

from tests.route_parity.fixture_models import SdkInputBase
from tests.route_parity.fixture_recorder import ProviderSpec, generate_case_inputs, record_cases

LOGGER: Final = logging.getLogger(__name__)
InputT = TypeVar("InputT", bound=SdkInputBase)
InputT_contra = TypeVar("InputT_contra", bound=SdkInputBase, contravariant=True)
DependencyT_contra = TypeVar("DependencyT_contra", contravariant=True)
CaseT = TypeVar("CaseT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class GeneratorArgs:
    concurrency: int
    examples: int
    fixture_dir: Path | None


@dataclass(frozen=True, slots=True)
class FixtureTarget(Generic[InputT]):
    name: str
    provider_spec: ProviderSpec
    strategy: SearchStrategy[InputT]
    invocation: FixtureInvocation[InputT]
    required_inputs: tuple[InputT, ...] = ()


class FixtureInvocation(Protocol[InputT_contra]):
    def execute(self, provider_url: str, case_input: InputT_contra) -> None: ...


class FixtureSource(Protocol[InputT, DependencyT_contra]):
    def targets(
        self,
        environ: Mapping[str, str],
        dependency: DependencyT_contra,
        /,
    ) -> tuple[FixtureTarget[InputT], ...]: ...


def discover_fixture_targets(
    sources: tuple[FixtureSource[InputT, DependencyT_contra], ...],
    environ: Mapping[str, str],
    dependency: DependencyT_contra,
) -> tuple[FixtureTarget[InputT], ...]:
    return tuple(target for source in sources for target in source.targets(environ, dependency))


def generate_target_fixtures(
    target: FixtureTarget[InputT],
    root: Path,
    examples: int,
    concurrency: int,
    case_type: type[CaseT],
) -> None:
    generated_inputs: Final = generate_case_inputs(target.strategy, examples)
    case_inputs: Final = (*target.required_inputs, *generated_inputs)
    results: Final = record_cases(
        target.provider_spec,
        root / target.name,
        case_inputs,
        target.invocation.execute,
        case_type,
        concurrency,
    )
    for result in results:
        LOGGER.info(
            "%s %s",
            "cached" if result.cache_hit else "recorded",
            target.name,
        )


def require_targets(
    targets: tuple[FixtureTarget[InputT], ...], error_message: str
) -> tuple[FixtureTarget[InputT], ...]:
    if targets:
        return targets
    raise SystemExit(error_message)


def parse_generator_args(argv: Sequence[str] | None = None) -> GeneratorArgs:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--examples", type=int, default=4)
    parser.add_argument("--fixture-dir", type=Path)
    namespace: Final = parser.parse_args(argv)
    return GeneratorArgs(
        concurrency=cast(int, namespace.concurrency),
        examples=cast(int, namespace.examples),
        fixture_dir=cast(Path | None, namespace.fixture_dir),
    )

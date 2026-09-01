from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Generic, Literal, Protocol, TypeVar, cast

from hypothesis.strategies import SearchStrategy
from pydantic import BaseModel

from tests.route_parity.fixtures.inputs import generate_case_inputs
from tests.route_parity.fixtures.recording import ProviderSpec, record_upstream_responses
from tests.route_parity.fixtures.store import (
    FixtureInput,
    canonical_json,
    fixture_cache_key,
    fixture_id,
    fixture_path,
    load_fixture,
    save_fixture,
)

LOGGER: Final = logging.getLogger(__name__)
InputT = TypeVar("InputT", bound=FixtureInput)
InputT_contra = TypeVar("InputT_contra", bound=FixtureInput, contravariant=True)
CaseT = TypeVar("CaseT", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class RecordingArgs:
    concurrency: int
    examples: int
    fixture_dir: Path | None


class RecordingInvocation(Protocol[InputT_contra]):
    def execute(self, provider_url: str, case_input: InputT_contra) -> None: ...


@dataclass(frozen=True, slots=True)
class RecordingTarget(Generic[InputT]):
    name: str
    provider_spec: ProviderSpec
    strategy: SearchStrategy[InputT]
    invocation: RecordingInvocation[InputT] = field(repr=False)
    required_inputs: tuple[InputT, ...] = ()


@dataclass(frozen=True, slots=True)
class RecordingJob(Generic[InputT]):
    target_name: str
    directory: Path
    provider_spec: ProviderSpec
    case_input: InputT
    invocation: RecordingInvocation[InputT] = field(repr=False)

    @property
    def case_id(self) -> str:
        return fixture_id(self.case_input, self.target_name)


@dataclass(frozen=True, slots=True)
class RecordedFixture:
    target_name: str
    case_id: str
    path: Path
    kind: Literal["recorded"] = field(default="recorded", init=False)


@dataclass(frozen=True, slots=True)
class CachedFixture:
    target_name: str
    case_id: str
    path: Path
    kind: Literal["cached"] = field(default="cached", init=False)


@dataclass(frozen=True, slots=True)
class FailedFixture:
    target_name: str
    case_id: str
    error: Exception = field(repr=False)
    kind: Literal["failed"] = field(default="failed", init=False)


RecordingOutcome = RecordedFixture | CachedFixture | FailedFixture


@dataclass(frozen=True, slots=True)
class RecordingSummary:
    recorded: tuple[RecordedFixture, ...]
    cached: tuple[CachedFixture, ...]
    failed: tuple[FailedFixture, ...]

    @property
    def exit_code(self) -> int:
        return 1 if self.failed else 0


def _unique_inputs(target: RecordingTarget[InputT], examples: int) -> tuple[InputT, ...]:
    generated_inputs: Final = generate_case_inputs(target.strategy, examples)
    case_inputs: Final = (*target.required_inputs, *generated_inputs)
    return tuple({canonical_json(fixture_cache_key(case_input)): case_input for case_input in case_inputs}.values())


def build_recording_jobs(
    targets: tuple[RecordingTarget[InputT], ...],
    root: Path,
    examples: int,
) -> tuple[RecordingJob[InputT], ...]:
    if examples < 1:
        raise ValueError("examples must be at least 1")
    return tuple(
        RecordingJob(
            target_name=target.name,
            directory=root / target.name,
            provider_spec=target.provider_spec,
            case_input=case_input,
            invocation=target.invocation,
        )
        for target in targets
        for case_input in _unique_inputs(target, examples)
    )


def _record_job(job: RecordingJob[InputT], case_type: type[CaseT]) -> RecordedFixture | CachedFixture:
    cached: Final = load_fixture(job.directory, job.case_input, case_type)
    if cached is not None:
        return CachedFixture(
            target_name=job.target_name,
            case_id=job.case_id,
            path=fixture_path(job.directory, job.case_input),
        )
    provider_responses: Final = record_upstream_responses(
        job.provider_spec,
        job.case_input,
        job.invocation.execute,
    )
    case: Final = case_type.model_validate({"litellm_input": job.case_input, "provider_responses": provider_responses})
    path: Final = save_fixture(job.directory, job.case_input, case)
    return RecordedFixture(target_name=job.target_name, case_id=job.case_id, path=path)


def _completed_outcome(
    completed: int,
    total: int,
    job: RecordingJob[InputT],
    future: Future[RecordedFixture | CachedFixture],
) -> RecordingOutcome:
    try:
        outcome: Final = future.result()
    except Exception as error:
        failed: Final = FailedFixture(target_name=job.target_name, case_id=job.case_id, error=error)
        LOGGER.error(
            "[%d/%d] failed %s %s: %s",
            completed,
            total,
            failed.target_name,
            failed.case_id,
            type(error).__name__,
        )
        return failed
    LOGGER.info("[%d/%d] %s %s %s", completed, total, outcome.kind, outcome.target_name, outcome.case_id)
    return outcome


def record_fixtures(
    targets: tuple[RecordingTarget[InputT], ...],
    root: Path,
    examples: int,
    concurrency: int,
    case_type: type[CaseT],
) -> RecordingSummary:
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    jobs: Final = build_recording_jobs(targets, root, examples)
    total: Final = len(jobs)
    LOGGER.info("Recording %d fixtures across %d targets with concurrency %d", total, len(targets), concurrency)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_jobs: Final = MappingProxyType({executor.submit(_record_job, job, case_type): job for job in jobs})
        outcomes: Final = tuple(
            _completed_outcome(completed, total, future_jobs[future], future)
            for completed, future in enumerate(as_completed(future_jobs), start=1)
        )
    summary: Final = RecordingSummary(
        recorded=tuple(outcome for outcome in outcomes if isinstance(outcome, RecordedFixture)),
        cached=tuple(outcome for outcome in outcomes if isinstance(outcome, CachedFixture)),
        failed=tuple(outcome for outcome in outcomes if isinstance(outcome, FailedFixture)),
    )
    LOGGER.info(
        "Finished %d fixtures: %d recorded, %d cached, %d failed",
        total,
        len(summary.recorded),
        len(summary.cached),
        len(summary.failed),
    )
    return summary


def _positive_int(value: str) -> int:
    parsed: Final = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_recording_args(argv: Sequence[str] | None = None) -> RecordingArgs:
    parser: Final = argparse.ArgumentParser()
    parser.add_argument("--concurrency", type=_positive_int, default=4)
    parser.add_argument("--examples", type=_positive_int, default=4)
    parser.add_argument("--fixture-dir", type=Path)
    namespace: Final = parser.parse_args(argv)
    return RecordingArgs(
        concurrency=cast(int, namespace.concurrency),
        examples=cast(int, namespace.examples),
        fixture_dir=cast(Path | None, namespace.fixture_dir),
    )

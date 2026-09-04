from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Generic, Literal, Protocol, TypeVar

from hypothesis.strategies import SearchStrategy
from pydantic import BaseModel

from .inputs import generate_case_inputs
from .recording import UpstreamEndpoint, record_upstream_interactions
from .store import (
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


class RecordingInvocation(Protocol[InputT_contra]):
    def execute(self, provider_url: str, case_input: InputT_contra) -> None: ...


@dataclass(frozen=True, slots=True)
class RecordingTarget(Generic[InputT]):
    name: str
    upstream: UpstreamEndpoint
    strategy: SearchStrategy[InputT]
    invocation: RecordingInvocation[InputT] = field(repr=False)
    required_inputs: tuple[InputT, ...] = ()


@dataclass(frozen=True, slots=True)
class RecordingJob(Generic[InputT]):
    target_name: str
    directory: Path
    upstream: UpstreamEndpoint
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
            upstream=target.upstream,
            case_input=case_input,
            invocation=target.invocation,
        )
        for target in targets
        for case_input in _unique_inputs(target, examples)
    )


def _record_job(job: RecordingJob[InputT], case_type: type[CaseT]) -> RecordedFixture | CachedFixture:
    cached: Final = load_fixture(job.directory, job.case_input, case_type)
    if cached is not None:
        path: Final = fixture_path(job.directory, job.case_input)
        return CachedFixture(
            target_name=job.target_name,
            case_id=job.case_id,
            path=path if path.is_file() else path.with_suffix(".json"),
        )
    interactions: Final = record_upstream_interactions(
        job.upstream,
        job.case_input,
        job.invocation.execute,
    )
    status: Final = interactions[-1].response.status_code
    if status in {408, 429} or status >= 500:
        raise RuntimeError(f"Upstream returned transient HTTP {status}; rerun recording to retry")
    case: Final = case_type.model_validate(
        {
            "litellm_input": job.case_input,
            "provider_responses": tuple(item.response for item in interactions),
        }
    )
    saved_path: Final = save_fixture(job.directory, job.case_input, case, interactions)
    return RecordedFixture(target_name=job.target_name, case_id=job.case_id, path=saved_path)


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

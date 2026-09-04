from __future__ import annotations

import asyncio
import tempfile
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Final, Generic, Protocol, TypeVar

from pydantic import BaseModel

from ....shared.parity.compare import assert_parity
from ....shared.parity.models import SDKCommand, SDKReport, WorkerFailure, WorkerResult, WorkerSuccess
from ....shared.parity.normalization import NormalizationSpec, normalize_execution
from ....shared.parity.recorded_http import ReplayItem
from ....shared.parity.runner import ExecutionVariant, SubprocessRunner, SubprocessWorker, execution_worker_pair
from ..runner import E2ECheck

CaseT = TypeVar("CaseT", bound=BaseModel)


class SdkParityContract(Protocol[CaseT]):
    @property
    def name(self) -> str: ...

    @property
    def modes(self) -> tuple[str, ...]: ...

    @property
    def baseline(self) -> ExecutionVariant: ...

    @property
    def candidate(self) -> ExecutionVariant: ...

    @property
    def baseline_user_agent(self) -> str: ...

    def cases(self) -> tuple[CaseT, ...]: ...

    def dump_case(self, case: CaseT) -> bytes: ...

    def load_case(self, data: bytes) -> CaseT: ...

    def case_name(self, case: CaseT, mode: str) -> str: ...

    def responses(self, case: CaseT) -> tuple[ReplayItem, ...]: ...

    def normalization(self, case: CaseT) -> NormalizationSpec: ...

    def execute(
        self,
        case: CaseT,
        mode: str,
        mock_url: str,
        event_loop: asyncio.AbstractEventLoop,
    ) -> SDKReport: ...

    def assert_baseline(self, case: CaseT, report: SDKReport) -> None: ...


class BaseSdkParityContract(Generic[CaseT]):
    def normalization(self, case: CaseT) -> NormalizationSpec:
        del case
        return NormalizationSpec()

    def assert_baseline(self, case: CaseT, report: SDKReport) -> None:
        del case, report


def _check_case(
    contract: SdkParityContract[CaseT],
    case: CaseT,
    mode: str,
    case_file: Path,
    workers: tuple[SubprocessWorker, SubprocessWorker],
) -> None:
    baseline_worker, candidate_worker = workers
    responses: Final = contract.responses(case)
    baseline: Final = baseline_worker.execute(case_file, mode, responses)
    candidate: Final = candidate_worker.execute(case_file, mode, responses)
    normalization: Final = contract.normalization(case)
    normalized_baseline: Final = normalize_execution(baseline, normalization)
    normalized_candidate: Final = normalize_execution(candidate, normalization)
    assert_parity(normalized_baseline, normalized_candidate, contract.baseline_user_agent)
    contract.assert_baseline(case, normalized_baseline.report)


def _write_cases(directory: Path, contract: SdkParityContract[CaseT], cases: tuple[CaseT, ...]) -> tuple[Path, ...]:
    paths: Final = tuple(directory / f"case-{index}.json" for index in range(len(cases)))
    for path, case in zip(paths, cases, strict=True):
        path.write_bytes(contract.dump_case(case))
    return paths


@contextmanager
def contract_checks(
    contract: SdkParityContract[CaseT],
    entrypoint: Path,
) -> Generator[tuple[E2ECheck, ...]]:
    cases: Final = contract.cases()
    runner: Final = SubprocessRunner(
        entrypoint=entrypoint,
        baseline_user_agent=contract.baseline_user_agent,
        route_label=contract.name,
    )
    with tempfile.TemporaryDirectory(prefix=f"litellm-{contract.name.lower()}-parity-") as raw_directory:
        paths: Final = _write_cases(Path(raw_directory), contract, cases)
        with execution_worker_pair(runner, contract.baseline, contract.candidate) as workers:
            yield tuple(
                E2ECheck(
                    contract.case_name(case, mode),
                    partial(_check_case, contract, case, mode, path, workers),
                )
                for case, path in zip(cases, paths, strict=True)
                for mode in contract.modes
            )


def execute_contract_command(
    contract: SdkParityContract[CaseT],
    command_json: str,
    mock_url: str,
    event_loop: asyncio.AbstractEventLoop,
) -> WorkerResult:
    try:
        command: Final = SDKCommand.model_validate_json(command_json)
        case: Final = contract.load_case(Path(command.case_file).read_bytes())
        return WorkerSuccess(report=contract.execute(case, command.route, mock_url, event_loop))
    except Exception:
        return WorkerFailure(error=traceback.format_exc())

from pathlib import Path
from typing import Final

from ...shared.reporting.models import SDK_FUNCTIONS, Coverage
from ...shared.reporting.strategy import (
    CaseDefinition,
    ModuleCaseSpec,
    NotImplementedCaseSpec,
    RunnerArgumentDefinition,
    StrategyDefinition,
)
from .reporting import render_benchmark_results
from .runner import run_benchmark_cases

STRATEGY: Final = StrategyDefinition(
    id="e2e_benchmark",
    order=15,
    label="End-to-end benchmark",
    description="Compare Python/Rust SDK latency, CPU time, and RSS using local provider replays.",
    directory=Path(__file__).parent,
    runnable_spec=ModuleCaseSpec,
    cases=tuple(
        CaseDefinition(
            function,
            ModuleCaseSpec(
                coverage=Coverage.PARTIAL,
                module="tests.rust-python-harness.strategies.e2e_benchmark.workloads",
                note="Sync/async Mistral OCR with scaled recorded fixtures; concurrency is one.",
            )
            if function == "ocr"
            else NotImplementedCaseSpec(reason="No benchmark workload is implemented for this SDK function yet."),
            surface="sdk",
        )
        for function in SDK_FUNCTIONS
    ),
    run=run_benchmark_cases,
    render=render_benchmark_results,
    surfaces=("sdk",),
    runner_argument=RunnerArgumentDefinition(
        option="--benchmark-arg",
        help="benchmark option, e.g. --benchmark-arg=--iterations=100; see the strategy README",
    ),
)

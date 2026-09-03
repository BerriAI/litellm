from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ...shared.reporting.models import HarnessCase, HarnessRun
from ...shared.reporting.pytest_runner import UpdateCallback, run_pytest


def run(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    pytest_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    return run_pytest(cases, repo_root, on_update, pytest_args)


def main(argv: Sequence[str] | None = None) -> int:
    from ...cli import main as harness_main

    return harness_main(argv, strategy_id="e2e_parity")


if __name__ == "__main__":
    raise SystemExit(main())

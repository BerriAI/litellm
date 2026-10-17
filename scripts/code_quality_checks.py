from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT: Final = Path(__file__).resolve().parents[1]
MANIFEST: Final = REPO_ROOT / ".github" / "code-quality-checks.yml"


class Check(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    run: str
    env: Mapping[str, str] = Field(default_factory=dict)
    ci_only: str | None = None


class Manifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    checks: tuple[Check, ...]


@dataclass(frozen=True, slots=True)
class Outcome:
    check: Check
    returncode: int
    output: str
    seconds: float


def load_checks(manifest: Path) -> tuple[Check, ...]:
    return Manifest.model_validate(yaml.safe_load(manifest.read_text(encoding="utf-8"))).checks


def execute(check: Check) -> Outcome:
    started: Final = time.monotonic()
    completed: Final = subprocess.run(
        ("bash", "-e", "-c", check.run),
        cwd=REPO_ROOT,
        env={**os.environ, **check.env},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return Outcome(
        check=check, returncode=completed.returncode, output=completed.stdout, seconds=time.monotonic() - started
    )


def _reported(outcome: Outcome, *, annotate: bool) -> Outcome:
    name: Final = outcome.check.name
    if outcome.returncode == 0 and annotate:
        print(f"::group::PASS {name} ({outcome.seconds:.1f}s)\n{outcome.output.rstrip()}\n::endgroup::")
    elif outcome.returncode == 0:
        print(f"PASS {name} ({outcome.seconds:.1f}s)")
    else:
        print(f"FAIL {name} ({outcome.seconds:.1f}s, exit {outcome.returncode})\n{outcome.output.rstrip()}")
        if annotate:
            print(f"::error title=code-quality::{name} failed with exit code {outcome.returncode}")
    sys.stdout.flush()
    return outcome


def run_checks(
    checks: tuple[Check, ...],
    *,
    local: bool,
    annotate: bool,
    run_one: Callable[[Check], Outcome] = execute,
) -> int:
    skipped: Final = tuple(check for check in checks if local and check.ci_only)
    runnable: Final = tuple(check for check in checks if not (local and check.ci_only))
    for check in skipped:
        print(f"SKIP {check.name} (CI-only: {check.ci_only})")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        outcomes: Final = tuple(_reported(outcome, annotate=annotate) for outcome in pool.map(run_one, runnable))
    failed: Final = tuple(outcome.check.name for outcome in outcomes if outcome.returncode != 0)
    print(
        f"code-quality: {len(outcomes) - len(failed)} passed, {len(failed)} failed, {len(skipped)} skipped as CI-only"
    )
    print(f"code-quality: FAIL {', '.join(failed)}" if failed else "code-quality: PASS")
    return 1 if failed else 0


def main() -> int:
    return run_checks(
        load_checks(MANIFEST),
        local="--local" in sys.argv[1:],
        annotate=os.environ.get("GITHUB_ACTIONS") == "true",
    )


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run the tests/code_coverage_tests scripts that test-linting.yml's code-quality job runs.

The list is read from the workflow itself, so a script added to that job runs under
`make lint` (and so `make check`) as soon as it lands, and a local pass means those
CI steps pass too.
"""

import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT: Final = Path(__file__).resolve().parent.parent
WORKFLOW: Final = REPO_ROOT / ".github" / "workflows" / "test-linting.yml"
JOB: Final = "code-quality"
SCRIPT_STEP: Final = re.compile(r"uv run --no-sync python (?P<script>\./tests/code_coverage_tests/[\w/]+\.py)")
REWRITES_TRACKED_FILES: Final = frozenset({"./tests/code_coverage_tests/check_licenses.py"})


class Step(BaseModel):
    model_config = ConfigDict(frozen=True)
    run: str = ""


class Job(BaseModel):
    model_config = ConfigDict(frozen=True)
    steps: tuple[Step, ...] = ()


class Workflow(BaseModel):
    model_config = ConfigDict(frozen=True)
    jobs: Mapping[str, Job] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Outcome:
    script: str
    returncode: int
    output: str


def scripts(workflow_source: str) -> tuple[str, ...]:
    job: Final = Workflow.model_validate(yaml.safe_load(workflow_source)).jobs.get(JOB, Job())
    matches: Final = tuple(SCRIPT_STEP.fullmatch(step.run.strip()) for step in job.steps)
    return tuple(match["script"] for match in matches if match and match["script"] not in REWRITES_TRACKED_FILES)


def run(script: str, cwd: Path) -> Outcome:
    completed: Final = subprocess.run((sys.executable, script), cwd=cwd, capture_output=True, text=True, check=False)
    return Outcome(script, completed.returncode, completed.stdout + completed.stderr)


def failures(selected: Sequence[str], cwd: Path) -> tuple[Outcome, ...]:
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        outcomes: Final = tuple(pool.map(partial(run, cwd=cwd), selected))
    return tuple(outcome for outcome in outcomes if outcome.returncode != 0)


def main(workflow: Path = WORKFLOW, cwd: Path = REPO_ROOT) -> int:
    selected: Final = scripts(workflow.read_text())
    if not selected:
        sys.stderr.write(f"✗ No tests/code_coverage_tests scripts found in the {JOB} job of {workflow}\n")
        return 1
    failed: Final = failures(selected, cwd)
    sys.stderr.write(
        "".join(f"✗ {outcome.script} exited {outcome.returncode}\n{outcome.output}\n" for outcome in failed)
    )
    sys.stdout.write(f"{JOB}: {len(selected) - len(failed)} of {len(selected)} scripts passed\n")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

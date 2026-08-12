"""Guard the CI cache for Prisma's CLI and engine binaries.

``prisma generate`` shells out to ``npm install prisma@<version>`` whenever the
prisma-client-py binary cache directory has no CLI entrypoint, pulling ~85 MB of
engines over the network. The download is normally seconds and occasionally
minutes, and a job timeout cannot tell the difference from a hung test, so an
uncached job is one slow npm response away from cancelling a passing test run.

Three invariants keep that download off the critical path:

1. No workflow sets ``PRISMA_BINARY_CACHE_DIR``. The prisma-client-py default is
   ``~/.cache/prisma-python/binaries/<prisma-version>/<engine-version>``, already
   keyed by both versions and the only path the cache action restores. Pointing
   it elsewhere (``runner.temp`` especially, which is wiped every job) silently
   guarantees a cold download.
2. Every job that generates the client also restores the cache.
3. The cache key resolves to a real version from ``uv.lock``. The action fails
   the job when it cannot, so a lock format change must break here instead.
"""

import re
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Final

import yaml
from pydantic import BaseModel, Field, ValidationError

REPO_ROOT: Final = Path(__file__).resolve().parent.parent.parent
WORKFLOWS_DIR: Final = REPO_ROOT / ".github" / "workflows"
UV_LOCK: Final = REPO_ROOT / "uv.lock"
CACHE_ACTION: Final = "./.github/actions/cache-prisma-binaries"

# Commands that reach the prisma binary cache: a direct generate, or a script
# that runs one on the caller's behalf.
PRISMA_GENERATE_MARKERS: Final = ("prisma generate", "type_check_gate.py")


class PrismaBinaryCacheError(Exception):
    pass


def resolve_prisma_version(lock_text: str) -> str | None:
    """Mirror of the shell lookup in the cache action's version step."""
    match: Final = re.search(
        r'^name = "prisma"\n^version = "(?P<version>[^"]+)"$',
        lock_text,
        re.MULTILINE,
    )
    return match.group("version") if match else None


class WorkflowStep(BaseModel):
    """The two step fields this guard reads; every other key is ignored."""

    run: str | None = None
    uses: str | None = None

    def generates_prisma_client(self) -> bool:
        return self.run is not None and any(m in self.run for m in PRISMA_GENERATE_MARKERS)

    def restores_cache(self) -> bool:
        return self.uses == CACHE_ACTION


class WorkflowJob(BaseModel):
    # Absent for jobs that delegate to a reusable workflow via a job-level `uses`.
    steps: tuple[WorkflowStep, ...] = ()


class Workflow(BaseModel):
    jobs: Mapping[str, WorkflowJob] = Field(default_factory=dict)


def parse_workflow(text: str) -> Workflow | str:
    """Validate untyped YAML at the boundary so the checks below stay typed.

    Returns the parsed workflow, or a description of why it could not be read.
    """
    parsed: Final = yaml.safe_load(text)
    try:
        return Workflow.model_validate(parsed if isinstance(parsed, dict) else {})
    except ValidationError as exc:
        return f"does not parse as a workflow: {exc.error_count()} schema error(s)"


def lock_errors(lock_text: str) -> Iterator[str]:
    if not resolve_prisma_version(lock_text):
        yield (
            "uv.lock has no resolvable `prisma` package version. The version step "
            f"in {CACHE_ACTION} greps the same shape and will fail every job that "
            "generates the Prisma client."
        )


def workflow_errors(rel: Path, text: str) -> Iterator[str]:
    if "PRISMA_BINARY_CACHE_DIR" in text:
        yield (
            f"{rel}: sets PRISMA_BINARY_CACHE_DIR. Leave it unset so the binaries "
            f"land in the version-keyed default path the {CACHE_ACTION} action restores."
        )

    workflow: Final = parse_workflow(text)
    if isinstance(workflow, str):
        yield f"{rel}: {workflow}"
        return

    for job_name, job in workflow.jobs.items():
        if any(s.generates_prisma_client() for s in job.steps) and not any(
            s.restores_cache() for s in job.steps
        ):
            yield (
                f"{rel}: job `{job_name}` generates the Prisma client without a "
                f"`uses: {CACHE_ACTION}` step, so it downloads ~85 MB of engines "
                "on every run."
            )


def main() -> None:
    errors: Final = (
        *lock_errors(UV_LOCK.read_text()),
        *(
            error
            for path in sorted(WORKFLOWS_DIR.glob("*.y*ml"))
            for error in workflow_errors(path.relative_to(REPO_ROOT), path.read_text())
        ),
    )

    if errors:
        raise PrismaBinaryCacheError(
            "Prisma binary cache invariants violated:\n  - " + "\n  - ".join(errors)
        )

    print("Prisma binary cache invariants hold across .github/workflows/")


if __name__ == "__main__":
    try:
        main()
    except PrismaBinaryCacheError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

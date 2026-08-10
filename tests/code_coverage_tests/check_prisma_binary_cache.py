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
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import yaml

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


def iter_jobs(workflow: object) -> Iterator[tuple[str, dict]]:
    jobs: Final = workflow.get("jobs") if isinstance(workflow, dict) else None
    if not isinstance(jobs, dict):
        return
    yield from ((name, job) for name, job in jobs.items() if isinstance(job, dict))


def job_steps(job: dict) -> tuple[dict, ...]:
    steps: Final = job.get("steps")
    return tuple(s for s in steps if isinstance(s, dict)) if isinstance(steps, list) else ()


def step_generates_prisma_client(step: dict) -> bool:
    run: Final = step.get("run")
    return isinstance(run, str) and any(m in run for m in PRISMA_GENERATE_MARKERS)


def step_restores_cache(step: dict) -> bool:
    return step.get("uses") == CACHE_ACTION


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

    for job_name, job in iter_jobs(yaml.safe_load(text)):
        steps: Final = job_steps(job)
        if any(map(step_generates_prisma_client, steps)) and not any(
            map(step_restores_cache, steps)
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

"""Validate that every collected e2e pytest declares coverage metadata.

Run from tests/e2e:

    PYTHONPATH=. python -m coverage_registry.check_coverage_sync

This is the CI guardrail for coverage drift. A new pytest fails this check until
it declares module, endpoint, provider, and params via @pytest.mark.e2e_coverage.
"""

from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

from .collector import E2E_DIR, collect_coverage_markers, compute_coverage

IGNORED_SUITE_DIRS = frozenset({".pytest_cache", "__pycache__"})


def suite_dirs_missing_tests(e2e_dir: Path = E2E_DIR) -> tuple[Path, ...]:
    """Return top-level suite dirs that contain no pytest tests."""
    return tuple(
        sorted(
            path
            for path in e2e_dir.iterdir()
            if path.is_dir()
            and path.name not in IGNORED_SUITE_DIRS
            and not any(path.rglob("test_*.py"))
        )
    )


def sync_errors(e2e_dir: Path = E2E_DIR) -> tuple[str, ...]:
    """Return human-readable sync errors for CI output."""
    report = compute_coverage(collect_coverage_markers(e2e_dir))

    errors: list[str] = []
    if report.collection_errors:
        errors.append(
            "Pytest collection errors while reading coverage markers:\n  "
            + "\n  ".join(report.collection_errors)
        )
    if report.invalid_markers:
        errors.append(
            "Invalid @pytest.mark.e2e_coverage metadata:\n  "
            + "\n  ".join(report.invalid_markers)
        )
    if report.unmarked_nodeids:
        errors.append(
            "Collected e2e tests missing @pytest.mark.e2e_coverage. Add module, "
            "endpoint, provider, and params fields, for example:\n\n"
            "  @pytest.mark.e2e_coverage(\n"
            '      module="core_llms",\n'
            '      endpoint="/chat/completions",\n'
            '      provider="openai",\n'
            '      params=["tools"],\n'
            "  )\n\n"
            "Unmapped tests:\n  " + "\n  ".join(report.unmarked_nodeids)
        )
    return tuple(errors)


def main() -> int:
    parser = ArgumentParser()
    parser.parse_args()

    errors = sync_errors()
    if errors:
        print("E2E coverage metadata is out of sync:\n", file=sys.stderr)
        print("\n\n".join(errors), file=sys.stderr)
        return 1

    print("E2E coverage metadata is in sync.")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())

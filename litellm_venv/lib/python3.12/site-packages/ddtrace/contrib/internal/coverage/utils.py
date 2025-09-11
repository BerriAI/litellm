import os
import sys
from typing import List

from ddtrace.contrib.internal.coverage.data import _original_sys_argv_command


def is_coverage_loaded() -> bool:
    return "coverage" in sys.modules


def _is_coverage_patched():
    if not is_coverage_loaded():
        return False

    return getattr(sys.modules["coverage"], "_datadog_patch", False)


def _command_invokes_coverage_run(sys_argv_command: List[str]) -> bool:
    return "coverage run -m" in " ".join(sys_argv_command)


def _is_coverage_invoked_by_coverage_run() -> bool:
    if os.environ.get("COVERAGE_RUN", False):
        return True
    return _command_invokes_coverage_run(_original_sys_argv_command)

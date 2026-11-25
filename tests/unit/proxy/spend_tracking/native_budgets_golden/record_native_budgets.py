from __future__ import annotations

import json
import sys
from typing import Final

import pytest

from tests.unit.proxy.spend_tracking.native_budgets_golden import golden

SUITES: Final = {
    golden.NATIVE_BUDGETS: "tests/unit/proxy/spend_tracking/test_native_budgets.py",
    golden.NATIVE_BUDGETS_PROXY_PATH: "tests/unit/proxy/spend_tracking/test_native_budgets_proxy_path.py",
}


def main() -> None:
    if sys.argv[1:] != ["--check"]:
        raise SystemExit("usage: record_native_budgets.py --check")
    golden.RECORDINGS.update({path: {} for path in SUITES})
    exit_code: Final = pytest.main(["-q", "-p", "no:cacheprovider", *SUITES.values()])
    if exit_code != 0:
        raise SystemExit(f"the native suites failed while checking: {exit_code}")
    differing: Final = [
        path.name for path in SUITES if json.loads(path.read_text())["python"] != golden.RECORDINGS[path]
    ]
    if differing:
        raise SystemExit(f"the native run differs from {differing}")


if __name__ == "__main__":
    main()

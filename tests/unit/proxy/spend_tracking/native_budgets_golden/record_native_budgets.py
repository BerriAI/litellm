from __future__ import annotations

import json
import subprocess
import sys
from typing import Final

import pytest

from tests.unit.proxy.spend_tracking.native_budgets_golden import golden

SUITES: Final = {
    golden.NATIVE_BUDGETS: "tests/unit/proxy/spend_tracking/test_native_budgets.py",
    golden.NATIVE_BUDGETS_PROXY_PATH: "tests/unit/proxy/spend_tracking/test_native_budgets_proxy_path.py",
}


def main() -> None:
    check_only: Final = sys.argv[1:] == ["--check"]
    golden.RECORDINGS.update({path: {} for path in SUITES})
    exit_code: Final = pytest.main(["-q", "-p", "no:cacheprovider", *SUITES.values()])
    if exit_code != 0:
        raise SystemExit(f"the differential suites failed while recording: {exit_code}")
    if check_only:
        differing: Final = [
            path.name for path in SUITES if json.loads(path.read_text())["python"] != golden.RECORDINGS[path]
        ]
        if differing:
            raise SystemExit(f"a second recording differs from {differing}")
        return
    commit: Final = subprocess.run(
        ("git", "rev-parse", "HEAD"), capture_output=True, text=True, check=True
    ).stdout.strip()
    for path in SUITES:
        _ = path.write_text(
            json.dumps({"recorded_from": commit, "python": golden.RECORDINGS[path]}, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()

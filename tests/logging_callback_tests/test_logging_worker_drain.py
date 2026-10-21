import os
from pathlib import Path
from typing import Final

from tests.test_litellm_rust.support.child_interpreter import run_child_interpreter

CANARY_MODULE: Final = Path(__file__).with_name("logging_worker_drain_canary.py")
CANARY_RUN: Final = (
    "import pytest\n"
    f"raise SystemExit(pytest.main([{str(CANARY_MODULE)!r}, '-p', 'no:xdist', '-p', 'no:cacheprovider', '-q']))\n"
)


def test_drain_fixture_runs_pending_events_before_the_next_test_starts() -> None:
    env_without_xdist: Final = {key: value for key, value in os.environ.items() if not key.startswith("PYTEST_XDIST")}
    result: Final = run_child_interpreter(CANARY_RUN, env=env_without_xdist, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr

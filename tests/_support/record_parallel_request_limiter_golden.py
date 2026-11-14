import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

from tests._support.parallel_request_limiter_golden import GOLDEN_PATH, GOLDEN_RECORD_ENV

SUITE: Final = "tests/unit/proxy/hooks/test_parallel_request_limiter_v3_native.py"
REPOSITORY: Final = Path(__file__).resolve().parents[2]


def _record(target: Path) -> dict[str, object]:
    subprocess.run(
        [sys.executable, "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", SUITE],
        env={**os.environ, "PYTHONPATH": str(REPOSITORY), GOLDEN_RECORD_ENV: str(target)},
        cwd=REPOSITORY,
        check=True,
    )
    recorded: Final = json.loads(target.read_text())
    assert isinstance(recorded, dict)
    return recorded


def main() -> int:
    if shutil.which("redis-server") is None:
        print("redis-server is required to record the redis cases", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as scratch:
        first: Final = _record(Path(scratch) / "first.json")
        second: Final = _record(Path(scratch) / "second.json")
        if first != second:
            differing = sorted(case for case in first.keys() | second.keys() if first.get(case) != second.get(case))
            print("the recording is not deterministic:", *differing, sep="\n", file=sys.stderr)
            return 1
        shutil.copyfile(Path(scratch) / "first.json", GOLDEN_PATH)
    print(f"recorded {len(first)} cases into {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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


def _record(target: Path, selection: tuple[str, ...]) -> dict[str, object]:
    subprocess.run(
        [sys.executable, "-P", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", SUITE, *selection],
        env={**os.environ, "PYTHONPATH": str(REPOSITORY), GOLDEN_RECORD_ENV: str(target)},
        cwd=REPOSITORY,
        check=True,
    )
    recorded: Final = json.loads(target.read_text())
    assert isinstance(recorded, dict)
    return recorded


def main(selection: tuple[str, ...]) -> int:
    if shutil.which("redis-server") is None:
        print("redis-server is required to record the redis cases", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as scratch:
        first: Final = _record(Path(scratch) / "first.json", selection)
        second: Final = _record(Path(scratch) / "second.json", selection)
        if first != second:
            differing = sorted(case for case in first.keys() | second.keys() if first.get(case) != second.get(case))
            print("the recording is not deterministic:", *differing, sep="\n", file=sys.stderr)
            return 1
        kept: Final = json.loads(GOLDEN_PATH.read_text()) if selection else {}
        merged: Final = {**kept, **first}
        lines: Final = (
            f"{json.dumps(case)}: {json.dumps(points, sort_keys=True, separators=(',', ':'))}"
            for case, points in sorted(merged.items())
        )
        GOLDEN_PATH.write_text("{\n" + ",\n".join(lines) + "\n}\n")
        print(f"recorded {len(first)} of {len(merged)} cases into {GOLDEN_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(tuple(sys.argv[1:])))

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SUITE = Path(__file__).resolve().parents[1] / "test_s3_v3_native.py"

if __name__ == "__main__":
    sys.exit(
        subprocess.call(
            [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:xdist", str(SUITE)],
            cwd=ROOT,
            env={**os.environ, "S3_V2_RECORD_GOLDEN": "1"},
        )
    )

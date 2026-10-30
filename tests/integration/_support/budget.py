import json
import os
import subprocess
import sys
import time
from functools import cache
from pathlib import Path
from typing import Final

IMPORT_SECONDS_VARIABLE: Final = "INTEGRATION_IMPORT_SECONDS"
READINESS_OVERRIDE_VARIABLE: Final = "INTEGRATION_PROXY_READY_SECONDS"
READINESS_IMPORTS: Final = 12
CALL_IMPORTS: Final = 6
PROXY_IMPORT: Final = "import litellm.proxy.proxy_server"


def measure_import_seconds(root: Path, environment: dict[str, str]) -> float:
    started: Final = time.monotonic()
    subprocess.run(
        (sys.executable, "-P", "-c", PROXY_IMPORT), cwd=root, env=environment, capture_output=True, check=True
    )
    return time.monotonic() - started


def boot_units(workers: int) -> int:
    return 1 if workers <= 1 else 2


def readiness_for(imported: float, workers: int) -> float:
    return READINESS_IMPORTS * imported * boot_units(workers)


def case_timeout_for(imported: float) -> float:
    return readiness_for(imported, 2) + readiness_for(imported, 1) + CALL_IMPORTS * imported


def calibration(imported: float) -> str:
    return json.dumps(
        {
            "import_seconds": imported,
            "readiness_seconds_1_worker": readiness_for(imported, 1),
            "readiness_seconds_2_workers": readiness_for(imported, 2),
            "case_timeout_seconds": case_timeout_for(imported),
        },
        indent=2,
    )


@cache
def import_seconds() -> float:
    exported: Final = os.environ.get(IMPORT_SECONDS_VARIABLE)
    if exported:
        return float(exported)
    root: Final = Path(os.environ.get("INTEGRATION_PROXY_ROOT") or Path(__file__).resolve().parents[3])
    return measure_import_seconds(root, dict(os.environ))


def readiness_seconds(workers: int = 1) -> float:
    override: Final = os.environ.get(READINESS_OVERRIDE_VARIABLE)
    return float(override) if override else readiness_for(import_seconds(), workers)


def case_timeout_seconds() -> float:
    return case_timeout_for(import_seconds())

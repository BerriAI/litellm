#!/usr/bin/env python3
"""Fail unless mypy reads the Router constructor stub in litellm/router.py through `from litellm import Router`.

litellm ships py.typed, so a stub that mypy reads as Any accepts every argument name in SDK users' code,
while basedpyright, which `make lint-e2e-basedpyright` runs, can still read it.
"""

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Final

PROBE: Final = """\
from litellm import Router

Router(model_list=[], num_retries=2)
Router(model_list=[], redis_host="localhost", redis_port="6379")
Router(model_list=[], not_a_parameter=1)
"""
EXPECTED_ERROR: Final = 'Unexpected keyword argument "not_a_parameter" for "Router"'


def mypy_errors(probe: Path) -> tuple[str, ...]:
    completed: Final = subprocess.run(
        (
            sys.executable,
            "-I",
            "-m",
            "mypy",
            "--config-file=",
            "--follow-imports=skip",
            "--ignore-missing-imports",
            "litellm/__init__.py",
            "litellm/router.py",
            str(probe),
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    return tuple(line for line in completed.stdout.splitlines() if ": error: " in line)


def main() -> int:
    with tempfile.TemporaryDirectory() as directory:
        probe: Final = Path(directory) / "router_constructor_probe.py"
        probe.write_text(PROBE)
        errors: Final = mypy_errors(probe)
    probe_errors: Final = tuple(error for error in errors if Path(error.partition(":")[0]).name == probe.name)
    if len(probe_errors) == 1 and EXPECTED_ERROR in probe_errors[0]:
        print("mypy reads the Router constructor stub")
        return 0
    print(f"expected only {EXPECTED_ERROR!r} from the probe, got: {errors}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

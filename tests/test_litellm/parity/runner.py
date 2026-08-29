from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import ValidationError

from tests.test_litellm.parity.models import Execution, SDKReport
from tests.test_litellm.parity.replay import ReplayServer


@dataclass(frozen=True, slots=True)
class PythonScriptRunner:
    entrypoint: Path
    rust_env_var: str
    python_user_agent: str

    def command(self, case_file: Path, route: str, provider_url: str, report_file: Path) -> tuple[str, ...]:
        return (
            sys.executable,
            str(self.entrypoint.resolve()),
            str(case_file),
            route,
            provider_url,
            str(report_file),
        )


def run_execution(
    runner: PythonScriptRunner,
    case_file: Path,
    route: str,
    report_file: Path,
    provider: ReplayServer,
    rust_enabled: bool,
) -> Execution:
    project_root: Final = str(runner.entrypoint.resolve().parents[3])
    existing_pythonpath: Final = os.environ.get("PYTHONPATH")
    env: Final = {
        **os.environ,
        runner.rust_env_var: "1" if rust_enabled else "0",
        "LITELLM_USER_AGENT": runner.python_user_agent,
        "PYTHONPATH": os.pathsep.join(path for path in (project_root, existing_pythonpath) if path),
    }
    mode: Final = "Rust" if rust_enabled else "Python"
    command: Final = runner.command(case_file, route, provider.url, report_file)
    try:
        completed: Final = subprocess.run(
            command,
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise AssertionError(f"{mode} OCR subprocess timed out after {error.timeout}s: {' '.join(command)}") from error
    if completed.returncode != 0:
        raise AssertionError(
            f"{mode} OCR subprocess failed with exit code {completed.returncode}\n"
            f"command: {' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if not report_file.is_file():
        raise AssertionError(f"{mode} OCR subprocess succeeded without writing report {report_file}")
    try:
        report: Final = SDKReport.model_validate_json(report_file.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as error:
        raise AssertionError(f"{mode} OCR subprocess wrote an invalid report at {report_file}: {error}") from error
    return Execution(request=provider.take_request(), report=report)

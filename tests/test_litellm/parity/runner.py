from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from tests.test_litellm.parity.models import Execution, SDKReport
from tests.test_litellm.parity.replay import JsonReplayServer


@dataclass(frozen=True, slots=True)
class PythonScriptRunner:
    entrypoint: Path
    rust_env_var: str
    python_user_agent: str

    def command(self, case_file: Path, provider_url: str, report_file: Path) -> tuple[str, ...]:
        return (
            sys.executable,
            str(self.entrypoint.resolve()),
            str(case_file),
            provider_url,
            str(report_file),
        )


def run_execution(
    runner: PythonScriptRunner,
    case_file: Path,
    report_file: Path,
    provider: JsonReplayServer,
    rust_enabled: bool,
) -> Execution:
    env: Final = {
        **os.environ,
        runner.rust_env_var: "1" if rust_enabled else "0",
        "LITELLM_USER_AGENT": runner.python_user_agent,
    }
    completed: Final = subprocess.run(
        runner.command(case_file, provider.url, report_file),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (
        f"SDK subprocess failed with exit code {completed.returncode}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    report: Final = SDKReport.model_validate_json(report_file.read_text(encoding="utf-8"))
    return Execution(report=report, requests=provider.take_requests())

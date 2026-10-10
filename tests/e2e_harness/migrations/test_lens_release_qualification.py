from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Final

import pytest

HARNESS: Final = Path(__file__).resolve().parents[2] / "e2e/migrations/lens_release_qualification.sh"


@pytest.mark.parametrize("chart", ["litellm", "litellm-helm"])
def test_inference_reaches_gateway_and_management_reaches_backend(tmp_path: Path, chart: str) -> None:
    curl: Final = tmp_path / "curl"
    curl.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "${@: -1}" >> "$qa_dir/requests"\nprintf 200\n')
    curl.chmod(0o755)
    forward: Final = tmp_path / "forward"
    forward.write_text('#!/usr/bin/env bash\nprintf "%s %s %s\\n" "$@" >> "$qa_dir/forwards"\n')
    forward.chmod(0o755)
    result: Final = subprocess.run(
        [
            "bash",
            "-euo",
            "pipefail",
            "-c",
            'source "$HARNESS"\n'
            "qualification_forward_control\n"
            "qualification_initialize\n"
            "qualification_request fixture POST /v1/chat/completions 200 inference\n"
            "qualification_request fixture GET /spend/logs 200 billing\n"
            "qualification_request fixture POST /model/new 200 management\n",
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "HARNESS": str(HARNESS),
            "qa_dir": str(tmp_path),
            "chart": chart,
            "qualification_mode": "release",
            "control": "lens-backend" if chart == "litellm" else "lens",
            "control_port": "4001" if chart == "litellm" else "4000",
            "LENS_QUALIFICATION_RESULTS_DIR": "",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    inference_port: Final = 14420 if chart == "litellm" else 14418
    assert (tmp_path / "requests").read_text().splitlines() == [
        f"http://127.0.0.1:{inference_port}/v1/chat/completions",
        "http://127.0.0.1:14418/spend/logs",
        "http://127.0.0.1:14418/model/new",
    ]
    expected_forwards: Final = (
        ["lens-backend 14418 4001", "lens-gateway 14420 4000"] if chart == "litellm" else ["lens 14418 4000"]
    )
    assert (tmp_path / "forwards").read_text().splitlines() == expected_forwards

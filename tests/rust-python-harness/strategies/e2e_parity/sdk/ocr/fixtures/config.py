from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Final

FIXTURE_DIR_ENV: Final = "LITELLM_OCR_FIXTURE_DIR"
DEFAULT_FIXTURE_DIRECTORY: Final = Path(__file__).with_name("data")


def read_gcloud(arguments: tuple[str, ...]) -> str:
    try:
        result: Final = subprocess.run(("gcloud", *arguments), capture_output=True, text=True, timeout=45, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def recording_environment(
    environ: Mapping[str, str],
    command_reader: Callable[[tuple[str, ...]], str] = read_gcloud,
) -> Mapping[str, str]:
    project: Final = (
        environ.get("VERTEXAI_PROJECT")
        or environ.get("VERTEX_PROJECT")
        or command_reader(("config", "get-value", "project"))
    )
    if not project or project == "(unset)":
        return environ
    token: Final = (
        environ.get("VERTEX_AI_ACCESS_TOKEN")
        or environ.get("VERTEX_AI_API_KEY")
        or command_reader(("auth", "print-access-token"))
    )
    if not token:
        raise SystemExit("Vertex OCR needs OAuth credentials. Run `gcloud auth login` or set VERTEX_AI_ACCESS_TOKEN")
    return {**environ, "VERTEXAI_PROJECT": project, "VERTEX_AI_API_KEY": token}


def configured_fixture_directory() -> Path:
    configured: Final = os.environ.get(FIXTURE_DIR_ENV)
    return Path(configured).expanduser() if configured is not None else DEFAULT_FIXTURE_DIRECTORY

import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

import pytest


@pytest.mark.parametrize("script", ["build_ui.sh", "build_ui_custom_path.sh", "build_release_ui.sh"])
@pytest.mark.parametrize("failure", [None, "build", "copy"])
def test_ui_build_stages_assets_without_committing(tmp_path: Path, script: str, failure: str | None) -> None:
    source: Final = Path(__file__).resolve().parents[2] / "ui/litellm-dashboard"
    dashboard: Final = tmp_path / "ui/litellm-dashboard"
    dashboard.mkdir(parents=True)
    for name in ("build_ui.sh", "build_ui_custom_path.sh", "build_release_ui.sh"):
        shutil.copy(source / name, dashboard / name)
    commands: Final = tmp_path / "bin"
    commands.mkdir()
    programs: Final = {
        "nvm": "exit 0",
        "npm": "exit 43" if failure == "build" else "mkdir -p out; printf dashboard > out/index.html; printf hidden > out/.asset",
        "git": f"touch '{tmp_path / 'git-called'}'; exit 99",
        **({"cp": "exit 42"} if failure == "copy" else {}),
    }
    for name, body in programs.items():
        command: Final = commands / name
        command.write_text(f"#!/bin/sh\n{body}\n")
        command.chmod(0o755)
    args: Final = ["/bin/bash", script, *(["/custom"] if script == "build_ui_custom_path.sh" else [])]
    result: Final = subprocess.run(
        args, cwd=dashboard, env={**os.environ, "PATH": f"{commands}:{os.environ['PATH']}"},
        capture_output=True, text=True, check=False,
    )
    assert not (tmp_path / "git-called").exists()
    if failure is not None:
        assert result.returncode != 0
        if failure == "copy":
            assert (dashboard / "out/index.html").read_text() == "dashboard"
        assert "Deployment completed" not in result.stdout
    else:
        assert result.returncode == 0, result.stderr
        destination: Final = tmp_path / "litellm-proxy-extras/litellm_proxy_extras/ui"
        assert (destination / "index.html").read_text() == "dashboard"
        assert (destination / ".asset").read_text() == "hidden"

import os
import shlex
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Final

import pytest

from litellm.proxy.client.cli.commands import claude_settings
from litellm.proxy.client.cli.commands.agents import run_agent

REAL_CLAUDE_SETTINGS: Final = Path(os.path.expanduser("~")) / ".claude" / "settings.json"
LAUNCH_PROXY: Final = "http://localhost:4000"


def _current_bytes() -> bytes | None:
    return REAL_CLAUDE_SETTINGS.read_bytes() if REAL_CLAUDE_SETTINGS.exists() else None


@pytest.fixture(autouse=True)
def _statusline_script_under_tmp(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(claude_settings, "STATUSLINE_SCRIPT_PATH", tmp_path / "litellm-home" / "statusline.py")


@pytest.fixture
def fake_codex_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[[str | None, int], Path]:
    directory: Final = tmp_path / "codex-bin"
    directory.mkdir()
    binary: Final = directory / ("codex.cmd" if os.name == "nt" else "codex")
    version_output: Final = directory / "version-output.txt"
    monkeypatch.setenv("PATH", str(directory))

    def install(output: str | None, returncode: int = 0) -> Path:
        if output is None:
            binary.unlink(missing_ok=True)
            return binary
        version_output.write_text(output)
        if os.name == "nt":
            binary.write_text(
                '@echo off\nif not "%~1"=="--version" exit /b 2\n'
                'if not "%~2"=="" exit /b 2\ntype "%~dp0version-output.txt"\n'
                f'exit /b {returncode}\n'
            )
        else:
            binary.write_text(
                '#!/bin/sh\nif [ "$#" -ne 1 ] || [ "$1" != "--version" ]; then\n    exit 2\nfi\n'
                f'/bin/cat {shlex.quote(str(version_output))}\nexit {returncode}\n'
            )
        binary.chmod(0o700)
        return binary

    install("codex-cli 0.129.0\n")
    return install


@pytest.fixture(autouse=True)
def _isolated_codex_version_for_configure_tests(request: pytest.FixtureRequest) -> None:
    if request.node.path.name in ("test_codex_settings.py", "test_configure_commands.py"):
        request.getfixturevalue("fake_codex_version")


@pytest.fixture(autouse=True)
def isolated_claude_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    before: Final = _current_bytes()
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / ".codex"))
    yield tmp_path
    after: Final = _current_bytes()
    if after == before:
        return
    if before is None:
        REAL_CLAUDE_SETTINGS.unlink()
    else:
        REAL_CLAUDE_SETTINGS.write_bytes(before)
    pytest.fail(
        f"this test wrote the developer's real {REAL_CLAUDE_SETTINGS}; the original bytes were restored. "
        "Resolve the Claude settings path at call time (never Path.home() at import) and point the test at tmp_path"
    )


@pytest.fixture
def launch_claude(isolated_claude_home: Path) -> Callable[[bool], Path]:
    """Launch `lite claude` over a `lite login --pkce --config-claude` setup and return the `--settings` file it
    wrote, which holds the launch key. Claude Code's config dir is read at call time, so a test can move it first."""

    def launch(subscription: bool) -> Path:
        launched: Final[dict[str, tuple[str, ...]]] = {}
        run_agent(
            LAUNCH_PROXY,
            "sk-launch",
            ["claude"],
            subscription=subscription,
            base_env={"CLAUDE_CONFIG_DIR": os.environ["CLAUDE_CONFIG_DIR"]},
            which=lambda name: f"/usr/local/bin/{name}",
            verify=lambda *args: None,
            verify_subscription=lambda *args: None,
            launcher=lambda path, args, env: launched.update(args=tuple(args)),
            read_settings=lambda path: {
                "apiKeyHelper": f"/usr/local/bin/lite --base-url {LAUNCH_PROXY} auth print-token",
                "env": {"ANTHROPIC_AUTH_TOKEN": ""},
            },
            project_dir=isolated_claude_home / "repo",
        )
        args: Final = launched["args"]
        return Path(args[args.index("--settings") + 1])

    return launch

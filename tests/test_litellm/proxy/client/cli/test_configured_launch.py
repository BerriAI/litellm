import os
import subprocess
import sys
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

from litellm.proxy.client.cli.commands.agents import AgentRunError
from litellm.proxy.client.cli.commands.configured_launch import launch_configured_agents


def test_single_agent_hands_off_with_only_its_persisted_config(tmp_path):
    calls = []
    restored = []
    binary = tmp_path / "bin with spaces" / "claude"
    launch_configured_agents(
        ("claude",),
        started_interactive=True,
        environ={
            "PATH": "/usr/bin",
            "CLAUDE_CONFIG_DIR": "/config with spaces/claude",
            "CODEX_HOME": "/config/codex",
            "ANTHROPIC_AUTH_TOKEN": "secret",
            "ANTHROPIC_MODEL": "wrong-model",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "wrong-tier",
            "OPENAI_API_KEY": "secret-two",
        },
        which=lambda command: str(binary) if command == "claude" else None,
        hand_off=lambda path, argv, env: calls.append((path, argv, env)),
        restore_terminal=lambda: restored.append(True),
    )
    assert restored == [True]
    assert calls == [
        (
            str(binary.resolve()),
            ("claude",),
            {"PATH": "/usr/bin", "CLAUDE_CONFIG_DIR": "/config with spaces/claude"},
        )
    ]


def test_two_agents_open_separate_macos_terminals_without_secrets(tmp_path):
    calls = []
    binaries = {
        "claude": str(tmp_path / "claude binary"),
        "codex": str(tmp_path / "codex;binary"),
        "osascript": str(tmp_path / "osascript"),
    }
    environ = {
        "PATH": "/usr/bin",
        "CLAUDE_CONFIG_DIR": "/claude profile",
        "CODEX_HOME": "/codex;profile",
        "ANTHROPIC_AUTH_TOKEN": "secret-a",
        "OPENAI_API_KEY": "secret-b",
    }

    def run(argv, **kwargs):
        calls.append((argv, kwargs))

    launch_configured_agents(
        ("claude", "codex"),
        started_interactive=True,
        environ=environ,
        platform="darwin",
        cwd="/project; touch /tmp/nope",
        which=binaries.get,
        run=run,
    )

    assert len(calls) == 2
    assert all(call[0][0] == str(Path(binaries["osascript"]).resolve()) for call in calls)
    assert all(
        call[0][1:3] == ("-e", 'on run argv\ntell application "Terminal" to do script (item 1 of argv)\nend run')
        for call in calls
    )
    assert "secret" not in repr(calls)
    assert "'" in calls[0][0][3] and "/project; touch /tmp/nope" in calls[0][0][3]
    assert calls[0][1]["env"] == {"PATH": "/usr/bin", "CLAUDE_CONFIG_DIR": "/claude profile"}
    assert calls[1][1]["env"] == {"PATH": "/usr/bin", "CODEX_HOME": "/codex;profile"}


def test_two_agents_use_linux_terminal_argv(tmp_path):
    calls = []
    binaries = {
        "claude": str(tmp_path / "claude"),
        "codex": str(tmp_path / "codex"),
        "x-terminal-emulator": str(tmp_path / "terminal"),
    }
    launch_configured_agents(
        ("claude", "codex"),
        started_interactive=False,
        environ={"PATH": "/usr/bin"},
        platform="linux",
        cwd="/work",
        which=binaries.get,
        popen=lambda argv, **kwargs: calls.append((argv, kwargs)),
    )
    assert [call[0][:4] for call in calls] == [
        (str(Path(binaries["x-terminal-emulator"]).resolve()), "-e", "/bin/sh", "-lc"),
        (str(Path(binaries["x-terminal-emulator"]).resolve()), "-e", "/bin/sh", "-lc"),
    ]


@pytest.mark.parametrize("suffix", ["exe", "cmd", "bat"])
def test_two_windows_agents_use_new_consoles_and_the_shared_command_builder(suffix):
    calls = []
    binaries = {
        "claude": rf"C:\Program Files\100% & !name!\claude.{suffix}",
        "codex": rf"C:\Users\me\tools\codex.{suffix}",
    }
    workdir = r"C:\work & %PATH% !name!"
    launch_configured_agents(
        ("claude", "codex"),
        started_interactive=True,
        environ={"PATH": r"C:\Windows\System32", "ANTHROPIC_MODEL": "stale", "OPENAI_API_KEY": "secret"},
        platform="win32",
        cwd=workdir,
        which=binaries.get,
        popen=lambda command, **kwargs: calls.append((command, kwargs)),
    )
    assert len(calls) == 2
    for agent, (command, kwargs) in zip(("claude", "codex"), calls):
        assert kwargs == {
            "cwd": workdir,
            "env": {"PATH": r"C:\Windows\System32"},
            "creationflags": 0x00000010,
        }
        if suffix == "exe":
            assert command == (binaries[agent],)
        else:
            assert isinstance(command, str)
            assert command.startswith('cmd.exe /d /e:on /v:off /s /c "')
            assert command.endswith('""')
            if agent == "claude":
                assert "100%%cd:~,% & !name!" in command
        assert "start" not in command


@pytest.mark.parametrize("newline", ["\n", "\r"])
def test_windows_shim_path_cannot_inject_a_second_command(newline):
    calls = []
    with pytest.raises(click.ClickException, match="line break"):
        launch_configured_agents(
            ("claude", "codex"),
            started_interactive=True,
            platform="win32",
            which=lambda agent: f"C:\\tools\\{agent}{newline}injected.cmd",
            popen=lambda *args, **kwargs: calls.append(args),
        )
    assert calls == []


@pytest.mark.skipif(sys.platform != "win32", reason="Requires Windows CreateProcess and cmd.exe")
@pytest.mark.parametrize("suffix", ["cmd", "bat"])
def test_windows_new_console_executes_real_clients(tmp_path, suffix):
    workdir = tmp_path / "work & %PATH% !name!"
    workdir.mkdir()
    binaries = {}
    for agent in ("claude", "codex"):
        directory = tmp_path / agent / "bin with spaces & %PATH% !name!"
        directory.mkdir(parents=True)
        binary = directory / f"{agent}.{suffix}"
        binary.write_text('@echo off\nif defined ANTHROPIC_MODEL exit /b 1\ncd > "%~dp0launched.txt"\n')
        binaries[agent] = str(binary)
    processes = []

    def spawn(command, **kwargs):
        process = subprocess.Popen(command, **kwargs)
        processes.append(process)
        return process

    launch_configured_agents(
        ("claude", "codex"),
        started_interactive=True,
        environ={**os.environ, "ANTHROPIC_MODEL": "must-not-leak"},
        cwd=str(workdir),
        which=binaries.get,
        popen=spawn,
    )
    for process in processes:
        assert process.wait(timeout=20) == 0
    for binary in binaries.values():
        assert Path(binary).with_name("launched.txt").read_text().strip() == str(workdir)


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
@pytest.mark.parametrize("stage", ["which", "restore", "handoff", "first-terminal", "second-terminal"])
@pytest.mark.parametrize(
    "error",
    [PermissionError("launch denied"), AgentRunError("invalid launch"), subprocess.CalledProcessError(1, "launcher")],
    ids=["os-error", "agent-error", "process-error"],
)
def test_launch_errors_preserve_configs_and_report_recovery(tmp_path, platform, stage, error):
    settings = tmp_path / "settings.json"
    config = tmp_path / "config.toml"
    settings.write_text('{"env":{"ANTHROPIC_BASE_URL":"http://gateway.test"}}')
    config.write_text('model_provider = "litellm"\n')
    original = settings.read_bytes(), config.read_bytes()
    selected = ("claude", "codex") if stage.endswith("terminal") else ("claude",)
    calls = []

    def fail():
        raise error

    def which(name):
        if stage == "which":
            fail()
        return str(tmp_path / name)

    def restore():
        if stage == "restore":
            fail()

    def handoff(*args):
        calls.append("handoff")
        if stage == "handoff":
            fail()

    def spawn(*args, **kwargs):
        calls.append("terminal")
        if stage == "first-terminal" or (stage == "second-terminal" and len(calls) == 2):
            fail()

    @click.command()
    def launch():
        launch_configured_agents(
            selected,
            started_interactive=True,
            platform=platform,
            cwd=str(tmp_path),
            environ={},
            which=which,
            restore_terminal=restore,
            hand_off=handoff,
            run=spawn,
            popen=spawn,
        )

    result = CliRunner().invoke(launch)
    failed_agent = "codex" if stage == "second-terminal" else "claude"
    assert result.exit_code == 1
    assert f"Error: Could not open `{failed_agent}`." in result.output
    assert f"Configuration was saved; start `{failed_agent}` manually." in result.output
    assert "Traceback" not in result.output
    assert (settings.read_bytes(), config.read_bytes()) == original
    assert len(calls) == (2 if stage.endswith("terminal") else 0 if stage in ("which", "restore") else 1)


@pytest.mark.parametrize("error", [SystemExit(7), KeyboardInterrupt(), TypeError("programming error")])
def test_launch_does_not_relabel_exit_interrupt_or_programming_errors(tmp_path, error):
    def handoff(*args):
        raise error

    with pytest.raises(type(error), match=r"7|programming error|^$"):
        launch_configured_agents(
            ("claude",), started_interactive=False, which=lambda _: str(tmp_path / "claude"), hand_off=handoff
        )


def test_missing_terminal_keeps_configuration_and_names_manual_fallback(tmp_path):
    binaries = {"claude": str(tmp_path / "claude"), "codex": str(tmp_path / "codex")}
    with pytest.raises(click.ClickException, match="Configuration was saved; start `claude` manually"):
        launch_configured_agents(
            ("claude", "codex"),
            started_interactive=True,
            platform="linux",
            which=binaries.get,
        )

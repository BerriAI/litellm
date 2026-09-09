import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import click

from .agents import AgentRunError, hand_off, restore_controlling_terminal, windows_command

_CREATE_NEW_CONSOLE: Final = 0x00000010

_ROUTING_ENV: Final = frozenset(
    {
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_BASE_URL",
        "ANTHROPIC_DEFAULT_FABLE_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_MODEL",
        "LITELLM_PROXY_API_KEY",
        "LITELLM_PROXY_URL",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
    }
)
_CONFIG_ENV: Final = {"claude": "CLAUDE_CONFIG_DIR", "codex": "CODEX_HOME"}


@contextmanager
def _launch_errors(agents: Sequence[str]) -> Generator[None, None, None]:
    try:
        yield
    except (AgentRunError, OSError, subprocess.SubprocessError) as e:
        commands: Final = " and ".join(f"`{agent}`" for agent in agents)
        raise click.ClickException(
            f"Could not open {commands}. Configuration was saved; start {commands} manually. {e}"
        ) from e


def _agent_env(agent: str, environ: Mapping[str, str]) -> dict[str, str]:
    config_key: Final = _CONFIG_ENV[agent]
    return {
        key: value
        for key, value in environ.items()
        if key not in _ROUTING_ENV and (key not in _CONFIG_ENV.values() or key == config_key)
    }


def _shell_command(agent: str, binary: str, cwd: str, environ: Mapping[str, str], quote: Callable[[str], str]) -> str:
    config_key: Final = _CONFIG_ENV[agent]
    config: Final = environ.get(config_key)
    assignment: Final = () if config is None else (f"{config_key}={quote(config)}",)
    unsets: Final = tuple(part for key in sorted(_ROUTING_ENV) for part in ("-u", key))
    command: Final = " ".join(("env", *unsets, *assignment, quote(binary)))
    return f"cd -- {quote(cwd)} && exec {command}"


def _launch_terminal(
    agent: str,
    binary: str,
    cwd: str,
    environ: Mapping[str, str],
    platform: str,
    which: Callable[[str], str | None],
    run: Callable[..., subprocess.CompletedProcess[str]],
    popen: Callable[..., subprocess.Popen[bytes]],
) -> None:
    env: Final = _agent_env(agent, environ)
    if platform.startswith("win"):
        popen(windows_command(binary, (agent,)), cwd=cwd, env=env, creationflags=_CREATE_NEW_CONSOLE)
        return
    command: Final = _shell_command(agent, binary, cwd, environ, shlex.quote)
    if platform == "darwin":
        osascript: Final = which("osascript")
        if osascript is None:
            raise AgentRunError("Terminal is unavailable.")
        script: Final = 'on run argv\ntell application "Terminal" to do script (item 1 of argv)\nend run'
        run((str(Path(osascript).resolve()), "-e", script, command), env=env, check=True)
        return
    terminal: Final = which("x-terminal-emulator")
    if terminal is None:
        raise AgentRunError("No terminal emulator is available.")
    popen((str(Path(terminal).resolve()), "-e", "/bin/sh", "-lc", command), env=env)


def launch_configured_agents(
    agents: Sequence[str],
    *,
    started_interactive: bool,
    environ: Mapping[str, str] = os.environ,
    platform: str = sys.platform,
    cwd: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    hand_off: Callable[[str, Sequence[str], Mapping[str, str]], None] = hand_off,
    restore_terminal: Callable[[], None] = restore_controlling_terminal,
    run: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    popen: Callable[..., subprocess.Popen[bytes]] = subprocess.Popen,
) -> None:
    selected: Final = tuple(dict.fromkeys(agents))
    if not selected or any(agent not in _CONFIG_ENV for agent in selected):
        raise click.ClickException("Choose Claude Code, Codex, or both.")
    with _launch_errors(selected):
        binaries: Final = tuple((agent, which(agent)) for agent in selected)
        missing: Final = tuple(agent for agent, binary in binaries if binary is None)
        if missing:
            raise AgentRunError(f"Could not find {', '.join(missing)} on PATH.")
        resolved: Final = tuple(
            (agent, str(Path(binary).resolve()) if not platform.startswith("win") else binary)
            for agent, binary in binaries
            if binary is not None
        )
        launch_cwd: Final = cwd or os.getcwd()
    errors: Final[list[str]] = []  # mutable-ok: accumulate independent launch failures before reporting them together
    for agent, binary in resolved:
        try:
            with _launch_errors((agent,)):
                if len(resolved) == 1:
                    if started_interactive:
                        restore_terminal()
                    hand_off(binary, (agent,), _agent_env(agent, environ))
                else:
                    _launch_terminal(agent, binary, launch_cwd, environ, platform, which, run, popen)
        except click.ClickException as e:
            errors.append(e.message)
    if errors:
        raise click.ClickException("\n".join(errors))

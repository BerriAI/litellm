"""`lite alias claude`: make the bare `claude` command run `lite claude`.

`lite claude` routes one invocation through the proxy, but customers wanted the
plain `claude` they already type to do it: no new prefix, no settings patching.
`lite alias <agent>` installs a tiny shim named after the agent into
~/.litellm/bin and puts that directory first on PATH via the shell's rc file, so
`claude` launches `lite claude` (same login, key check, and env wiring) while
the real Claude Code still does the work. `lite unalias <agent>` removes it.

Forwarding cannot loop: the shim marks its directory with ALIAS_SHIM_DIR_ENV,
and the agent launch resolves the *real* binary past that directory
(which_ignoring_alias_dir in agents.py), giving claude -> lite claude ->
real claude.
"""

# stdlib imports
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Final

# third party imports
import click

# local imports
from .agents import (
    _HIDDEN_AGENTS,
    _INSTALL_DOCS,
    _KNOWN_AGENTS,
    ALIAS_SHIM_DIR_ENV,
    which_ignoring_alias_dir,
)

SHIM_DIR_NAME: Final = ".litellm/bin"
SHIM_MARKER_TEMPLATE: Final = "installed by `lite alias {agent}`"
_RC_MARKER: Final = "# added by `lite alias` (LiteLLM coding-agent shims)"
_RC_PRESENT_TOKEN: Final = "/.litellm/bin"


class AliasError(Exception):
    """Raised for any user-actionable failure while (un)installing an agent alias."""


def default_shim_dir() -> Path:
    """~/.litellm/bin, resolved at call time (never Path.home() at import)."""
    return Path.home() / ".litellm" / "bin"


def shim_filename(agent: str, platform: str) -> str:
    """The shim's file name: `agent` everywhere but Windows, which needs an extension."""
    return f"{agent}.cmd" if platform.startswith("win") else agent


def shim_content(agent: str, shim_dir: Path, lite_bin: str, *, platform: str) -> str:
    """The shim script: forward every argument to `lite <agent>` as a marked launch.

    lite_bin is the already-resolved `lite` entry point when one was found, so the
    shim survives environments whose PATH lost the install directory; it falls
    back to a bare `lite` when running from source.
    """
    marker: Final = SHIM_MARKER_TEMPLATE.format(agent=agent)
    if platform.startswith("win"):
        return "\r\n".join(
            (
                "@echo off",
                f"rem litellm coding-agent shim: {marker}; remove with `lite unalias {agent}`.",
                f'set "{ALIAS_SHIM_DIR_ENV}={shim_dir}"',
                f'"{lite_bin}" {agent} %*',
                "",
            )
        )
    return "\n".join(
        (
            "#!/bin/sh",
            f"# litellm coding-agent shim: {marker}; remove with `lite unalias {agent}`.",
            f'{ALIAS_SHIM_DIR_ENV}="{shim_dir}" exec {lite_bin} {agent} "$@"',
            "",
        )
    )


def is_our_shim(path: Path, agent: str) -> bool:
    """True when path holds a shim `lite alias agent` wrote (marker line present)."""
    try:
        content: Final = path.read_text(encoding="utf-8")
    except OSError:
        return False
    return SHIM_MARKER_TEMPLATE.format(agent=agent) in content


def install_shim(
    agent: str,
    shim_dir: Path,
    *,
    platform: str = sys.platform,
    lite_bin: str | None = None,
    which: Callable[[str], str | None] = shutil.which,
    force: bool = False,
) -> tuple[Path, bool]:
    """Write the shim for agent into shim_dir; return (path, overwrote_our_own).

    Raises AliasError when the target already holds a file `lite alias` did not
    write, unless force is set. The shim is executable on POSIX; on Windows the
    `.cmd` extension already makes it runnable.
    """
    path: Final = shim_dir / shim_filename(agent, platform)
    overwrote = path.exists()
    if overwrote and not force and not is_our_shim(path, agent):
        raise AliasError(
            f"Refusing to overwrite {path}: it already exists and was not installed by "
            f"`lite alias {agent}`. Move it aside, or pass --force to replace it."
        )
    resolved: Final = lite_bin if lite_bin is not None else (which("lite") or "lite")
    shim_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(shim_content(agent, shim_dir, resolved, platform=platform), encoding="utf-8")
    if not platform.startswith("win"):
        path.chmod(0o755)
    return path, overwrote


def remove_shim(agent: str, shim_dir: Path, *, platform: str = sys.platform) -> Path:
    """Delete the shim for agent; return its path.

    Raises AliasError when there is no shim, or when the file at the shim path
    was not written by `lite alias agent`.
    """
    path: Final = shim_dir / shim_filename(agent, platform)
    if not path.exists():
        raise AliasError(f"`{agent}` is not aliased: no shim at {path}.")
    if not is_our_shim(path, agent):
        raise AliasError(f"Refusing to remove {path}: it was not installed by `lite alias {agent}`.")
    path.unlink()
    return path


def rc_block(shell: str) -> str:
    """The rc snippet that puts ~/.litellm/bin first on PATH for that shell."""
    if os.path.basename(shell) == "fish":
        return f"{_RC_MARKER}\nfish_add_path -g $HOME/{SHIM_DIR_NAME}\n"
    return f'{_RC_MARKER}\nexport PATH="$HOME/{SHIM_DIR_NAME}:$PATH"\n'


def rc_candidates(home: Path, shell: str) -> tuple[tuple[Path, str], ...]:
    """(rc file, snippet) pairs for the login shell; bash's .bashrc by default.

    Only the interactive rc of the shell in $SHELL is touched: the alias targets
    typing `claude` at a prompt, not scripts, which keep their own PATH.
    """
    name: Final = os.path.basename(shell)
    if name == "zsh":
        return ((home / ".zshrc", rc_block(shell)),)
    if name == "fish":
        return ((home / ".config" / "fish" / "config.fish", rc_block(shell)),)
    return ((home / ".bashrc", rc_block(shell)),)


def ensure_rc_lines(home: Path, shell: str) -> tuple[Path, ...]:
    """Append the PATH snippet to the rc file when missing; return files written.

    A file that already mentions the shim directory is left alone, so a second
    `lite alias <other agent>` never duplicates the line.
    """
    written: list[Path] = []
    for path, block in rc_candidates(home, shell):
        existing: str = path.read_text(encoding="utf-8") if path.exists() else ""
        if _RC_PRESENT_TOKEN in existing:
            continue
        if existing and not existing.endswith("\n"):
            existing += "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(existing + block, encoding="utf-8")
        written.append(path)
    return tuple(written)


def find_real_binary(agent: str, shim_dir: Path) -> str | None:
    """Resolve the real agent binary, looking past our own shim directory."""
    saved: Final = os.environ.get(ALIAS_SHIM_DIR_ENV)
    os.environ[ALIAS_SHIM_DIR_ENV] = str(shim_dir)
    try:
        return which_ignoring_alias_dir(agent)
    finally:
        if saved is None:
            os.environ.pop(ALIAS_SHIM_DIR_ENV, None)
        else:
            os.environ[ALIAS_SHIM_DIR_ENV] = saved


_RESTART_HINT: Final = f'Restart your shell (or run `export PATH="$HOME/{SHIM_DIR_NAME}:$PATH"`)'


def _install_alias(agent: str, display_name: str, force: bool) -> None:
    shim_dir: Final = default_shim_dir()
    real: Final = find_real_binary(agent, shim_dir)
    if real is None:
        docs: Final = _INSTALL_DOCS.get(agent)
        hint: Final = f" Install it first: {docs}." if docs else ""
        raise click.ClickException(f"Could not find `{agent}` on your PATH, so there is nothing to alias.{hint}")

    try:
        path, overwrote = install_shim(agent, shim_dir, force=force)
    except AliasError as e:
        raise click.ClickException(str(e)) from e

    relaunch: Final = f"`{agent}` will run `lite {agent}` (routed through your LiteLLM proxy)"
    if sys.platform.startswith("win"):
        click.echo(f"Installed {path}; {display_name} found at {real}.")
        click.echo(f"Add {shim_dir} to your user PATH (ahead of that install) and {relaunch}.")
        return

    written: Final = ensure_rc_lines(Path.home(), os.environ.get("SHELL", ""))
    if overwrote:
        click.echo(f"Refreshed {path} ({display_name} found at {real}).")
    else:
        click.echo(f"Installed {path} ({display_name} found at {real}).")
    for rc in written:
        click.echo(f"Added the shim directory to PATH in {rc}.")
    current: Final = shutil.which(agent)
    if current is not None and Path(current) == path:
        click.echo(f"Already active in this shell: {relaunch}.")
    else:
        click.echo(f"{_RESTART_HINT} and {relaunch}.")


def _remove_alias(agent: str) -> None:
    shim_dir: Final = default_shim_dir()
    try:
        removed: Final = remove_shim(agent, shim_dir)
    except AliasError as e:
        raise click.ClickException(str(e)) from e
    click.echo(f"Removed {removed}; `{agent}` now runs whatever is next on your PATH.")
    click.echo(
        "The PATH line `lite alias` may have added to your shell rc is harmless once the "
        f"directory is empty; delete the line mentioning {SHIM_DIR_NAME} to drop it."
    )


def _make_alias_command(agent: str, display_name: str) -> click.Command:
    @click.command(name=agent)
    @click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Replace a file at the shim path that `lite alias` did not write",
    )
    def _command(force: bool) -> None:
        _install_alias(agent, display_name, force)

    _command.short_help = f"Make plain `{agent}` run `lite {agent}`"
    _command.help = (
        f"Install a shim so typing `{agent}` launches `lite {agent}` through your LiteLLM proxy.\n\n"
        f"Writes a small `{agent}` shim into ~/{SHIM_DIR_NAME} and puts that directory first on "
        f"PATH via your shell rc, so the bare `{agent}` you already type gets `lite {agent}`'s "
        f"login, key check, and env wiring: no prefix, no settings patching, arguments forwarded "
        f"untouched. Remove it with `lite unalias {agent}`."
    )
    return _command


def _make_unalias_command(agent: str) -> click.Command:
    @click.command(name=agent)
    def _command() -> None:
        _remove_alias(agent)

    _command.short_help = f"Restore plain `{agent}` (remove the `lite alias {agent}` shim)"
    _command.help = (
        f"Remove the `{agent}` shim `lite alias {agent}` installed from ~/{SHIM_DIR_NAME}.\n\n"
        f"After this, `{agent}` runs the real binary next on your PATH again."
    )
    return _command


@click.group()
def alias() -> None:
    """Make a bare coding-agent command (e.g. `claude`) run through `lite`"""


@click.group()
def unalias() -> None:
    """Remove an agent alias installed by `lite alias`"""


def alias_commands() -> tuple[click.Command, ...]:
    """Build `lite alias <agent>` for each public known agent (claude, codex, opencode)."""
    return tuple(
        _make_alias_command(agent, name)
        for agent, (name, _profiles) in _KNOWN_AGENTS.items()
        if agent not in _HIDDEN_AGENTS
    )


def unalias_commands() -> tuple[click.Command, ...]:
    """Build `lite unalias <agent>` for each public known agent."""
    return tuple(
        _make_unalias_command(agent)
        for agent, (name, _profiles) in _KNOWN_AGENTS.items()
        if agent not in _HIDDEN_AGENTS
    )


__all__ = [
    "AliasError",
    "alias_commands",
    "default_shim_dir",
    "ensure_rc_lines",
    "find_real_binary",
    "install_shim",
    "is_our_shim",
    "rc_candidates",
    "remove_shim",
    "shim_content",
    "shim_filename",
    "unalias_commands",
]

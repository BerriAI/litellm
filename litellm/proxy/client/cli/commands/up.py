import atexit
import contextlib
import json
import os
import shlex
import shutil
import signal
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import IO, Iterator, Mapping

import click
from pydantic import JsonValue, TypeAdapter

from litellm.litellm_core_utils.cli_token_utils import is_cli_token_fresh

from .agents import AgentRunError, resolve_api_key, verify_proxy_key
from .auth import load_token, login

ENV_KEY = "env"
API_KEY_HELPER_KEY = "apiKeyHelper"
ANTHROPIC_BASE_URL_KEY = "ANTHROPIC_BASE_URL"
ANTHROPIC_API_KEY_KEY = "ANTHROPIC_API_KEY"

CLAUDE_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
BACKUP_PATH = Path.home() / ".litellm" / "claude_settings_backup.json"


class UpError(Exception):
    """Raised for any user-actionable failure while starting/stopping interception."""


@dataclass(frozen=True, slots=True)
class BackupRecord:
    """Snapshot of ~/.claude/settings.json taken right before `lite up` patches it."""

    existed: bool
    content: dict[str, JsonValue] | None


_SETTINGS_ADAPTER = TypeAdapter(dict[str, JsonValue])
_BACKUP_RECORD_ADAPTER = TypeAdapter(BackupRecord)


def load_json_or_empty(path: Path) -> dict[str, JsonValue]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        content = f.read()
    if not content.strip():
        return {}
    return _SETTINGS_ADAPTER.validate_json(content)


def merge_claude_settings(
    settings: Mapping[str, JsonValue], base_url: str, api_key_helper: str
) -> dict[str, JsonValue]:
    """Return a new settings dict wired to route Claude Code through the proxy.

    Only env.ANTHROPIC_BASE_URL and the top-level apiKeyHelper are overridden; a
    stray env.ANTHROPIC_API_KEY is dropped so it cannot outrank the helper-issued
    token (same reasoning as build_agent_env in agents.py). Every other key is
    preserved untouched.
    """
    raw_env = settings.get(ENV_KEY, {})
    base_env = raw_env if isinstance(raw_env, dict) else {}
    env = {**base_env, ANTHROPIC_BASE_URL_KEY: base_url.rstrip("/")}
    env.pop(ANTHROPIC_API_KEY_KEY, None)
    return {**settings, ENV_KEY: env, API_KEY_HELPER_KEY: api_key_helper}


@contextlib.contextmanager
def secure_create(path: Path) -> Iterator[IO[str]]:
    """Open path for writing with mode 0600 fixed up before any content is written.

    A plain `open(path, "w")` creates a *new* file at the umask-derived default (commonly 0644)
    and leaves it world- or group-readable until a later `chmod` call catches up -- a real window
    in which a file holding a credential is readable by another local account. Passing the mode to
    `os.open` closes that window for a brand-new file, but `O_CREAT`'s mode argument is only
    applied on creation: if the file already exists its old, broader permissions carry over
    untouched. `os.fchmod` right after opening -- before a single byte of the new content is
    written -- covers both cases.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    f: IO[str] = os.fdopen(fd, "w")
    try:
        yield f
    finally:
        f.close()


def write_backup(record: BackupRecord, backup_path: Path | None = None) -> None:
    path = backup_path if backup_path is not None else BACKUP_PATH
    path.parent.mkdir(exist_ok=True)
    with secure_create(path) as f:
        json.dump({"existed": record.existed, "content": record.content}, f, indent=2)


def read_backup(backup_path: Path | None = None) -> BackupRecord | None:
    path = backup_path if backup_path is not None else BACKUP_PATH
    if not path.exists():
        return None
    with open(path, "r") as f:
        return _BACKUP_RECORD_ADAPTER.validate_json(f.read())


def restore_claude_settings(settings_path: Path | None = None, backup_path: Path | None = None) -> BackupRecord | None:
    """Restore settings_path from the backup at backup_path, then delete the backup.

    Returns the restored record, or None if there was nothing to restore.
    """
    resolved_settings_path = settings_path if settings_path is not None else CLAUDE_SETTINGS_PATH
    resolved_backup_path = backup_path if backup_path is not None else BACKUP_PATH
    record = read_backup(resolved_backup_path)
    if record is None:
        return None
    if record.existed and record.content is not None:
        with open(resolved_settings_path, "w") as f:
            json.dump(record.content, f, indent=2)
    elif resolved_settings_path.exists():
        resolved_settings_path.unlink()
    resolved_backup_path.unlink()
    return record


def resolve_api_key_helper(base_url: str) -> str:
    """Build the shell command Claude Code should run for its apiKeyHelper.

    Resolves `lite` to an absolute path so the helper works regardless of the
    PATH visible to whatever subprocess Claude Code spawns it from. Passing
    --base-url explicitly (rather than relying on the bare invocation Claude
    Code would otherwise use) makes `print-token` enforce that the cached
    token was actually issued for this proxy -- without it, a token minted
    for a different, previously-logged-into proxy would be handed to
    whichever server `up` currently points at.
    """
    lite_path = shutil.which("lite")
    if lite_path is None:
        raise UpError(
            "Could not find `lite` on your PATH. Claude Code's apiKeyHelper needs "
            "an absolute path to it, so `lite up` cannot continue."
        )
    return f"{shlex.quote(lite_path)} auth print-token --base-url {shlex.quote(base_url)}"


def _ensure_fresh_login(ctx: click.Context) -> None:
    base_url = ctx.obj["base_url"].rstrip("/")
    token_data = load_token()
    if token_data and token_data.get("base_url") == base_url and is_cli_token_fresh(token_data):
        return

    if not sys.stdin.isatty():
        raise UpError(
            "No fresh LiteLLM login found for this proxy. Run `lite login` first (apiKeyHelper "
            "reads this token on every Claude Code request)."
        )

    click.echo("No fresh LiteLLM login found for this proxy; starting login...")
    ctx.invoke(login)
    token_data = load_token()
    if not token_data or token_data.get("base_url") != base_url or not is_cli_token_fresh(token_data):
        raise UpError("Login did not produce a usable token; cannot start `lite up`.")


def _restore_and_report() -> None:
    record = restore_claude_settings()
    if record is None:
        click.echo("Nothing to restore.")
        return
    if record.existed:
        click.echo(f"Restored {CLAUDE_SETTINGS_PATH} to its original contents.")
    else:
        click.echo(f"Removed {CLAUDE_SETTINGS_PATH} (it did not exist before `lite up`).")


@click.command(name="up")
@click.pass_context
def up(ctx: click.Context) -> None:
    """Route every Claude Code session through your LiteLLM proxy until stopped.

    Patches ~/.claude/settings.json so Claude Code picks up the proxy on its own
    next startup, from any terminal -- no need to launch it through `lite`.
    Press Ctrl-C to stop and restore your original settings. Assumes the proxy
    is already running (this does not start one for you). Cursor is not
    supported: it has no equivalent file-based config to patch.
    """
    base_url = ctx.obj["base_url"]

    try:
        _ensure_fresh_login(ctx)
        api_key = resolve_api_key(ctx)
        verify_proxy_key(base_url, api_key)

        if BACKUP_PATH.exists():
            raise UpError(
                f"{BACKUP_PATH} already exists -- `lite up` looks like it's already "
                "running (or crashed without cleanup). Run `lite down` first."
            )

        api_key_helper = resolve_api_key_helper(base_url)
        original_existed = CLAUDE_SETTINGS_PATH.exists()
        original_settings = load_json_or_empty(CLAUDE_SETTINGS_PATH)
        write_backup(
            BackupRecord(
                existed=original_existed,
                content=original_settings if original_existed else None,
            )
        )

        CLAUDE_SETTINGS_PATH.parent.mkdir(exist_ok=True)
        merged = merge_claude_settings(original_settings, base_url, api_key_helper)
        with open(CLAUDE_SETTINGS_PATH, "w") as f:
            json.dump(merged, f, indent=2)
    except (AgentRunError, UpError) as e:
        raise click.ClickException(str(e))

    click.echo(f"litellm: routing Claude Code through proxy at {base_url.rstrip('/')}")
    click.echo("Press Ctrl-C to stop and restore your original settings.")

    stop_event = threading.Event()
    restored = threading.Lock()

    def _handle_signal(_signum: int, _frame: FrameType | None) -> None:
        stop_event.set()

    def _restore_once() -> None:
        if restored.acquire(blocking=False):
            _restore_and_report()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    atexit.register(_restore_once)

    stop_event.wait()
    _restore_once()


@click.command(name="down")
def down() -> None:
    """Restore ~/.claude/settings.json if a `lite up` session left it patched.

    Use this after a `lite up` process was killed uncleanly (e.g. `kill -9`)
    instead of stopped with Ctrl-C.
    """
    _restore_and_report()


__all__ = [
    "BACKUP_PATH",
    "CLAUDE_SETTINGS_PATH",
    "BackupRecord",
    "UpError",
    "down",
    "load_json_or_empty",
    "merge_claude_settings",
    "read_backup",
    "resolve_api_key_helper",
    "restore_claude_settings",
    "up",
    "write_backup",
]

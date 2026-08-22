import atexit
import contextlib
import json
import os
import signal
import sys
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import FrameType
from typing import IO, Final

import click
from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.cli_keyring import SecretVault
from litellm.litellm_core_utils.cli_token_utils import is_cli_token_fresh
from litellm.litellm_core_utils.private_json import ensure_private_dir

from .agents import AgentRunError, resolve_api_key, verify_proxy_key
from .auth import CliContextObj, context_secret_vault, get_stored_api_key, load_token, login
from .claude_settings import (
    BACKUP_PATH,
    CLAUDE_SETTINGS_PATH,
    ClaudeSettingsError,
    load_json_or_empty,
    merge_claude_settings,
    resolve_api_key_helper,
)


class UpError(ClaudeSettingsError):
    """Raised for any user-actionable failure while starting/stopping interception."""


@dataclass(frozen=True, slots=True)
class BackupRecord:
    """Snapshot of ~/.claude/settings.json taken right before `lite up` patches it."""

    existed: bool
    content: dict[str, JsonValue] | None


_BACKUP_RECORD_ADAPTER: Final = TypeAdapter(BackupRecord)


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
    fd: Final = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    f: Final[IO[str]] = os.fdopen(fd, "w")
    try:
        yield f
    finally:
        f.close()


def write_backup(record: BackupRecord, backup_path: Path | None = None) -> None:
    path: Final = backup_path if backup_path is not None else BACKUP_PATH
    ensure_private_dir(path.parent)
    with secure_create(path) as f:
        json.dump({"existed": record.existed, "content": record.content}, f, indent=2)


def read_backup(backup_path: Path | None = None) -> BackupRecord | None:
    path: Final = backup_path if backup_path is not None else BACKUP_PATH
    if not path.exists():
        return None
    with open(path, "r") as f:
        content: Final = f.read()
    try:
        return _BACKUP_RECORD_ADAPTER.validate_json(content)
    except ValidationError:
        raise UpError(f"{path} contains invalid or unexpected JSON; cannot restore from it safely.")


def restore_claude_settings(settings_path: Path | None = None, backup_path: Path | None = None) -> BackupRecord | None:
    """Restore settings_path from the backup at backup_path, then delete the backup.

    Returns the restored record, or None if there was nothing to restore.
    """
    resolved_settings_path: Final = settings_path if settings_path is not None else CLAUDE_SETTINGS_PATH
    resolved_backup_path: Final = backup_path if backup_path is not None else BACKUP_PATH
    record: Final = read_backup(resolved_backup_path)
    if record is None:
        return None
    if record.existed and record.content is not None:
        resolved_settings_path.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved_settings_path, "w") as f:
            json.dump(record.content, f, indent=2)
    elif resolved_settings_path.exists():
        resolved_settings_path.unlink()
    resolved_backup_path.unlink()
    return record


def _usable_login(api_key: str | None, vault: SecretVault) -> bool:
    if api_key is None:
        return False
    token_data: Final = load_token(vault=vault)
    return token_data is not None and is_cli_token_fresh(token_data)


def _key_resolved_on_the_way_in(ctx_obj: CliContextObj, base_url: str, vault: SecretVault) -> str | None:
    if ctx_obj.get("api_key_from_token_file"):
        return ctx_obj.get("api_key")
    return get_stored_api_key(expected_base_url=base_url, vault=vault)


def _stored_login_is_pkce(vault: SecretVault) -> bool:
    token_data: Final = load_token(vault=vault)
    return token_data is not None and token_data.get("refresh_token") is not None


def _ensure_fresh_login(ctx: click.Context) -> None:
    ctx_obj: Final[CliContextObj] = ctx.obj
    base_url: Final = ctx_obj["base_url"].rstrip("/")
    vault: Final = context_secret_vault(ctx)
    if _usable_login(_key_resolved_on_the_way_in(ctx_obj, base_url, vault), vault):
        return

    pkce: Final = _stored_login_is_pkce(vault)
    login_command: Final = "lite login --pkce" if pkce else "lite login"
    if not sys.stdin.isatty():
        raise UpError(
            f"No fresh LiteLLM login found for this proxy. Run `{login_command}` first (apiKeyHelper "
            "reads this token on every Claude Code request)."
        )

    click.echo("No fresh LiteLLM login found for this proxy; starting login...")
    ctx.invoke(login, pkce=pkce)
    if not _usable_login(get_stored_api_key(expected_base_url=base_url, vault=vault), vault):
        raise UpError("Login did not produce a usable token; cannot start `lite up`.")


def _restore_and_report() -> None:
    record: Final = restore_claude_settings()
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
    base_url: Final = ctx.obj["base_url"]

    try:
        _ensure_fresh_login(ctx)
        api_key: Final = resolve_api_key(ctx)
        verify_proxy_key(base_url, api_key)

        if BACKUP_PATH.exists():
            raise UpError(
                f"{BACKUP_PATH} already exists -- `lite up` looks like it's already "
                "running (or crashed without cleanup). Run `lite down` first."
            )

        api_key_helper: Final = resolve_api_key_helper(base_url)
        original_existed: Final = CLAUDE_SETTINGS_PATH.exists()
        original_settings: Final = load_json_or_empty(CLAUDE_SETTINGS_PATH)
        write_backup(
            BackupRecord(
                existed=original_existed,
                content=original_settings if original_existed else None,
            )
        )

        CLAUDE_SETTINGS_PATH.parent.mkdir(exist_ok=True)
        merged: Final = merge_claude_settings(original_settings, base_url, api_key_helper)
        with open(CLAUDE_SETTINGS_PATH, "w") as f:
            json.dump(merged, f, indent=2)
    except (AgentRunError, ClaudeSettingsError) as e:
        raise click.ClickException(str(e))

    click.echo(f"litellm: routing Claude Code through proxy at {base_url.rstrip('/')}")
    click.echo("Press Ctrl-C to stop and restore your original settings.")

    stop_event: Final = threading.Event()
    restored: Final = threading.Lock()

    def _handle_signal(_signum: int, _frame: FrameType | None) -> None:
        stop_event.set()

    def _restore_once() -> None:
        if not restored.acquire(blocking=False):
            return
        try:
            _restore_and_report()
        except ClaudeSettingsError as e:
            # Runs from atexit/a signal handler, outside Click's own exception
            # handling -- raising here would only produce an unhandled-exception
            # warning on stderr, not a clean message.
            click.echo(str(e), err=True)

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
    try:
        _restore_and_report()
    except ClaudeSettingsError as e:
        raise click.ClickException(str(e))


__all__ = [
    "BACKUP_PATH",
    "CLAUDE_SETTINGS_PATH",
    "BackupRecord",
    "ClaudeSettingsError",
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

import atexit
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
from typing import Dict, Mapping, Optional

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
    content: Optional[Dict[str, JsonValue]]


_SETTINGS_ADAPTER = TypeAdapter(Dict[str, JsonValue])
_BACKUP_RECORD_ADAPTER = TypeAdapter(BackupRecord)


def load_json_or_empty(path: Path) -> Dict[str, JsonValue]:
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return _SETTINGS_ADAPTER.validate_json(f.read())


def merge_claude_settings(
    settings: Mapping[str, JsonValue], base_url: str, api_key_helper: str
) -> Dict[str, JsonValue]:
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


def write_backup(record: BackupRecord) -> None:
    BACKUP_PATH.parent.mkdir(exist_ok=True)
    with open(BACKUP_PATH, "w") as f:
        json.dump({"existed": record.existed, "content": record.content}, f, indent=2)
    os.chmod(BACKUP_PATH, 0o600)


def read_backup() -> Optional[BackupRecord]:
    if not BACKUP_PATH.exists():
        return None
    with open(BACKUP_PATH, "r") as f:
        return _BACKUP_RECORD_ADAPTER.validate_json(f.read())


def restore_claude_settings() -> Optional[BackupRecord]:
    """Restore ~/.claude/settings.json from the backup, then delete the backup.

    Returns the restored record, or None if there was nothing to restore.
    """
    record = read_backup()
    if record is None:
        return None
    if record.existed and record.content is not None:
        with open(CLAUDE_SETTINGS_PATH, "w") as f:
            json.dump(record.content, f, indent=2)
    elif CLAUDE_SETTINGS_PATH.exists():
        CLAUDE_SETTINGS_PATH.unlink()
    BACKUP_PATH.unlink()
    return record


def resolve_api_key_helper() -> str:
    """Build the shell command Claude Code should run for its apiKeyHelper.

    Resolves `lite` to an absolute path so the helper works regardless of the
    PATH visible to whatever subprocess Claude Code spawns it from.
    """
    lite_path = shutil.which("lite")
    if lite_path is None:
        raise UpError(
            "Could not find `lite` on your PATH. Claude Code's apiKeyHelper needs "
            "an absolute path to it, so `lite up` cannot continue."
        )
    return f"{shlex.quote(lite_path)} auth print-token"


def _ensure_fresh_login(ctx: click.Context) -> None:
    token_data = load_token()
    if token_data and is_cli_token_fresh(token_data):
        return

    if not sys.stdin.isatty():
        raise UpError(
            "No fresh LiteLLM login found. Run `lite login` first (apiKeyHelper "
            "reads this token on every Claude Code request)."
        )

    click.echo("No fresh LiteLLM login found; starting login...")
    ctx.invoke(login)
    token_data = load_token()
    if not token_data or not is_cli_token_fresh(token_data):
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

        api_key_helper = resolve_api_key_helper()
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

    def _handle_signal(_signum: int, _frame: Optional[FrameType]) -> None:
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
    "up",
    "down",
    "BackupRecord",
    "load_json_or_empty",
    "merge_claude_settings",
    "write_backup",
    "read_backup",
    "restore_claude_settings",
    "resolve_api_key_helper",
    "UpError",
    "CLAUDE_SETTINGS_PATH",
    "BACKUP_PATH",
]

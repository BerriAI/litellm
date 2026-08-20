"""Shared handling of Claude Code's ~/.claude/settings.json.

`lite up` patches this file temporarily and restores it on exit; `lite login
--config-claude` patches it persistently. Both need the same merge and the same
apiKeyHelper command, and `up` already imports from `auth`, so the shared parts
live here rather than in either command module.
"""

import shlex
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.private_json import write_private_json

ENV_KEY: Final = "env"
API_KEY_HELPER_KEY: Final = "apiKeyHelper"
ANTHROPIC_BASE_URL_KEY: Final = "ANTHROPIC_BASE_URL"
ANTHROPIC_API_KEY_KEY: Final = "ANTHROPIC_API_KEY"

CLAUDE_SETTINGS_PATH: Final = Path.home() / ".claude" / "settings.json"
BACKUP_PATH: Final = Path.home() / ".litellm" / "claude_settings_backup.json"
AUTOROUTE_BACKUP_PATH: Final = Path.home() / ".litellm" / "autorouter" / "claude_settings_backup.json"


@dataclass(frozen=True, slots=True)
class SettingsFileOwner:
    """A command that takes temporary ownership of CLAUDE_SETTINGS_PATH and restores it later."""

    backup_path: Path
    start_command: str
    stop_command: str


SETTINGS_FILE_OWNERS: Final = (
    SettingsFileOwner(BACKUP_PATH, "lite up", "lite down"),
    SettingsFileOwner(AUTOROUTE_BACKUP_PATH, "lite autoroute up", "lite autoroute down"),
)

_SETTINGS_ADAPTER: Final = TypeAdapter(dict[str, JsonValue])


class ClaudeSettingsError(Exception):
    """Raised for any user-actionable failure while reading or writing Claude Code settings."""


def load_json_or_empty(path: Path) -> dict[str, JsonValue]:
    try:
        content: Final = path.read_bytes() if path.exists() else b""
    except OSError as e:
        raise ClaudeSettingsError(f"Could not read {path}: {e}") from e
    if not content.strip():
        return {}
    try:
        return _SETTINGS_ADAPTER.validate_json(content)
    except ValidationError:
        raise ClaudeSettingsError(
            f"{path} contains invalid JSON (or its root is not an object); cannot proceed safely."
        )


def merge_claude_settings(
    settings: Mapping[str, JsonValue], base_url: str, api_key_helper: str
) -> dict[str, JsonValue]:
    """Return a new settings dict wired to route Claude Code through the proxy.

    Only env.ANTHROPIC_BASE_URL and the top-level apiKeyHelper are overridden; a
    stray env.ANTHROPIC_API_KEY is dropped so it cannot outrank the helper-issued
    token (same reasoning as build_agent_env in agents.py). Every other key is
    preserved untouched.
    """
    raw_env: Final = settings.get(ENV_KEY, {})
    base_env: Final = raw_env if isinstance(raw_env, dict) else {}
    env: Final = {
        **{key: value for key, value in base_env.items() if key != ANTHROPIC_API_KEY_KEY},
        ANTHROPIC_BASE_URL_KEY: base_url.rstrip("/"),
    }
    return {**settings, ENV_KEY: env, API_KEY_HELPER_KEY: api_key_helper}


def resolve_api_key_helper(base_url: str) -> str:
    """Build the shell command Claude Code should run for its apiKeyHelper.

    Resolves `lite` to an absolute path so the helper works regardless of the
    PATH visible to whatever subprocess Claude Code spawns it from. Passing
    --base-url explicitly (rather than relying on the bare invocation Claude
    Code would otherwise use) makes `print-token` enforce that the cached
    token was actually issued for this proxy -- without it, a token minted
    for a different, previously-logged-into proxy would be handed to
    whichever server the settings currently point at.

    --base-url belongs to the top-level `lite` group, so it has to precede the
    subcommand; click rejects it outright after `print-token`.
    """
    lite_path: Final = shutil.which("lite")
    if lite_path is None:
        raise ClaudeSettingsError(
            "Could not find `lite` on your PATH. Claude Code's apiKeyHelper needs an absolute path to it."
        )
    return f"{shlex.quote(lite_path)} --base-url {shlex.quote(base_url)} auth print-token"


def write_claude_settings(base_url: str, settings_path: Path, owners: Sequence[SettingsFileOwner]) -> None:
    """Persistently point Claude Code at base_url, preserving every unrelated setting.

    Refuses while any owner holds a backup: each restores its backup when it
    stops, which would silently undo this write.
    """
    for owner in owners:
        if owner.backup_path.exists():
            raise ClaudeSettingsError(
                f"`{owner.start_command}` is currently managing {settings_path} (backup at "
                f"{owner.backup_path}) and will restore it when it stops. "
                f"Run `{owner.stop_command}` first, then retry."
            )
    normalized_base_url: Final = base_url.rstrip("/")
    api_key_helper: Final = resolve_api_key_helper(normalized_base_url)
    existing: Final = load_json_or_empty(settings_path)
    raw_env: Final = existing.get(ENV_KEY)
    if raw_env is not None and not isinstance(raw_env, dict):
        raise ClaudeSettingsError(
            f'{settings_path} has a non-object "{ENV_KEY}" value, which this would discard. '
            "Fix or remove it, then retry."
        )
    merged: Final = merge_claude_settings(existing, normalized_base_url, api_key_helper)
    # os.replace() swaps the symlink itself for a regular file, silently detaching a
    # settings.json that is symlinked into a dotfiles repo. There is no backup to undo
    # that here, unlike `lite up`, so write through to the link's target instead.
    target: Final = settings_path.resolve() if settings_path.is_symlink() else settings_path
    try:
        write_private_json(str(target), merged)
    except OSError as e:
        raise ClaudeSettingsError(f"Could not write {target}: {e}") from e


__all__ = (
    "ANTHROPIC_API_KEY_KEY",
    "ANTHROPIC_BASE_URL_KEY",
    "API_KEY_HELPER_KEY",
    "AUTOROUTE_BACKUP_PATH",
    "BACKUP_PATH",
    "CLAUDE_SETTINGS_PATH",
    "ENV_KEY",
    "SETTINGS_FILE_OWNERS",
    "ClaudeSettingsError",
    "SettingsFileOwner",
    "load_json_or_empty",
    "merge_claude_settings",
    "resolve_api_key_helper",
    "write_claude_settings",
)

"""Shared handling of Claude Code's ~/.claude/settings.json.

`lite up` and `lite autoroute up` patch this file temporarily and restore it on
exit; `lite login --config-claude` and `lite configure claude` patch it
persistently and record how to undo it. All of them need the same merge and the
same apiKeyHelper command, and `up` already imports from `auth`, so the shared
parts live here rather than in any one command module.
"""

import hashlib
import json
import shlex
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from pathlib import Path
from types import MappingProxyType
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.litellm_core_utils.private_json import (
    commit_staged_json,
    discard_staged_json,
    ensure_private_dir,
    stage_private_json,
)

from .cmd_quoting import quote_for_cmd

ENV_KEY: Final = "env"
API_KEY_HELPER_KEY: Final = "apiKeyHelper"
MODEL_KEY: Final = "model"
ANTHROPIC_BASE_URL_KEY: Final = "ANTHROPIC_BASE_URL"
ANTHROPIC_AUTH_TOKEN_KEY: Final = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_API_KEY_KEY: Final = "ANTHROPIC_API_KEY"
ENABLE_TOOL_SEARCH_KEY: Final = "ENABLE_TOOL_SEARCH"
ENABLE_TOOL_SEARCH_VALUE: Final = "true"
ENABLE_GATEWAY_MODEL_DISCOVERY_KEY: Final = "CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY"
ENABLE_GATEWAY_MODEL_DISCOVERY_VALUE: Final = "1"
ANTHROPIC_DEFAULT_MODEL_ENV_KEYS: Final = (
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "ANTHROPIC_DEFAULT_FABLE_MODEL",
)
OWNED_ENV_KEYS: Final = (
    ENABLE_TOOL_SEARCH_KEY,
    ENABLE_GATEWAY_MODEL_DISCOVERY_KEY,
    ANTHROPIC_BASE_URL_KEY,
    ANTHROPIC_AUTH_TOKEN_KEY,
    ANTHROPIC_API_KEY_KEY,
)
OWNED_TOP_LEVEL_KEYS: Final = (API_KEY_HELPER_KEY, MODEL_KEY)
OWNED_PATHS: Final = (*(f"{ENV_KEY}.{key}" for key in OWNED_ENV_KEYS), *OWNED_TOP_LEVEL_KEYS)
_CREDENTIAL_ENV_KEYS: Final = frozenset((ANTHROPIC_API_KEY_KEY, ANTHROPIC_AUTH_TOKEN_KEY))
_CREDENTIAL_PATHS: Final = (*(f"{ENV_KEY}.{key}" for key in sorted(_CREDENTIAL_ENV_KEYS)), API_KEY_HELPER_KEY)
_BASE_URL_PATH: Final = f"{ENV_KEY}.{ANTHROPIC_BASE_URL_KEY}"
STARTING_MODEL_ROLE: Final = "the /model picker's default row, the model Claude Code starts on"

CLAUDE_SETTINGS_PATH: Final = Path.home() / ".claude" / "settings.json"
CLAUDE_CONFIG_DIR_ENV: Final = "CLAUDE_CONFIG_DIR"
BACKUP_PATH: Final = Path.home() / ".litellm" / "claude_settings_backup.json"
AUTOROUTE_BACKUP_PATH: Final = Path.home() / ".litellm" / "autorouter" / "claude_settings_backup.json"
CONFIGURE_STATE_PATH: Final = Path.home() / ".litellm" / "claude_configure_state.json"


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


def claude_settings_path(environ: Mapping[str, str]) -> Path:
    """The settings.json Claude Code reads: under CLAUDE_CONFIG_DIR when set, else ~/.claude/settings.json."""
    config_dir: Final = environ.get(CLAUDE_CONFIG_DIR_ENV, "")
    if not config_dir:
        return CLAUDE_SETTINGS_PATH
    return Path(config_dir).expanduser() / "settings.json"


def _is_default_settings_file(settings_path: Path) -> bool:
    return settings_path.resolve() == CLAUDE_SETTINGS_PATH.resolve()


def settings_file_owners(settings_path: Path) -> tuple[SettingsFileOwner, ...]:
    """The commands whose backups guard settings_path: `lite up` and `lite autoroute up` only ever manage the default file."""
    return SETTINGS_FILE_OWNERS if _is_default_settings_file(settings_path) else ()


def configure_state_path(settings_path: Path) -> Path:
    """The receipt describing settings_path: the default file keeps CONFIGURE_STATE_PATH, and any other file
    (a CLAUDE_CONFIG_DIR) gets its own beside it, keyed by its resolved path, so two settings files never
    share one undo record."""
    if _is_default_settings_file(settings_path):
        return CONFIGURE_STATE_PATH
    digest: Final = hashlib.sha256(str(settings_path.resolve()).encode()).hexdigest()
    return CONFIGURE_STATE_PATH.parent / CONFIGURE_STATE_PATH.stem / f"{digest}.json"


@dataclass(frozen=True, slots=True)
class StaticToken:
    """A long-lived virtual key, written into env.ANTHROPIC_AUTH_TOKEN."""

    token: str


@dataclass(frozen=True, slots=True)
class ApiKeyHelper:
    """A `lite auth print-token` command Claude Code runs per request, so a login renews in place."""

    command: str


ClaudeCredential: TypeAlias = StaticToken | ApiKeyHelper


@dataclass(frozen=True, slots=True)
class KeepModel:
    """Leave the top-level `model` as it is, the user's or an earlier configure's (a re-login)."""


@dataclass(frozen=True, slots=True)
class UnpinModel:
    """Let go of a `model` an earlier configure pinned; one the user set themselves stays."""


@dataclass(frozen=True, slots=True)
class StartOn:
    """Pin the top-level `model`, the row Claude Code starts on."""

    model: str


ModelChoice: TypeAlias = KeepModel | UnpinModel | StartOn


class OwnedValue(BaseModel):
    """What one key held at a moment in time; `present=False` is an absent key, not a null one."""

    model_config = ConfigDict(frozen=True)

    present: bool
    value: JsonValue = None


class ConfigureReceipt(BaseModel):
    """What `lite configure claude` found and what it owns, keyed by dotted path (`env.X` or a top-level key).

    Ownership moves only by a write: `written` fingerprints the keys some configure changed, at the
    value it wrote; a repeat configure refreshes a fingerprint only for a key its merge changed and
    carries the earlier one otherwise, so a key the user edited in between stops matching and is left
    alone. `previous` is what each key held before configure took it over; a repeat keeps the earlier
    snapshot while the key still holds our value and snapshots afresh otherwise, so whatever the
    repeat displaces is what comes back. `endpoints` is the ANTHROPIC_BASE_URL each credential slot
    was captured beside, so a credential is only ever put back next to the server it was issued for.
    No fingerprint is a second copy of a token.
    """

    model_config = ConfigDict(frozen=True)

    file_existed: bool
    env_present: bool
    env_was_object: bool
    previous: Mapping[str, OwnedValue]
    written: Mapping[str, str]
    endpoints: Mapping[str, OwnedValue]


@dataclass(frozen=True, slots=True)
class WithheldCredential:
    """A credential left removed: captured beside `endpoint`, while the restored file points elsewhere."""

    key: str
    endpoint: str


@dataclass(frozen=True, slots=True)
class UnconfigureOutcome:
    """Keys whose value unconfigure changed back, keys the user changed since and so were left as they
    are, credentials withheld (the receipt is kept for them, so a later unconfigure can finish once the
    URL points back), and whether no settings file remains."""

    restored: tuple[str, ...]
    kept: tuple[str, ...]
    withheld: tuple[WithheldCredential, ...] = ()
    file_removed: bool = False


@dataclass(frozen=True, slots=True)
class _Claim:
    previous: OwnedValue
    written: str | None
    endpoint: OwnedValue | None


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


def _env_object(settings: Mapping[str, JsonValue], path: Path) -> Mapping[str, JsonValue]:
    raw_env: Final = settings.get(ENV_KEY)
    if raw_env is None:
        return MappingProxyType({})
    if not isinstance(raw_env, dict):
        raise ClaudeSettingsError(
            f'{path} has a non-object "{ENV_KEY}" value, which this would discard. Fix or remove it, then retry.'
        )
    return raw_env


def refuse_while_owned(settings_path: Path, owners: Sequence[SettingsFileOwner]) -> None:
    """Refuse while `lite up` or `lite autoroute up` holds a backup it will restore over any write; a
    purely local check, so commands run it before any login prompt or request."""
    for owner in owners:
        if owner.backup_path.exists():
            raise ClaudeSettingsError(
                f"`{owner.start_command}` is currently managing {settings_path} (backup at "
                f"{owner.backup_path}) and will restore it when it stops. "
                f"Run `{owner.stop_command}` first, then retry."
            )


def _write_target(settings_path: Path) -> Path:
    """Write through a symlinked settings.json rather than replacing the link, which would silently
    detach a file symlinked into a dotfiles repo."""
    try:
        return settings_path.resolve() if settings_path.is_symlink() else settings_path
    except OSError as e:
        raise ClaudeSettingsError(f"Could not resolve {settings_path}: {e}") from e


def _stage(path: Path, document: Mapping[str, object]) -> str:
    try:
        return stage_private_json(str(path), document)
    except OSError as e:
        raise ClaudeSettingsError(f"Could not write {path}: {e}") from e


def _land(
    path: Path,
    staged: str | None,
    also_discard: Sequence[str | None] = (),
    commit: Callable[[str, str], None] = commit_staged_json,
) -> None:
    """Commit a staged file to `path`, or remove `path` when nothing is staged for it. The one place a
    filesystem error becomes a ClaudeSettingsError; on failure the operation's other staged files are
    discarded, so no temp file holding a token is left behind."""
    try:
        if staged is None:
            path.unlink(missing_ok=True)
        else:
            commit(staged, str(path))
    except OSError as e:
        for other in also_discard:
            if other is not None:
                discard_staged_json(other)
        raise ClaudeSettingsError(f"Could not {'remove' if staged is None else 'write'} {path}: {e}") from e


def merge_claude_settings(
    settings: Mapping[str, JsonValue],
    base_url: str,
    credential: ClaudeCredential,
    default_model: str | None = None,
    tier_model: str | None = None,
) -> Mapping[str, JsonValue]:
    """Return a new settings mapping wired to route Claude Code through the proxy.

    A StaticToken lands in env.ANTHROPIC_AUTH_TOKEN, an ApiKeyHelper in the top-level apiKeyHelper;
    the other credential slots are removed either way, since Claude Code given two credentials may
    send the wrong one. ENABLE_TOOL_SEARCH and CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY get their
    defaults only when missing. `default_model` is the top-level `model`, the row Claude Code starts
    on; `tier_model` is `lite autoroute up`'s knob that points every ANTHROPIC_DEFAULT_*_MODEL at one
    group. Apart from those tier keys, exactly OWNED_PATHS are touched.
    """
    raw_env: Final = settings.get(ENV_KEY, {})
    current_env: Final = raw_env if isinstance(raw_env, dict) else {}
    env: Final = dict(  # mutable-ok: JSON document handed to json.dump, which rejects a read-only mapping
        chain(
            (
                (ENABLE_TOOL_SEARCH_KEY, ENABLE_TOOL_SEARCH_VALUE),
                (ENABLE_GATEWAY_MODEL_DISCOVERY_KEY, ENABLE_GATEWAY_MODEL_DISCOVERY_VALUE),
            ),
            ((key, value) for key, value in current_env.items() if key not in _CREDENTIAL_ENV_KEYS),
            ((ANTHROPIC_BASE_URL_KEY, base_url.rstrip("/")),),
            ((ANTHROPIC_AUTH_TOKEN_KEY, credential.token),) if isinstance(credential, StaticToken) else (),
            ((key, tier_model) for key in ANTHROPIC_DEFAULT_MODEL_ENV_KEYS if tier_model is not None),
        )
    )
    return dict(  # mutable-ok: JSON document handed to json.dump, which rejects a read-only mapping
        chain(
            ((key, value) for key, value in settings.items() if key not in (API_KEY_HELPER_KEY, ENV_KEY)),
            ((ENV_KEY, env),),
            ((API_KEY_HELPER_KEY, credential.command),) if isinstance(credential, ApiKeyHelper) else (),
            ((MODEL_KEY, default_model),) if default_model is not None else (),
        )
    )


def resolve_api_key_helper(base_url: str, platform: str = sys.platform) -> str:
    """Build the shell command Claude Code should run for its apiKeyHelper.

    Claude Code hands the string to the system shell, `sh` on POSIX and cmd.exe
    on Windows, so every token is quoted for the shell that will read it.

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
    quote: Final = quote_for_cmd if platform.startswith("win") else shlex.quote
    return " ".join(quote(token) for token in (lite_path, "--base-url", base_url, "auth", "print-token"))


def lite_api_key_helper_configured(base_url: str, settings_path: Path) -> bool:
    """Whether settings_path already carries the apiKeyHelper `lite login --config-claude` writes for base_url.

    Only an exact match counts: a helper for another proxy, a hand-written one, or
    settings that cannot be read leave the caller on the env-token path.
    """
    try:
        configured_helper: Final = load_json_or_empty(settings_path).get(API_KEY_HELPER_KEY)
        return configured_helper == resolve_api_key_helper(base_url.rstrip("/"))
    except ClaudeSettingsError:
        return False


def _owned(container: Mapping[str, JsonValue], key: str) -> OwnedValue:
    return OwnedValue(present=key in container, value=container.get(key))


def _fingerprint(owned: OwnedValue) -> str:
    return hashlib.sha256(json.dumps(owned.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def _env(settings: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    raw_env: Final = settings.get(ENV_KEY)
    return raw_env if isinstance(raw_env, dict) else MappingProxyType({})


def _lookup(settings: Mapping[str, JsonValue], path: str) -> OwnedValue:
    section, _, key = path.rpartition(".")
    return _owned(_env(settings) if section else settings, key)


def _with_key(container: Mapping[str, JsonValue], key: str, owned: OwnedValue) -> Mapping[str, JsonValue]:
    return dict(  # mutable-ok: JSON document handed to json.dump, which rejects a read-only mapping
        chain(((k, v) for k, v in container.items() if k != key), ((key, owned.value),) if owned.present else ())
    )


def _with(settings: Mapping[str, JsonValue], path: str, owned: OwnedValue) -> Mapping[str, JsonValue]:
    """`settings` with the key at `path` set (or removed when `owned` is absent); nothing else changes."""
    section, _, key = path.rpartition(".")
    if not section:
        return _with_key(settings, key, owned)
    return _with_key(settings, section, OwnedValue(present=True, value=_with_key(_env(settings), key, owned)))


def _with_all(settings: Mapping[str, JsonValue], updates: Mapping[str, OwnedValue]) -> Mapping[str, JsonValue]:
    return reduce(lambda acc, item: _with(acc, *item), updates.items(), settings)


def _ours(settings: Mapping[str, JsonValue], path: str, receipt: ConfigureReceipt) -> bool:
    """Whether the key still holds what a configure wrote (a key no configure ever changed is never ours)."""
    return receipt.written.get(path) == _fingerprint(_lookup(settings, path))


def _claim(
    path: str,
    current: Mapping[str, JsonValue],
    merged: Mapping[str, JsonValue],
    earlier: ConfigureReceipt | None,
    url_now: OwnedValue,
) -> _Claim:
    """What this configure records for one key; see ConfigureReceipt for the rules."""
    before, after = _lookup(current, path), _lookup(merged, path)
    carried: Final = earlier if earlier is not None and _ours(current, path, earlier) else None
    return _Claim(
        previous=before if carried is None else carried.previous.get(path, before),
        written=_fingerprint(after) if before != after else (None if earlier is None else earlier.written.get(path)),
        endpoint=None
        if path not in _CREDENTIAL_PATHS
        else (url_now if carried is None else carried.endpoints.get(path, url_now)),
    )


def _receipt(
    current: Mapping[str, JsonValue],
    merged: Mapping[str, JsonValue],
    earlier: ConfigureReceipt | None,
    file_exists: bool,
) -> ConfigureReceipt:
    url_now: Final = _lookup(current, _BASE_URL_PATH)
    claims: Final = MappingProxyType({path: _claim(path, current, merged, earlier, url_now) for path in OWNED_PATHS})
    return ConfigureReceipt(
        file_existed=file_exists if earlier is None else earlier.file_existed,
        env_present=ENV_KEY in current if earlier is None else earlier.env_present,
        env_was_object=isinstance(current.get(ENV_KEY), dict) if earlier is None else earlier.env_was_object,
        previous=MappingProxyType({path: claim.previous for path, claim in claims.items()}),
        written=MappingProxyType({path: claim.written for path, claim in claims.items() if claim.written is not None}),
        endpoints=MappingProxyType(
            {path: claim.endpoint for path, claim in claims.items() if claim.endpoint is not None}
        ),
    )


def read_configure_receipt(state_path: Path) -> ConfigureReceipt | None:
    if not state_path.exists():
        return None
    try:
        return ConfigureReceipt.model_validate_json(state_path.read_bytes())
    except (OSError, ValidationError) as e:
        raise ClaudeSettingsError(
            f"{state_path} is not a readable `lite configure claude` receipt ({e}). "
            "Remove it and edit Claude Code's settings by hand if they still point at the proxy."
        ) from e


def configure_claude_settings(
    base_url: str,
    credential: ClaudeCredential,
    model: ModelChoice,
    settings_path: Path,
    state_path: Path,
    owners: Sequence[SettingsFileOwner],
    commit: Callable[[str, str], None] = commit_staged_json,
) -> None:
    """Persistently route Claude Code through base_url, recording how to undo it.

    Both files are staged before either is committed, so a full disk or a read-only directory fails
    before anything changes. The two commits are still two renames: a receipt rename that fails
    discards the staged settings, and a settings rename that fails after the receipt landed puts the
    earlier receipt back (or removes the new one), so the receipt on disk never describes settings
    that were not written. `model`: StartOn pins the starting model, UnpinModel lets go of a pin an
    earlier configure made (never of the user's own), KeepModel leaves it alone (a re-login).
    """
    refuse_while_owned(settings_path, owners)
    current: Final = load_json_or_empty(settings_path)
    _env_object(current, settings_path)
    earlier: Final = read_configure_receipt(state_path)
    existing: Final = (
        _with(current, MODEL_KEY, earlier.previous[MODEL_KEY])
        if isinstance(model, UnpinModel) and earlier is not None and _ours(current, MODEL_KEY, earlier)
        else current
    )
    merged: Final = merge_claude_settings(
        existing, base_url, credential, model.model if isinstance(model, StartOn) else None
    )
    receipt: Final = _receipt(current, merged, earlier, settings_path.exists())
    target: Final = _write_target(settings_path)
    try:
        ensure_private_dir(state_path.parent)
    except OSError as e:
        raise ClaudeSettingsError(f"Could not write {state_path}: {e}") from e
    staged_receipt: Final = _stage(state_path, receipt.model_dump(mode="json"))
    try:
        staged_settings: Final = _stage(target, merged)
    except ClaudeSettingsError:
        discard_staged_json(staged_receipt)
        raise
    _land(state_path, staged_receipt, (staged_settings,), commit)
    try:
        _land(target, staged_settings, commit=commit)
    except ClaudeSettingsError as settings_error:
        try:
            _land(state_path, None if earlier is None else _stage(state_path, earlier.model_dump(mode="json")))
        except ClaudeSettingsError as receipt_error:
            raise ClaudeSettingsError(
                f"{settings_error} The receipt at {state_path} now describes settings that were not written and "
                f"could not be put back either ({receipt_error}); remove it before retrying."
            ) from settings_error
        raise


def _endpoint_text(endpoint: OwnedValue) -> str:
    if not endpoint.present:
        return f"no {ANTHROPIC_BASE_URL_KEY} (Anthropic's default endpoint)"
    return endpoint.value if isinstance(endpoint.value, str) else json.dumps(endpoint.value)


def unconfigure_claude_settings(
    settings_path: Path, state_path: Path, owners: Sequence[SettingsFileOwner]
) -> UnconfigureOutcome:
    """Undo `lite configure claude`: put back every key still holding what configure wrote, leave the
    rest alone, and withhold a credential the restored file would send to a different server than it
    was issued for (the receipt stays, owning only those slots, so a later unconfigure can finish)."""
    refuse_while_owned(settings_path, owners)
    receipt: Final = read_configure_receipt(state_path)
    if receipt is None:
        raise ClaudeSettingsError(
            f"Claude Code is not configured by `lite configure claude` (no receipt at {state_path}); nothing to undo."
        )
    current: Final = load_json_or_empty(settings_path)
    _env_object(current, settings_path)
    ours: Final = tuple(path for path in receipt.written if _ours(current, path, receipt))
    kept: Final = tuple(path for path in receipt.written if path not in ours and _lookup(current, path).present)
    put_back: Final = _with_all(current, MappingProxyType({path: receipt.previous[path] for path in ours}))
    url_after: Final = _lookup(put_back, _BASE_URL_PATH)
    withheld: Final = tuple(
        WithheldCredential(path, _endpoint_text(receipt.endpoints[path]))
        for path in _CREDENTIAL_PATHS
        if path in ours and receipt.previous[path].present and receipt.endpoints[path] != url_after
    )
    absent: Final = OwnedValue(present=False)
    trimmed: Final = _with_all(put_back, MappingProxyType({item.key: absent for item in withheld}))
    settings: Final = (
        trimmed
        if _env(trimmed) or receipt.env_was_object
        else _with_key(trimmed, ENV_KEY, OwnedValue(present=receipt.env_present, value=None))
    )
    target: Final = _write_target(settings_path)
    file_removed: Final = not settings and not (receipt.file_existed and target.exists())
    kept_receipt: Final = (  # mutable-ok: pydantic serializes the update as given and rejects a mappingproxy
        receipt.model_copy(update={"written": {item.key: _fingerprint(absent) for item in withheld}})
        if withheld
        else None
    )
    staged_settings: Final = None if file_removed else _stage(target, settings)
    try:
        staged_receipt: Final = (
            None if kept_receipt is None else _stage(state_path, kept_receipt.model_dump(mode="json"))
        )
    except ClaudeSettingsError:
        if staged_settings is not None:
            discard_staged_json(staged_settings)
        raise
    _land(target, staged_settings, (staged_receipt,))
    _land(state_path, staged_receipt)
    return UnconfigureOutcome(
        restored=tuple(path for path in ours if _lookup(current, path) != _lookup(settings, path)),
        kept=kept,
        withheld=withheld,
        file_removed=file_removed,
    )


__all__ = (
    "ANTHROPIC_API_KEY_KEY",
    "ANTHROPIC_AUTH_TOKEN_KEY",
    "ANTHROPIC_BASE_URL_KEY",
    "ANTHROPIC_DEFAULT_MODEL_ENV_KEYS",
    "API_KEY_HELPER_KEY",
    "AUTOROUTE_BACKUP_PATH",
    "BACKUP_PATH",
    "CLAUDE_CONFIG_DIR_ENV",
    "CLAUDE_SETTINGS_PATH",
    "CONFIGURE_STATE_PATH",
    "ENABLE_GATEWAY_MODEL_DISCOVERY_KEY",
    "ENABLE_GATEWAY_MODEL_DISCOVERY_VALUE",
    "ENABLE_TOOL_SEARCH_KEY",
    "ENABLE_TOOL_SEARCH_VALUE",
    "ENV_KEY",
    "MODEL_KEY",
    "OWNED_ENV_KEYS",
    "OWNED_PATHS",
    "OWNED_TOP_LEVEL_KEYS",
    "SETTINGS_FILE_OWNERS",
    "STARTING_MODEL_ROLE",
    "ApiKeyHelper",
    "ClaudeCredential",
    "ClaudeSettingsError",
    "ConfigureReceipt",
    "KeepModel",
    "ModelChoice",
    "OwnedValue",
    "SettingsFileOwner",
    "StartOn",
    "StaticToken",
    "UnconfigureOutcome",
    "UnpinModel",
    "WithheldCredential",
    "claude_settings_path",
    "configure_claude_settings",
    "configure_state_path",
    "lite_api_key_helper_configured",
    "load_json_or_empty",
    "merge_claude_settings",
    "read_configure_receipt",
    "refuse_while_owned",
    "resolve_api_key_helper",
    "settings_file_owners",
    "unconfigure_claude_settings",
)

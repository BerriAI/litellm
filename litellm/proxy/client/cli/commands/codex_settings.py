import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import tomlkit
from pydantic import BaseModel, ConfigDict, ValidationError
from tomlkit.container import OutOfOrderTableProxy
from tomlkit.exceptions import TOMLKitError
from tomlkit.items import InlineTable, Table
from tomlkit.toml_document import TOMLDocument

from litellm.litellm_core_utils.private_json import (
    commit_staged_json,
    discard_staged_json,
    ensure_private_dir,
    stage_private_bytes,
    stage_private_json,
)

from .agents import CODEX_PROXY_PROVIDER, codex_proxy_provider

_PROVIDER_PATH: Final = f"model_providers.{CODEX_PROXY_PROVIDER}"
_OWNED_PATHS: Final = ("model_provider", "model", "profile", _PROVIDER_PATH)
_Table: TypeAlias = TOMLDocument | Table | InlineTable | OutOfOrderTableProxy
_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})
_MIN_CODEX_VERSION: Final = (0, 129, 0)


class CodexSettingsError(Exception):
    pass


class _Receipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    settings_path: str
    file_existed: bool
    providers_existed: bool
    previous: Mapping[str, str | None]
    written: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class CodexUnconfigureOutcome:
    restored: tuple[str, ...]
    kept: tuple[str, ...]
    file_removed: bool


def codex_configure_state_path(settings_path: Path) -> Path:
    target: Final = settings_path.resolve()
    digest: Final = hashlib.sha256(str(target).encode()).hexdigest()
    return target.parent / ".litellm" / f"codex_configure_{digest}.json"


def _read(settings_path: Path) -> TOMLDocument:
    try:
        document: Final = tomlkit.parse(settings_path.read_bytes()) if settings_path.exists() else tomlkit.document()
    except (OSError, UnicodeError, TOMLKitError) as error:
        raise CodexSettingsError(
            f"Could not read Codex settings at {settings_path}; no settings were changed"
        ) from error
    providers: Final = _mapping(document).get("model_providers")
    parent: Final = _table(providers)
    if providers is not None and parent is None:
        raise CodexSettingsError("Codex model_providers must be a TOML table; no settings were changed")
    entries: Final = _mapping(parent) if parent is not None else _EMPTY
    configured: Final = entries.get(CODEX_PROXY_PROVIDER)
    if configured is not None and _table(configured) is None:
        raise CodexSettingsError("Codex model_providers.litellm must be a TOML table; no settings were changed")
    return document


def _mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    return value


def _table(value: object) -> _Table | None:
    return value if isinstance(value, (TOMLDocument, Table, InlineTable, OutOfOrderTableProxy)) else None


def _snapshot(document: TOMLDocument, path: str) -> str | None:
    section, _, key = path.rpartition(".")
    parent: Final = _table(_mapping(document).get(section)) if section else document
    if parent is None or key not in parent:
        return None
    values: Final = _mapping(parent)
    return tomlkit.dumps(MappingProxyType({"value": values[key]}))


def _fingerprint(value: str | None) -> str:
    normalized: Final = "missing" if value is None else json.dumps(tomlkit.parse(value), sort_keys=True, default=str)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _with(document: TOMLDocument, path: str, snapshot: str | None) -> TOMLDocument:
    section, _, key = path.rpartition(".")
    if section and section not in document and snapshot is not None:
        contents: Final = tomlkit.parse(tomlkit.dumps(MappingProxyType({key: tomlkit.parse(snapshot).item("value")})))
        return tomlkit.parse(document.as_string() + "\n" + tomlkit.dumps(MappingProxyType({section: contents})))
    # mutable-ok: TOMLKit editing requires private node mutation to preserve comments and order
    updated: Final = tomlkit.parse(document.as_string())
    parent: Final = _table(_mapping(updated).get(section)) if section else updated
    if parent is None:
        return updated
    if snapshot is None:
        if key in parent:
            del parent[key]
    else:
        parent[key] = tomlkit.parse(snapshot).item("value")
    return updated


def _receipt(settings_path: Path) -> _Receipt | None:
    path: Final = codex_configure_state_path(settings_path)
    if not path.exists():
        return None
    try:
        receipt: Final = _Receipt.model_validate_json(path.read_bytes())
        if receipt.settings_path != str(settings_path.resolve()) or frozenset(receipt.previous) != frozenset(
            receipt.written
        ):
            raise ValueError("invalid receipt scope")
        if not frozenset(receipt.written) <= frozenset(_OWNED_PATHS):
            raise ValueError("invalid receipt ownership")
        for snapshot in receipt.previous.values():
            if snapshot is not None and tuple(tomlkit.parse(snapshot)) != ("value",):
                raise ValueError("invalid receipt snapshot")
    except (OSError, UnicodeError, TOMLKitError, ValidationError, ValueError) as error:
        raise CodexSettingsError(
            f"Could not read the Codex configure receipt at {path}; no settings were changed"
        ) from error
    return receipt


def _codex_version() -> str | None:
    try:
        result: Final = subprocess.run(("codex", "--version"), capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    return result.stdout if result.returncode == 0 else None


def require_safe_codex(*, version: Callable[[], str | None] = _codex_version) -> None:
    output: Final = version()
    matched: Final = re.fullmatch(r"codex-cli (\d+)\.(\d+)\.(\d+)", output.strip()) if output is not None else None
    if matched is not None and tuple(int(part) for part in matched.groups()) >= _MIN_CODEX_VERSION:
        return
    raise CodexSettingsError(
        "Codex 0.129.0 or newer (stable) must be installed before saving a gateway key. "
        "Older versions allow repository settings to redirect authenticated requests. "
        "Install or update Codex, check `codex --version`, then retry."
    )


def preflight_codex_settings(settings_path: Path) -> None:
    require_safe_codex()
    _read(settings_path)
    _receipt(settings_path)


def _ours(document: TOMLDocument, path: str, receipt: _Receipt) -> bool:
    return receipt.written.get(path) == _fingerprint(_snapshot(document, path))


def _stage_settings(path: Path, document: TOMLDocument) -> str:
    try:
        return stage_private_bytes(str(path), document.as_string().encode())
    except OSError as error:
        raise CodexSettingsError(f"Could not stage Codex settings at {path}; no settings were changed") from error


def _commit(path: Path, staged: str | None, commit: Callable[[str, str], None]) -> None:
    if staged is None:
        path.unlink(missing_ok=True)
    else:
        commit(staged, str(path))


def configure_codex_settings(
    base_url: str,
    api_key: str,
    model: str,
    settings_path: Path,
    *,
    commit: Callable[[str, str], None] = commit_staged_json,
) -> None:
    require_safe_codex()
    current: Final = _read(settings_path)
    earlier: Final = _receipt(settings_path)
    headers: Final = tomlkit.parse(tomlkit.dumps(MappingProxyType({"Authorization": f"Bearer {api_key}"})))
    provider_table: Final = tomlkit.parse(
        tomlkit.dumps(MappingProxyType({**codex_proxy_provider(base_url), "http_headers": headers}))
    )
    provider: Final = tomlkit.dumps(MappingProxyType({"value": provider_table}))
    selections: Final = tomlkit.parse(
        tomlkit.dumps(MappingProxyType({"model_provider": CODEX_PROXY_PROVIDER, "model": model}))
    )
    merged: Final = _with(
        _with(
            _with(_with(current, "profile", None), "model", _snapshot(selections, "model")),
            "model_provider",
            _snapshot(selections, "model_provider"),
        ),
        _PROVIDER_PATH,
        provider,
    )
    owned: Final = tuple(
        path
        for path in _OWNED_PATHS
        if _fingerprint(_snapshot(current, path)) != _fingerprint(_snapshot(merged, path))
        or (earlier is not None and _ours(current, path, earlier))
    )
    receipt: Final = _Receipt(
        settings_path=str(settings_path.resolve()),
        file_existed=settings_path.exists() if earlier is None else earlier.file_existed,
        providers_existed="model_providers" in current if earlier is None else earlier.providers_existed,
        previous=MappingProxyType(
            {
                path: earlier.previous[path]
                if earlier is not None and _ours(current, path, earlier)
                else _snapshot(current, path)
                for path in owned
            }
        ),
        written=MappingProxyType({path: _fingerprint(_snapshot(merged, path)) for path in owned}),
    )
    target: Final = settings_path.resolve()
    state_path: Final = codex_configure_state_path(settings_path)
    try:
        ensure_private_dir(state_path.parent)
        staged_receipt: Final = stage_private_json(str(state_path), receipt.model_dump(mode="json"))
    except OSError as error:
        raise CodexSettingsError(f"Could not stage the Codex configure receipt at {state_path}") from error
    try:
        staged_settings: Final = _stage_settings(target, merged)
    except CodexSettingsError:
        discard_staged_json(staged_receipt)
        raise
    try:
        commit(staged_receipt, str(state_path))
    except OSError as error:
        discard_staged_json(staged_receipt)
        discard_staged_json(staged_settings)
        raise CodexSettingsError(
            f"Could not write the Codex configure receipt at {state_path}; no settings were changed"
        ) from error
    try:
        commit(staged_settings, str(target))
    except OSError as error:
        discard_staged_json(staged_settings)
        try:
            _commit(
                state_path,
                None if earlier is None else stage_private_json(str(state_path), earlier.model_dump(mode="json")),
                commit_staged_json,
            )
        except OSError as rollback_error:
            raise CodexSettingsError(
                f"Codex settings were not written and its receipt at {state_path} could not be restored"
            ) from rollback_error
        raise CodexSettingsError(
            f"Could not write Codex settings at {settings_path}; the earlier receipt was restored"
        ) from error


def unconfigure_codex_settings(
    settings_path: Path, *, commit: Callable[[str, str], None] = commit_staged_json
) -> CodexUnconfigureOutcome:
    current: Final = _read(settings_path)
    receipt: Final = _receipt(settings_path)
    if receipt is None:
        raise CodexSettingsError("Codex is not configured by `lite configure codex`; nothing to undo")
    ours: Final = tuple(path for path in receipt.written if settings_path.exists() and _ours(current, path, receipt))
    restored_owned: Final = reduce(lambda document, path: _with(document, path, receipt.previous[path]), ours, current)
    providers: Final = _table(_mapping(restored_owned).get("model_providers"))
    restored: Final = (
        _with(restored_owned, "model_providers", None)
        if providers is not None and not providers and not receipt.providers_existed
        else restored_owned
    )
    target: Final = settings_path.resolve()
    file_removed: Final = not restored.as_string().strip() and not (receipt.file_existed and target.exists())
    staged: Final = None if file_removed else _stage_settings(target, restored)
    state_path: Final = codex_configure_state_path(settings_path)
    try:
        _commit(target, staged, commit)
        state_path.unlink()
    except OSError as error:
        if staged is not None:
            discard_staged_json(staged)
        raise CodexSettingsError(
            "Could not finish undoing Codex configuration; the receipt was kept for retry"
        ) from error
    return CodexUnconfigureOutcome(
        restored=tuple(path for path in ours if _snapshot(current, path) != _snapshot(restored, path)),
        kept=tuple(path for path in receipt.written if path not in ours and _snapshot(current, path) is not None),
        file_removed=file_removed,
    )

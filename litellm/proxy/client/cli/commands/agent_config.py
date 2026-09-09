"""Receipt-based configure and unconfigure for Codex's TOML config."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol, TypeAlias

import tomlkit
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError
from tomlkit.container import OutOfOrderTableProxy
from tomlkit.exceptions import ParseError
from tomlkit.items import InlineTable, Item, Table

from litellm.litellm_core_utils.private_json import (
    commit_staged_json,
    discard_staged_json,
    ensure_private_dir,
    stage_private_text,
)

ROOT_SECTION: Final = ""


class AgentConfigError(Exception):
    """Raised for any user-actionable failure while reading or writing an agent's config."""


@dataclass(frozen=True, slots=True)
class SettingsFileOwner:
    """A command that takes temporary ownership of a config file and restores it later."""

    backup_path: Path
    start_command: str
    stop_command: str


class OwnedValue(BaseModel):
    """What one key held at a moment in time; `present=False` is an absent key, not a null one."""

    model_config = ConfigDict(frozen=True)

    present: bool
    value: JsonValue = None


class SectionReceipt(BaseModel):
    """One owned section: whether it existed as an object, what its owned keys held, what was written."""

    model_config = ConfigDict(frozen=True)

    present: bool
    was_object: bool
    previous: Mapping[str, OwnedValue]
    written: Mapping[str, str]


class ConfigureReceipt(BaseModel):
    """What configure found and what it wrote, so unconfigure can undo only its own work.

    `previous` values are the ones every owned key had before the first configure; a repeat
    configure keeps them, since the values it would otherwise snapshot are its own. `written`
    holds fingerprints, so unconfigure can tell a key it still owns from one the user changed
    since, without keeping a second copy of a credential on disk. The file and section shapes
    are recorded too, so a file that did not exist, or a section that was absent or null, comes
    back exactly that way.
    """

    model_config = ConfigDict(frozen=True)

    file_existed: bool
    sections: Mapping[str, SectionReceipt]


@dataclass(frozen=True, slots=True)
class UnconfigureOutcome:
    """Which owned keys unconfigure put back, which it left because the user had changed them, and which
    credentials it left removed because the endpoint they belonged to was changed after configure."""

    restored: tuple[str, ...]
    kept: tuple[str, ...]
    withheld: tuple[str, ...] = ()
    file_removed: bool = False


@dataclass(frozen=True, slots=True)
class RestoreGroup:
    """Keys that only come back together: a credential is never restored next to an endpoint the user changed.

    `anchor` is the (section, key) of the endpoint; `dependents` are the credential slots. When the
    anchor no longer holds what configure wrote, the dependents stay removed instead of restored.
    """

    anchor: tuple[str, str]
    dependents: tuple[tuple[str, str], ...]


class ConfigDocument(Protocol):
    """A parsed config file, edited in place so the codec can keep the user's formatting."""

    def section(self, name: str) -> Mapping[str, JsonValue] | None: ...

    def section_is_object(self, name: str) -> bool: ...

    def section_is_scalar(self, name: str) -> bool: ...

    def set_value(self, section: str, key: str, value: object) -> None: ...

    def delete(self, section: str, key: str) -> None: ...

    def null_section(self, name: str) -> None: ...

    def drop_section(self, name: str) -> None: ...

    def is_empty(self) -> bool: ...

    def dumps(self) -> str: ...


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


_TOML_MAPPINGS: Final = (Table, InlineTable, OutOfOrderTableProxy)


def _plain(value: object) -> JsonValue:
    """A tomlkit item as the JSON value the receipt fingerprints; dates become their ISO text."""
    raw: Final = value.unwrap() if isinstance(value, (Item, OutOfOrderTableProxy)) else value
    return _json_value(raw)


class JsonDocument:
    """Claude Code's settings.json: sections are top-level object keys, the root is the file."""

    def __init__(self, root: dict[str, JsonValue]) -> None:  # mutable-ok: the document is edited in place
        self._root = root

    @classmethod
    def parse(cls, text: bytes, path: Path) -> "JsonDocument":
        if not text.strip():
            return cls({})  # mutable-ok: fresh document
        try:
            return cls(_JSON_OBJECT.validate_json(text))
        except ValidationError:
            raise AgentConfigError(
                f"{path} contains invalid JSON (or its root is not an object); cannot proceed safely."
            )

    def _container(self, section: str) -> dict[str, JsonValue]:  # mutable-ok: the document is edited in place
        if section == ROOT_SECTION:
            return self._root
        current: Final = self._root.get(section)
        if isinstance(current, dict):
            return current
        fresh: Final[dict[str, JsonValue]] = {}  # mutable-ok: new section object inserted into the document
        self._root[section] = fresh
        return fresh

    def section(self, name: str) -> Mapping[str, JsonValue] | None:
        value: Final = self._root if name == ROOT_SECTION else self._root.get(name)
        return value if isinstance(value, dict) else None

    def section_is_object(self, name: str) -> bool:
        return name == ROOT_SECTION or isinstance(self._root.get(name), dict)

    def section_is_scalar(self, name: str) -> bool:
        value: Final = self._root.get(name)
        return name in self._root and value is not None and not isinstance(value, dict)

    def set_value(self, section: str, key: str, value: object) -> None:
        self._container(section)[key] = _json_value(value)

    def delete(self, section: str, key: str) -> None:
        self._container(section).pop(key, None)

    def null_section(self, name: str) -> None:
        self._root[name] = None

    def drop_section(self, name: str) -> None:
        self._root.pop(name, None)

    def is_empty(self) -> bool:
        return not self._root

    def dumps(self) -> str:
        return json.dumps(self._root, indent=2)

    def root(self) -> Mapping[str, JsonValue]:
        return self._root


class TomlDocument:
    """Codex's config.toml, kept as a tomlkit document so comments and layout survive the rewrite."""

    def __init__(self, doc: tomlkit.TOMLDocument) -> None:
        self._doc = doc

    @classmethod
    def parse(cls, text: bytes, path: Path) -> "TomlDocument":
        try:
            return cls(tomlkit.parse(text.decode("utf-8")))
        except (UnicodeDecodeError, ParseError) as e:
            raise AgentConfigError(f"{path} is not valid TOML ({e}); cannot proceed safely.") from e

    def _mapping(self, name: str) -> Table | InlineTable | OutOfOrderTableProxy | None:
        """The section as tomlkit holds it: a standard table, an inline table, dotted keys, or fragments
        split by other tables all count, since every one of them is an object to TOML."""
        current: Final = self._doc.get(name)
        return current if isinstance(current, _TOML_MAPPINGS) else None

    def _container(self, section: str) -> tomlkit.TOMLDocument | Table | InlineTable | OutOfOrderTableProxy:
        if section == ROOT_SECTION:
            return self._doc
        current: Final = self._mapping(section)
        if current is not None:
            return current
        fresh: Final = tomlkit.table()
        self._doc[section] = fresh
        return fresh

    def section(self, name: str) -> Mapping[str, JsonValue] | None:
        source: Final = self._doc if name == ROOT_SECTION else self._mapping(name)
        if source is None:
            return None
        return MappingProxyType({str(key): _plain(item) for key, item in source.items()})

    def section_is_object(self, name: str) -> bool:
        return name == ROOT_SECTION or self._mapping(name) is not None

    def section_is_scalar(self, name: str) -> bool:
        return name in self._doc and self._mapping(name) is None

    def set_value(self, section: str, key: str, value: object) -> None:
        container: Final = self._container(section)
        if section == ROOT_SECTION or not isinstance(value, Mapping):
            container[key] = _toml_item(value)
            return
        self._set_child_table(section, key, value)

    def _set_child_table(self, section: str, key: str, value: Mapping[str, object]) -> None:
        container: Final = self._container(section)
        """Add a child table so the file still means what it says after tomlkit renders it.

        A `[section.key]` header is the readable form, but under a parent spelled as dotted root keys
        tomlkit emits that header before the remaining root keys, which TOML then reads as the child's
        members, and inside an inline parent it cannot render at all. The rendered document is checked
        against the intended meaning and the child falls back to an inline table when the header form
        would change it.
        """
        expected: Final = _json_value(value)
        container[key] = _toml_item(value)
        if self._child_means(section, key, expected):
            return
        del container[key]
        container[key] = _toml_item(value, inline=True)

    def _child_means(self, section: str, key: str, expected: JsonValue) -> bool:
        try:
            rendered: Final = tomlkit.parse(tomlkit.dumps(self._doc)).unwrap()
        except ParseError:
            return False
        parent: Final = rendered.get(section)
        return (
            isinstance(parent, dict)
            and _json_value(parent.get(key)) == expected
            and _json_value(rendered) == _json_value(self._doc.unwrap())
        )

    def delete(self, section: str, key: str) -> None:
        container: Final = self._container(section)
        if key in container:
            del container[key]

    def null_section(self, name: str) -> None:
        self.drop_section(name)

    def drop_section(self, name: str) -> None:
        if name in self._doc:
            del self._doc[name]

    def is_empty(self) -> bool:
        return not self._doc.unwrap()

    def dumps(self) -> str:
        return tomlkit.dumps(self._doc)


_MAX_CONFIG_DEPTH: Final = 32
_JsonContainer: TypeAlias = dict[str, JsonValue] | list[JsonValue]  # mutable-ok: a JSON document under construction
_JsonWorklist: TypeAlias = list[tuple[object, _JsonContainer, int]]  # mutable-ok: breadth-first worklist
_TomlTable: TypeAlias = Table | InlineTable
_TomlWorklist: TypeAlias = list[
    tuple[Mapping[object, object], _TomlTable, bool, int]
]  # mutable-ok: breadth-first worklist


def _scalar_json(value: object) -> JsonValue:
    if isinstance(value, (date, datetime, time)):
        return value.isoformat()
    return value  # pyright: ignore[reportReturnType]  # scalars pass through as the JSON value they already are


def _too_deep(depth: int) -> None:
    if depth > _MAX_CONFIG_DEPTH:
        raise AgentConfigError(f"The config nests more than {_MAX_CONFIG_DEPTH} levels deep; cannot proceed safely.")


def _empty_json_like(value: object) -> _JsonContainer:
    return {} if isinstance(value, Mapping) else []  # mutable-ok: JSON document under construction


def _json_value(value: object) -> JsonValue:
    """The JSON shape of a config value, built breadth-first so no config depth can recurse.

    Mappings become dicts, sequences become lists, dates become their ISO text; anything deeper
    than a config file could plausibly nest is an error rather than a stack.
    """
    if not isinstance(value, (Mapping, list, tuple)):
        return _scalar_json(value)
    root: Final = _empty_json_like(value)
    pending: Final[_JsonWorklist] = [(value, root, 0)]  # mutable-ok: worklist
    while pending:
        source, target, depth = pending.pop()
        _too_deep(depth)
        entries = source.items() if isinstance(source, Mapping) else enumerate(source)  # pyright: ignore[reportUnknownVariableType, reportAttributeAccessIssue]  # narrowed by the guards above
        for key, item in entries:
            nested = isinstance(item, (Mapping, list, tuple))
            converted = _empty_json_like(item) if nested else _scalar_json(item)
            if nested:
                pending.append((item, converted, depth + 1))  # pyright: ignore[reportArgumentType]  # nested is a fresh container
            if isinstance(target, dict):
                target[str(key)] = converted
            else:
                target.append(converted)
    return root


def _toml_item(value: object, inline: bool = False) -> object:
    """A tomlkit item for a config value, built breadth-first so no config depth can recurse.

    Mappings become tables (inline ones when `inline`, and everything under an inline table is
    inline too), tuples become arrays; scalars pass through for tomlkit to wrap.
    """
    if isinstance(value, tuple):
        return list(value)  # mutable-ok: tomlkit builds its Array from a list
    if not isinstance(value, Mapping):
        return value
    root: Final[_TomlTable] = tomlkit.inline_table() if inline else tomlkit.table()
    pending: Final[_TomlWorklist] = [(value, root, inline, 0)]  # mutable-ok: worklist
    while pending:
        source, target, as_inline, depth = pending.pop()
        _too_deep(depth)
        for key, item in source.items():
            if isinstance(item, Mapping):
                child = tomlkit.inline_table() if as_inline else tomlkit.table()
                target[str(key)] = child
                pending.append((item, child, as_inline, depth + 1))
            elif isinstance(item, tuple):
                target[str(key)] = list(item)  # mutable-ok: tomlkit builds its Array from a list
            else:
                target[str(key)] = item
    return root


def refuse_while_owned(path: Path, owners: Sequence[SettingsFileOwner]) -> None:
    for owner in owners:
        if owner.backup_path.exists():
            raise AgentConfigError(
                f"`{owner.start_command}` is currently managing {path} (backup at "
                f"{owner.backup_path}) and will restore it when it stops. "
                f"Run `{owner.stop_command}` first, then retry."
            )


def read_bytes_or_empty(path: Path) -> bytes:
    try:
        return path.read_bytes() if path.exists() else b""
    except OSError as e:
        raise AgentConfigError(f"Could not read {path}: {e}") from e


def write_target(path: Path) -> Path:
    """Write through a symlinked config file rather than replacing the link.

    os.replace() would swap the symlink itself for a regular file, silently detaching a config
    that is symlinked into a dotfiles repo, and there is no backup to undo that.
    """
    return path.resolve() if path.is_symlink() else path


def _stage(path: Path, text: str) -> str:
    try:
        return stage_private_text(str(path), text)
    except OSError as e:
        raise AgentConfigError(f"Could not write {path}: {e}") from e


def _owned(container: Mapping[str, JsonValue] | None, key: str) -> OwnedValue:
    if container is None:
        return OwnedValue(present=False)
    return OwnedValue(present=key in container, value=container.get(key))


def fingerprint(owned: OwnedValue) -> str:
    return hashlib.sha256(json.dumps(owned.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()


def _section_receipt(
    before: ConfigDocument, after: ConfigDocument, name: str, keys: Sequence[str], earlier: SectionReceipt | None
) -> SectionReceipt:
    before_section: Final = before.section(name)
    if name != ROOT_SECTION and before.section_is_scalar(name):
        raise AgentConfigError(
            f'The config has a non-object "{name}" value, which this would discard. Fix or remove it, then retry.'
        )
    after_section: Final = after.section(name)
    if earlier is None:
        return SectionReceipt(
            present=_raw_present(before, name),
            was_object=before.section_is_object(name),
            previous=MappingProxyType({key: _owned(before_section, key) for key in keys}),
            written=MappingProxyType({key: fingerprint(_owned(after_section, key)) for key in keys}),
        )
    return SectionReceipt(
        present=earlier.present,
        was_object=earlier.was_object,
        previous=MappingProxyType({key: earlier.previous.get(key, _owned(before_section, key)) for key in keys}),
        written=MappingProxyType(
            {key: _written_fingerprint(before_section, after_section, key, earlier) for key in keys}
        ),
    )


def _written_fingerprint(
    before: Mapping[str, JsonValue] | None, after: Mapping[str, JsonValue] | None, key: str, earlier: SectionReceipt
) -> str:
    """What the repeat configure counts as its own for one key.

    A key the merge changed is ours at its new value. A key the merge left alone keeps the earlier
    fingerprint: if the user edited it since the first configure, that fingerprint no longer matches
    and unconfigure will report it kept rather than deleting the user's edit as if it were ours.
    """
    after_owned: Final = _owned(after, key)
    if fingerprint(_owned(before, key)) != fingerprint(after_owned) or key not in earlier.written:
        return fingerprint(after_owned)
    return earlier.written[key]


def _raw_present(document: ConfigDocument, name: str) -> bool:
    root: Final = document.section(ROOT_SECTION)
    return name == ROOT_SECTION or (root is not None and name in root)


def read_configure_receipt(state_path: Path) -> ConfigureReceipt | None:
    if not state_path.exists():
        return None
    try:
        return ConfigureReceipt.model_validate_json(state_path.read_bytes())
    except (OSError, ValidationError) as e:
        raise AgentConfigError(
            f"{state_path} is not a readable configure receipt ({e}). "
            "Remove it and edit the agent's config by hand if it still points at the proxy."
        ) from e


def release_keys(document: ConfigDocument, receipt: ConfigureReceipt, keys: Sequence[tuple[str, str]]) -> None:
    """Put back the pre-configure value of each key that still holds what configure wrote."""
    for section, key in keys:
        recorded = receipt.sections.get(section)
        if recorded is None or key not in recorded.written:
            continue
        if fingerprint(_owned(document.section(section), key)) != recorded.written[key]:
            continue
        previous = recorded.previous.get(key, OwnedValue(present=False))
        if previous.present:
            document.set_value(section, key, previous.value)
        else:
            document.delete(section, key)


def configure_document(
    path: Path,
    state_path: Path,
    owners: Sequence[SettingsFileOwner],
    parse: Callable[[bytes, Path], ConfigDocument],
    owned: Mapping[str, Sequence[str]],
    merge: Callable[[ConfigDocument], None],
    release: Sequence[tuple[str, str]] = (),
    commit: Callable[[str, str], None] = commit_staged_json,
) -> None:
    """Persistently rewrite an agent's config, recording how to undo it.

    Both files are staged before either is committed, so a full disk or a read-only directory
    fails before anything changes. The two commits are still two renames, so if the config
    rename fails after the receipt landed, the earlier receipt is put back (or the new one
    removed on a first configure): the receipt on disk never describes a config that was not
    written. A repeat configure keeps the receipt's original `previous` snapshot and only
    refreshes what was written, so unconfigure still returns to the pre-configure state.
    `release` names owned keys whose earlier pin should be let go of before merging.
    """
    refuse_while_owned(path, owners)
    text: Final = read_bytes_or_empty(path)
    before: Final = parse(text, path)
    after: Final = parse(text, path)
    earlier: Final = read_configure_receipt(state_path)
    if earlier is not None:
        release_keys(after, earlier, release)
    merge(after)
    receipt: Final = ConfigureReceipt(
        file_existed=earlier.file_existed if earlier is not None else path.exists(),
        sections=MappingProxyType(
            {
                name: _section_receipt(before, after, name, keys, earlier.sections.get(name) if earlier else None)
                for name, keys in owned.items()
            }
        ),
    )
    target: Final = write_target(path)
    try:
        ensure_private_dir(state_path.parent)
    except OSError as e:
        raise AgentConfigError(f"Could not write {state_path}: {e}") from e
    staged_receipt: Final = _stage(state_path, json.dumps(receipt.model_dump(mode="json"), indent=2))
    try:
        staged_config: Final = _stage(target, after.dumps())
    except AgentConfigError:
        discard_staged_json(staged_receipt)
        raise
    try:
        commit(staged_receipt, str(state_path))
    except OSError as e:
        discard_staged_json(staged_config)
        raise AgentConfigError(f"Could not write {state_path}: {e}") from e
    try:
        commit(staged_config, str(target))
    except OSError as config_error:
        try:
            _restore_receipt(state_path, earlier, commit)
        except (AgentConfigError, OSError) as receipt_error:
            raise AgentConfigError(
                f"Could not write {target}: {config_error}. The receipt at {state_path} now describes a config "
                f"that was not written and could not be put back either ({receipt_error}); remove it before retrying."
            ) from config_error
        raise AgentConfigError(f"Could not write {target}: {config_error}") from config_error


def _restore_receipt(state_path: Path, earlier: ConfigureReceipt | None, commit: Callable[[str, str], None]) -> None:
    if earlier is None:
        state_path.unlink(missing_ok=True)
        return
    commit(_stage(state_path, json.dumps(earlier.model_dump(mode="json"), indent=2)), str(state_path))


def _still_ours(document: ConfigDocument, receipt: ConfigureReceipt, section: str, key: str) -> bool:
    recorded: Final = receipt.sections.get(section)
    return recorded is not None and fingerprint(_owned(document.section(section), key)) == recorded.written.get(key)


def unconfigure_document(
    path: Path,
    state_path: Path,
    owners: Sequence[SettingsFileOwner],
    parse: Callable[[bytes, Path], ConfigDocument],
    label: str,
    groups: Sequence[RestoreGroup] = (),
) -> UnconfigureOutcome:
    """Undo configure, restoring only the keys the user has not changed since."""
    refuse_while_owned(path, owners)
    receipt: Final = read_configure_receipt(state_path)
    if receipt is None:
        raise AgentConfigError(
            f"{label} is not configured by `lite configure` (no receipt at {state_path}); nothing to undo."
        )
    document: Final = parse(read_bytes_or_empty(path), path)
    for name in receipt.sections:
        if name != ROOT_SECTION and document.section_is_scalar(name):
            raise AgentConfigError(
                f'The config has a non-object "{name}" value, which this would discard. Fix or remove it, then retry.'
            )
    withheld_keys: Final = frozenset(
        dependent
        for group in groups
        if not _still_ours(document, receipt, *group.anchor)
        for dependent in group.dependents
    )
    restored: Final[list[str]] = []  # mutable-ok: report accumulator, frozen into the outcome below
    kept: Final[list[str]] = []  # mutable-ok: report accumulator, frozen into the outcome below
    withheld: Final[list[str]] = []  # mutable-ok: report accumulator, frozen into the outcome below
    for name, section in receipt.sections.items():
        for key in section.written:
            label_key = key if name == ROOT_SECTION else f"{name}.{key}"
            previous = section.previous.get(key, OwnedValue(present=False))
            if section.written[key] == fingerprint(previous):
                continue
            if fingerprint(_owned(document.section(name), key)) != section.written[key]:
                kept.append(label_key)
                continue
            if (name, key) in withheld_keys and previous.present:
                document.delete(name, key)
                withheld.append(label_key)
                continue
            if previous.present:
                document.set_value(name, key, previous.value)
            else:
                document.delete(name, key)
            restored.append(label_key)
        if name != ROOT_SECTION and not section.was_object and not document.section(name):
            if section.present:
                document.null_section(name)
            else:
                document.drop_section(name)
    target: Final = write_target(path)
    file_removed: Final = document.is_empty() and not receipt.file_existed
    try:
        if file_removed:
            target.unlink(missing_ok=True)
        else:
            commit_staged_json(_stage(target, document.dumps()), str(target))
        state_path.unlink(missing_ok=True)
    except OSError as e:
        raise AgentConfigError(f"Could not write {target}: {e}") from e
    return UnconfigureOutcome(
        restored=tuple(restored), kept=tuple(kept), withheld=tuple(withheld), file_removed=file_removed
    )


__all__ = (
    "ROOT_SECTION",
    "AgentConfigError",
    "ConfigDocument",
    "ConfigureReceipt",
    "JsonDocument",
    "OwnedValue",
    "RestoreGroup",
    "SectionReceipt",
    "SettingsFileOwner",
    "TomlDocument",
    "UnconfigureOutcome",
    "configure_document",
    "fingerprint",
    "read_bytes_or_empty",
    "read_configure_receipt",
    "refuse_while_owned",
    "release_keys",
    "unconfigure_document",
    "write_target",
)

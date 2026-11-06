"""Repack a stored skill upload into the archive shape Agent Skills clients install from.

Uploads follow the Anthropic Skills API layout, where every file sits under a single
top-level folder. Discovery clients read ``SKILL.md`` from the archive root, so that
folder is stripped and the zip is rebuilt with fixed entry timestamps, which keeps the
SHA-256 digest published in the index reproducible for identical uploads.
"""

import hashlib
import io
import lzma
import re
import stat
import struct
import zipfile
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Final

import yaml
from pydantic import ConfigDict, TypeAdapter, ValidationError

MAX_ARCHIVE_UNPACKED_BYTES: Final = 50 * 1024 * 1024
MAX_ARCHIVE_ENTRIES: Final = 1000
MAX_SKILL_FILE_BYTES: Final = 10 * 1024 * 1024
MAX_SKILL_MANIFEST_BYTES: Final = 1024 * 1024
SKILL_ARCHIVE_ERRORS: Final = (
    zipfile.BadZipFile,
    OSError,
    RuntimeError,
    EOFError,
    lzma.LZMAError,
    zlib.error,
    struct.error,
    UnicodeError,
)
SKILL_MANIFEST_FILENAME: Final = "SKILL.md"

_ZIP_ENTRY_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_ZIP_ENTRY_MODE: Final = 0o644
_FRONTMATTER_PATTERN: Final = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
_AGENT_SKILL_FRONTMATTER_PATTERN: Final = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)
_AGENT_SKILL_YAML_SCALARS: Final = MappingProxyType(
    {
        "tag:yaml.org,2002:null": re.compile(r"^(?:null|Null|NULL|~|)$"),
        "tag:yaml.org,2002:bool": re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
        "tag:yaml.org,2002:int": re.compile(r"^(?:[-+]?[0-9]+|0o[0-7]+|0x[0-9a-fA-F]+)$"),
        "tag:yaml.org,2002:float": re.compile(
            r"^(?:[-+]?(?:\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?[0-9]+)?"
            r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$"
        ),
    }
)
_MAX_FRONTMATTER_DEPTH: Final = 32
_WINDOWS_DRIVE_PATTERN: Final = re.compile(r"^[A-Za-z]:")
_EMPTY_FRONTMATTER: Final[Mapping[str, object]] = MappingProxyType({})
_DECODED_YAML: Final = TypeAdapter(object)
_YAML_EVENTS: Final = TypeAdapter(Iterable[object])
_YAML_NODES: Final = TypeAdapter(tuple[yaml.Node, ...], config=ConfigDict(arbitrary_types_allowed=True))
_YAML_PAIRS: Final = TypeAdapter(
    tuple[tuple[yaml.Node, yaml.Node], ...], config=ConfigDict(arbitrary_types_allowed=True)
)


class _AgentSkillFrontmatterLoader(yaml.BaseLoader):
    yaml_implicit_resolvers: ClassVar[Mapping[str | None, Sequence[tuple[str, re.Pattern[str]]]]] = {  # pyright: ignore[reportIncompatibleVariableOverride]  # PyYAML resolver tables are class state
        None: list(_AGENT_SKILL_YAML_SCALARS.items())
    }


@dataclass(frozen=True, slots=True)
class SkillArchive:
    content: bytes
    digest: str
    declared_name: str | None
    declared_description: str | None
    manifest_content: bytes


def read_skill_archive_members(
    stored_content: bytes, *, for_upload: bool = False, skill_path: str | None = None
) -> tuple[tuple[str, bytes, int], ...] | None:
    try:
        with zipfile.ZipFile(io.BytesIO(stored_content)) as uploaded:
            return _flattened_members(uploaded, for_upload=for_upload, skill_path=skill_path)
    except SKILL_ARCHIVE_ERRORS:
        return None


def read_skill_archive_entry(uploaded: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    with uploaded.open(info) as member:
        return member.read(info.file_size + 1)


def build_skill_archive(stored_content: bytes) -> SkillArchive | None:
    """Return the installable archive for an upload, or None when it holds no root SKILL.md."""
    members: Final = read_skill_archive_members(stored_content)
    if members is None:
        return None

    manifest_content: Final = next(data for name, data, _ in members if name == SKILL_MANIFEST_FILENAME)
    frontmatter: Final = _manifest_frontmatter(manifest_content)
    content: Final = pack_skill_archive(members)
    return SkillArchive(
        content=content,
        digest=f"sha256:{hashlib.sha256(content).hexdigest()}",
        declared_name=_frontmatter_text(frontmatter, "name"),
        declared_description=_frontmatter_text(frontmatter, "description"),
        manifest_content=manifest_content,
    )


def build_agent_skill_archive(
    content: bytes, *, skill_path: str | None = None, plugin_name: str | None = None
) -> SkillArchive | None:
    members: Final = read_skill_archive_members(content, for_upload=True, skill_path=skill_path)
    selected: Final = (
        members
        if members is not None or plugin_name is None
        else read_skill_archive_members(content, for_upload=True, skill_path=f"skills/{plugin_name}")
    )
    if selected is None:
        return None
    manifest_content: Final = next(data for path, data, _ in selected if path == SKILL_MANIFEST_FILENAME)
    manifest: Final = agent_skill_frontmatter(manifest_content)
    name: Final = _frontmatter_text(manifest, "name")
    description: Final = _frontmatter_text(manifest, "description")
    if (
        name is None
        or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name)
        or len(name) > 64
        or description is None
        or len(description) > 1024
    ):
        return None
    archive: Final = pack_skill_archive(selected)
    return SkillArchive(
        content=archive,
        digest=f"sha256:{hashlib.sha256(archive).hexdigest()}",
        declared_name=name,
        declared_description=description,
        manifest_content=manifest_content,
    )


def _flattened_members(
    uploaded: zipfile.ZipFile, *, for_upload: bool, skill_path: str | None
) -> tuple[tuple[str, bytes, int], ...] | None:
    kept: Final = skill_archive_entries(uploaded, for_upload=for_upload)
    if kept is None:
        return None
    prefix: Final = f"{skill_path}/" if skill_path is not None else _common_root_prefix(tuple(path for _, path in kept))
    flattened: Final = tuple((info, path[len(prefix) :]) for info, path in kept if path.startswith(prefix))
    names: Final = frozenset(name for _, name in flattened)
    if (
        SKILL_MANIFEST_FILENAME not in names
        or len(names) != len(flattened)
        or (for_upload and any(not safe_skill_archive_entry(info) for info, _ in flattened))
    ):
        return None

    return tuple(
        (name, read_skill_archive_entry(uploaded, info), _entry_mode(info, preserve_executable_bits=for_upload))
        for info, name in sorted(flattened, key=lambda member: member[1])
    )


def skill_archive_entries(
    uploaded: zipfile.ZipFile, *, for_upload: bool
) -> tuple[tuple[zipfile.ZipInfo, str], ...] | None:
    infos: Final = tuple(info for info in uploaded.infolist() if for_upload or not info.is_dir())
    if not infos or len(infos) > MAX_ARCHIVE_ENTRIES:
        return None
    if sum(info.file_size for info in infos) > MAX_ARCHIVE_UNPACKED_BYTES:
        return None

    normalized: Final = tuple(
        (info, _normalized_path(info.orig_filename if for_upload else info.filename)) for info in infos
    )
    if any(path is None for _, path in normalized):
        return None
    if for_upload and len(frozenset(path for _, path in normalized)) != len(normalized):
        return None

    return tuple(
        (info, path)
        for info, path in normalized
        if path is not None and not info.is_dir() and (not for_upload or not _is_macos_metadata(path))
    )


def _entry_mode(info: zipfile.ZipInfo, *, preserve_executable_bits: bool) -> int:
    if not preserve_executable_bits or info.create_system != 3:
        return _ZIP_ENTRY_MODE
    mode: Final = info.external_attr >> 16
    if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
        return _ZIP_ENTRY_MODE
    return _ZIP_ENTRY_MODE | (mode & 0o111)


def safe_skill_archive_entry(info: zipfile.ZipInfo) -> bool:
    return (
        info.file_size <= MAX_SKILL_FILE_BYTES
        and not info.flag_bits & 1
        and (info.create_system != 3 or stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR))
    )


def _is_macos_metadata(path: str) -> bool:
    return path.split("/", 1)[0] == "__MACOSX" or path.rsplit("/", 1)[-1] == ".DS_Store"


def _common_root_prefix(paths: tuple[str, ...]) -> str:
    roots: Final = frozenset(path.split("/", 1)[0] for path in paths)
    if len(roots) != 1 or not all("/" in path for path in paths):
        return ""
    return f"{next(iter(roots))}/"


def _normalized_path(raw_path: str) -> str | None:
    if not raw_path or "\0" in raw_path or "\\" in raw_path:
        return None
    if raw_path.startswith("/") or _WINDOWS_DRIVE_PATTERN.match(raw_path):
        return None
    parts: Final = tuple(part for part in raw_path.split("/") if part)
    if not parts or any(part in (".", "..") for part in parts):
        return None
    return "/".join(parts)


def _manifest_frontmatter(manifest: bytes) -> Mapping[str, object]:
    match: Final = _FRONTMATTER_PATTERN.match(manifest.decode("utf-8", errors="replace"))
    if match is None:
        return _EMPTY_FRONTMATTER
    try:
        parsed: Final = _DECODED_YAML.validate_python(yaml.safe_load(match.group(1)))
    except yaml.YAMLError:
        return _EMPTY_FRONTMATTER
    if not isinstance(parsed, dict):
        return _EMPTY_FRONTMATTER
    return parsed


def agent_skill_frontmatter(manifest: bytes) -> Mapping[str, str]:
    if len(manifest) > MAX_SKILL_MANIFEST_BYTES:
        return MappingProxyType({})
    try:
        match: Final = _AGENT_SKILL_FRONTMATTER_PATTERN.match(manifest.decode("utf-8-sig"))
        if match is None:
            return MappingProxyType({})
        frontmatter: Final = match.group(1)
        events: Final = _YAML_EVENTS.validate_python(yaml.parse(frontmatter, Loader=yaml.BaseLoader))  # pyright: ignore[reportUnknownMemberType]  # PyYAML parse stubs omit their event iterator return type
        if any(isinstance(event, yaml.AliasEvent) for event in events):
            return MappingProxyType({})
        node: Final = _DECODED_YAML.validate_python(yaml.compose(frontmatter, Loader=_AgentSkillFrontmatterLoader))  # pyright: ignore[reportUnknownMemberType]  # PyYAML compose stubs omit their return type
    except (yaml.YAMLError, UnicodeError, RecursionError, ValidationError):
        return MappingProxyType({})
    if not isinstance(node, yaml.MappingNode) or not _valid_frontmatter_node(node):
        return MappingProxyType({})
    pairs: Final = _YAML_PAIRS.validate_python(node.value)
    fields: Final = {
        key_text: value_text
        for key, value in pairs
        if (key_text := _frontmatter_string(key)) is not None and (value_text := _frontmatter_string(value)) is not None
    }
    if "name" not in fields or "description" not in fields:
        return MappingProxyType({})
    return MappingProxyType(fields)


def _frontmatter_string(node: yaml.Node) -> str | None:
    if not isinstance(node, yaml.ScalarNode) or node.tag != "tag:yaml.org,2002:str":
        return None
    value: Final = _DECODED_YAML.validate_python(node.value)
    return value if isinstance(value, str) else None


def _valid_frontmatter_node(node: yaml.Node, depth: int = 0) -> bool:
    if depth > _MAX_FRONTMATTER_DEPTH:
        return False
    if isinstance(node, yaml.ScalarNode):
        pattern: Final = _AGENT_SKILL_YAML_SCALARS.get(node.tag)
        scalar: Final = _DECODED_YAML.validate_python(node.value)
        return isinstance(scalar, str) and (
            node.tag == "tag:yaml.org,2002:str" or (pattern is not None and pattern.fullmatch(scalar) is not None)
        )
    if isinstance(node, yaml.SequenceNode):
        values: Final = _YAML_NODES.validate_python(node.value)
        return node.tag == "tag:yaml.org,2002:seq" and all(
            _valid_frontmatter_node(value, depth + 1) for value in values
        )
    if not isinstance(node, yaml.MappingNode) or node.tag != "tag:yaml.org,2002:map":
        return False
    pairs: Final = _YAML_PAIRS.validate_python(node.value)
    keys: Final = tuple(_frontmatter_string(key) for key, _ in pairs)
    return (
        None not in keys
        and len(frozenset(keys)) == len(keys)
        and all(_valid_frontmatter_node(value, depth + 1) for _, value in pairs)
    )


def _frontmatter_text(frontmatter: Mapping[str, object], key: str) -> str | None:
    value: Final = frontmatter.get(key)
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _zip_entry(name: str, mode: int) -> zipfile.ZipInfo:
    entry: Final = zipfile.ZipInfo(filename=name, date_time=_ZIP_ENTRY_TIMESTAMP)
    entry.create_system = 3
    entry.external_attr = mode << 16
    return entry


def pack_skill_archive(members: tuple[tuple[str, bytes, int], ...], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer: Final = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as repacked:
        for name, data, mode in members:
            repacked.writestr(_zip_entry(name, mode), data, compress_type=compression)
    return buffer.getvalue()

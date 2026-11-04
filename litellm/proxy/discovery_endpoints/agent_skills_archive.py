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
import zipfile
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

import yaml
from pydantic import TypeAdapter

MAX_ARCHIVE_UNPACKED_BYTES: Final = 50 * 1024 * 1024
MAX_ARCHIVE_ENTRIES: Final = 1000
SKILL_MANIFEST_FILENAME: Final = "SKILL.md"

_ZIP_ENTRY_TIMESTAMP: Final = (1980, 1, 1, 0, 0, 0)
_ZIP_ENTRY_MODE: Final = 0o644
_FRONTMATTER_PATTERN: Final = re.compile(r"^---\s*\n(.*?)\n---\s*(?:\n|$)", re.DOTALL)
_WINDOWS_DRIVE_PATTERN: Final = re.compile(r"^[A-Za-z]:")
_EMPTY_FRONTMATTER: Final[Mapping[str, object]] = MappingProxyType({})
_DECODED_YAML: Final = TypeAdapter(object)


@dataclass(frozen=True, slots=True)
class SkillArchive:
    content: bytes
    digest: str
    declared_name: str | None
    declared_description: str | None


def read_skill_archive_members(
    stored_content: bytes, *, for_upload: bool = False
) -> tuple[tuple[str, bytes, int], ...] | None:
    try:
        with zipfile.ZipFile(io.BytesIO(stored_content)) as uploaded:
            return _flattened_members(uploaded, for_upload=for_upload)
    except (zipfile.BadZipFile, OSError, RuntimeError, EOFError, lzma.LZMAError, zlib.error):
        return None


def build_skill_archive(stored_content: bytes) -> SkillArchive | None:
    """Return the installable archive for an upload, or None when it holds no root SKILL.md."""
    members: Final = read_skill_archive_members(stored_content)
    if members is None:
        return None

    frontmatter: Final = _manifest_frontmatter(
        next(data for name, data, _ in members if name == SKILL_MANIFEST_FILENAME)
    )
    content: Final = pack_skill_archive(members)
    return SkillArchive(
        content=content,
        digest=f"sha256:{hashlib.sha256(content).hexdigest()}",
        declared_name=_frontmatter_text(frontmatter, "name"),
        declared_description=_frontmatter_text(frontmatter, "description"),
    )


def _flattened_members(uploaded: zipfile.ZipFile, *, for_upload: bool) -> tuple[tuple[str, bytes, int], ...] | None:
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
    if for_upload and (
        len(frozenset(path for _, path in normalized)) != len(normalized)
        or any(not _safe_upload_entry(info) for info in infos)
    ):
        return None

    kept: Final = tuple(
        (info, path)
        for info, path in normalized
        if path is not None and not info.is_dir() and (not for_upload or not _is_macos_metadata(path))
    )
    prefix: Final = _common_root_prefix(tuple(path for _, path in kept))
    flattened: Final = tuple((info, path[len(prefix) :]) for info, path in kept)
    names: Final = frozenset(name for _, name in flattened)
    if SKILL_MANIFEST_FILENAME not in names or len(names) != len(flattened):
        return None

    return tuple(
        (name, uploaded.read(info), _entry_mode(info, preserve_executable_bits=for_upload))
        for info, name in sorted(flattened, key=lambda member: member[1])
    )


def _entry_mode(info: zipfile.ZipInfo, *, preserve_executable_bits: bool) -> int:
    if not preserve_executable_bits or info.create_system != 3:
        return _ZIP_ENTRY_MODE
    mode: Final = info.external_attr >> 16
    if stat.S_IFMT(mode) not in (0, stat.S_IFREG):
        return _ZIP_ENTRY_MODE
    return _ZIP_ENTRY_MODE | (mode & 0o111)


def _safe_upload_entry(info: zipfile.ZipInfo) -> bool:
    return not info.flag_bits & 1 and (
        info.create_system != 3 or stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG, stat.S_IFDIR)
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

import asyncio
import hashlib
import io
import os
import re
import stat
import struct
import tempfile
import zipfile
import zlib
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final, NamedTuple, TypeAlias
from urllib.parse import urlsplit

from litellm.litellm_core_utils.url_utils import SSRFError, async_safe_get
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # helper signature is partially unknown in the external HTTP layer
    httpxSpecialProvider,
)

MAX_ARCHIVE_BYTES: Final = 10 * 1024 * 1024
MAX_DOWNLOAD_BYTES: Final = MAX_ARCHIVE_BYTES * 5
MAX_ARCHIVE_TOTAL_ENTRIES: Final = 20_000
MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES: Final = 5 * 1024 * 1024
PLUGIN_JSON_PATH: Final = ".claude-plugin/plugin.json"
_FETCH_TIMEOUT_SECONDS: Final = 60.0
_ZIP_EPOCH: Final = (1980, 1, 1, 0, 0, 0)
_ALLOWED_COMPRESSION: Final = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_GITHUB_HOSTS: Final = frozenset({"github.com", "www.github.com"})
_GITHUB_REPO_RE: Final = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")
_GITHUB_REF_RE: Final = re.compile(r"^[A-Za-z0-9._/-]+$")
_COMMIT_SHA_RE: Final = re.compile(r"[0-9a-f]{40}")
_WINDOWS_DRIVE_RE: Final = re.compile(r"^[A-Za-z]:")
_ZIP_EOCD_SIGNATURE: Final = b"PK\x05\x06"
_ZIP64_EOCD_SIGNATURE: Final = b"PK\x06\x06"
_ZIP64_EOCD_LOCATOR_SIGNATURE: Final = b"PK\x06\x07"
_ZIP_EOCD_FIXED_SIZE: Final = 22
_ZIP_EOCD_MAX_COMMENT: Final = 0xFFFF
_ZIP64_EOCD_FIXED_SIZE: Final = 56
_ZIP64_EOCD_LOCATOR_SIZE: Final = 20


class ArchiveBuildError(Exception): ...


class BuiltArchive(NamedTuple):
    data: bytes
    sha256: str
    ref: str | None


class _Member(NamedTuple):
    content: bytes
    mode: int


_Members: TypeAlias = Mapping[str, _Member]
_Entries: TypeAlias = Mapping[str, zipfile.ZipInfo]


def _size_limit_label() -> str:
    return f"{MAX_ARCHIVE_BYTES // (1024 * 1024)} MB"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_dir() -> Path:
    configured: Final = os.environ.get("LITELLM_PLUGIN_ARCHIVE_DIR", "").strip()
    return Path(configured) if configured else Path(tempfile.gettempdir()) / "litellm_plugin_archives"


def archive_path(plugin_name: str) -> Path:
    base: Final = os.path.abspath(archive_dir())
    candidate: Final = os.path.normpath(os.path.join(base, f"{plugin_name}.zip"))
    if not candidate.startswith(base + os.sep):
        raise ArchiveBuildError(f"{plugin_name!r} is not a valid plugin name")
    return Path(candidate)


def read_stored_archive(plugin_name: str) -> bytes | None:
    try:
        return archive_path(plugin_name).read_bytes()
    except FileNotFoundError:
        return None


def store_archive(plugin_name: str, data: bytes) -> None:
    target: Final = archive_path(plugin_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as temp_file:
            temp_file.write(data)
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def delete_stored_archive(plugin_name: str) -> None:
    archive_path(plugin_name).unlink(missing_ok=True)


def _http_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(
        llm_provider=httpxSpecialProvider.PluginArchive,
        params={"timeout": _FETCH_TIMEOUT_SECONDS},
    )


@lru_cache(maxsize=None)
def archive_lock(plugin_name: str) -> asyncio.Lock:
    return asyncio.Lock()


def _zip_directory_budget(content: bytes) -> tuple[int, int] | None:
    if len(content) < _ZIP_EOCD_FIXED_SIZE:
        return None

    eocd_start: Final = max(0, len(content) - (_ZIP_EOCD_FIXED_SIZE + _ZIP_EOCD_MAX_COMMENT))
    eocd_offset: Final = content.rfind(_ZIP_EOCD_SIGNATURE, eocd_start)
    if eocd_offset < 0 or eocd_offset + _ZIP_EOCD_FIXED_SIZE > len(content):
        return None

    try:
        _, _, _, _, entries_total, directory_size, _, comment_len = struct.unpack_from("<4sHHHHIIH", content, eocd_offset)
    except struct.error:
        return None

    if eocd_offset + _ZIP_EOCD_FIXED_SIZE + comment_len > len(content):
        return None

    if entries_total != 0xFFFF and directory_size != 0xFFFFFFFF:
        return entries_total, directory_size

    locator_offset: Final = eocd_offset - _ZIP64_EOCD_LOCATOR_SIZE
    if locator_offset < 0:
        return None
    try:
        locator_signature, _, zip64_eocd_offset, _ = struct.unpack_from("<4sIQI", content, locator_offset)
    except struct.error:
        return None
    if locator_signature != _ZIP64_EOCD_LOCATOR_SIGNATURE:
        return None
    if zip64_eocd_offset + _ZIP64_EOCD_FIXED_SIZE > len(content):
        return None

    try:
        (
            signature,
            _,
            _,
            _,
            _,
            _,
            _,
            entries_total_zip64,
            directory_size_zip64,
            _,
        ) = struct.unpack_from("<4sQHHIIQQQQ", content, zip64_eocd_offset)
    except struct.error:
        return None
    if signature != _ZIP64_EOCD_SIGNATURE:
        return None
    if entries_total_zip64 > 0xFFFFFFFF or directory_size_zip64 > 0xFFFFFFFF:
        return None
    return int(entries_total_zip64), int(directory_size_zip64)


def _within_archive_metadata_budget(content: bytes) -> bool:
    budget: Final = _zip_directory_budget(content)
    return (
        budget is not None
        and budget[0] <= MAX_ARCHIVE_TOTAL_ENTRIES
        and budget[1] <= MAX_ARCHIVE_CENTRAL_DIRECTORY_BYTES
    )


def _reject_unsafe_member(name: str, info: zipfile.ZipInfo) -> None:
    path: Final = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or _WINDOWS_DRIVE_RE.match(name):
        raise ArchiveBuildError(f"the plugin archive contains an unsafe path: {name!r}")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ArchiveBuildError(f"the plugin archive contains a symlink: {name!r}")
    if info.compress_type not in _ALLOWED_COMPRESSION:
        raise ArchiveBuildError(f"the plugin archive uses an unsupported compression method: {name!r}")


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    # Bound decompression by metadata; malformed payloads that exceed advertised size are rejected.
    with archive.open(info) as member:
        data: Final = member.read(info.file_size + 1)
    if len(data) > info.file_size:
        raise ArchiveBuildError("the plugin source contains a ZIP member with inconsistent size metadata")
    return data


def _strip_prefix(entries: _Entries, prefix: str) -> _Entries:
    return MappingProxyType({name[len(prefix) :]: info for name, info in entries.items() if name.startswith(prefix)})


def _single_top_level_dir(names: Iterable[str]) -> str | None:
    parts: Final = tuple(PurePosixPath(name).parts for name in names)
    tops: Final = frozenset(path[0] for path in parts)
    if len(tops) != 1 or any(len(path) < 2 for path in parts):
        return None
    return f"{next(iter(tops))}/"


def _strip_repository_wrapper(entries: _Entries) -> _Entries:
    wrapper: Final = _single_top_level_dir(entries)
    if wrapper is None:
        raise ArchiveBuildError("the repository archive has no single top-level directory")
    return _strip_prefix(entries, wrapper)


def _locate_plugin(entries: _Entries, *, strip_top_level: bool, subdir: str | None) -> _Entries:
    unwrapped: Final = _strip_repository_wrapper(entries) if strip_top_level else entries
    scoped: Final = _strip_prefix(unwrapped, f"{subdir.strip('/')}/") if subdir else unwrapped
    if strip_top_level or PLUGIN_JSON_PATH in scoped:
        return scoped
    wrapper: Final = _single_top_level_dir(scoped)
    return scoped if wrapper is None else _strip_prefix(scoped, wrapper)


def _read_plugin_members(archive: zipfile.ZipFile, selected: _Entries) -> _Members:
    if PLUGIN_JSON_PATH not in selected:
        raise ArchiveBuildError(f"the plugin must contain {PLUGIN_JSON_PATH} at its root")
    if sum(info.file_size for info in selected.values()) > MAX_ARCHIVE_BYTES:
        raise ArchiveBuildError(f"the plugin is larger than {_size_limit_label()} uncompressed")
    for name, info in selected.items():
        _reject_unsafe_member(name, info)
    return MappingProxyType(
        {name: _Member(_read_member(archive, info), info.external_attr >> 16) for name, info in selected.items()}
    )


def _write_zip(members: _Members) -> bytes:
    output: Final = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as out:
        for name in sorted(members):
            member = members[name]
            info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if member.mode & 0o111 else 0o644) << 16
            out.writestr(info, member.content)
    return output.getvalue()


def normalize_archive(data: bytes, *, strip_top_level: bool, subdir: str | None = None) -> bytes:
    if not _within_archive_metadata_budget(data):
        raise ArchiveBuildError("the plugin source exceeds ZIP metadata safety limits")
    try:
        archive: Final = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise ArchiveBuildError("the plugin source is not a valid ZIP archive") from e
    with archive:
        # The end-of-directory count is attacker-controlled, so bound what was actually parsed.
        if len(archive.filelist) > MAX_ARCHIVE_TOTAL_ENTRIES:
            raise ArchiveBuildError("the plugin source exceeds ZIP metadata safety limits")
        entries: Final = MappingProxyType({info.filename: info for info in archive.infolist() if not info.is_dir()})
        selected: Final = _locate_plugin(entries, strip_top_level=strip_top_level, subdir=subdir)
        try:
            members: Final = _read_plugin_members(archive, selected)
        except (zipfile.BadZipFile, zlib.error) as e:
            raise ArchiveBuildError("the plugin source is a corrupt ZIP archive") from e
    # Fixed order, timestamps and modes keep a rebuilt zip byte-identical, so the recorded digest stays valid.
    result: Final = _write_zip(members)
    if len(result) > MAX_ARCHIVE_BYTES:
        raise ArchiveBuildError(f"the plugin archive is larger than {_size_limit_label()}")
    return result


def _repository_from_source(source: Mapping[str, str]) -> str:
    if source.get("source") == "github":
        return source.get("repo", "")
    parts: Final = urlsplit(source.get("url", ""))
    if (parts.hostname or "").lower() not in _GITHUB_HOSTS:
        raise ArchiveBuildError(
            "only GitHub-hosted sources can be mirrored by LiteLLM; "
            "host the plugin as a ZIP and register it with an 'archive' source instead"
        )
    return parts.path.strip("/").removesuffix(".git")


def _github_repo(source: Mapping[str, str]) -> str:
    repo: Final = _repository_from_source(source)
    if not _GITHUB_REPO_RE.match(repo):
        raise ArchiveBuildError(f"{repo!r} is not a valid GitHub repository (expected 'owner/repo')")
    return repo


async def _get(url: str, headers: Mapping[str, str] | None = None) -> bytes:
    try:
        response: Final = await async_safe_get(
            _http_client(),
            url,
            headers=dict(headers or {}),
            max_response_bytes=MAX_DOWNLOAD_BYTES,
        )
    except SSRFError as e:
        raise ArchiveBuildError(f"refusing to fetch {url}: {e}") from e
    except Exception as e:
        raise ArchiveBuildError(f"could not fetch {url}: {e}") from e
    if response.status_code != 200:
        raise ArchiveBuildError(f"fetching {url} returned HTTP {response.status_code}")
    if len(response.content) > MAX_DOWNLOAD_BYTES:
        raise ArchiveBuildError(f"{url} is larger than the {_size_limit_label()} plugin limit")
    return response.content


def _github_headers() -> Mapping[str, str]:
    token: Final = os.environ.get("GITHUB_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


async def _resolve_commit(repo: str, ref: str) -> str:
    if _COMMIT_SHA_RE.fullmatch(ref):
        return ref
    body: Final = await _get(
        f"https://api.github.com/repos/{repo}/commits/{ref}",
        {**_github_headers(), "Accept": "application/vnd.github.sha"},
    )
    commit: Final = body.decode().strip()
    if not _COMMIT_SHA_RE.fullmatch(commit):
        raise ArchiveBuildError(f"could not resolve {ref!r} in {repo} to a commit")
    return commit


async def _build_external_archive(source: Mapping[str, str]) -> BuiltArchive:
    raw: Final = await _get(source["url"])
    expected: Final = source.get("sha256")
    if expected and sha256_hex(raw).lower() != expected.lower():
        raise ArchiveBuildError("the downloaded archive does not match the sha256 in the plugin source")
    data: Final = await asyncio.to_thread(normalize_archive, raw, strip_top_level=False)
    return BuiltArchive(data, sha256_hex(data), None)


async def _build_github_archive(source: Mapping[str, str], pinned_ref: str | None) -> BuiltArchive:
    repo: Final = _github_repo(source)
    ref: Final = pinned_ref or source.get("ref") or "HEAD"
    if not _GITHUB_REF_RE.match(ref):
        raise ArchiveBuildError(f"{ref!r} is not a valid git ref")
    commit: Final = await _resolve_commit(repo, ref)
    codeload: Final = await _get(f"https://codeload.github.com/{repo}/zip/{commit}", _github_headers())
    data: Final = await asyncio.to_thread(
        normalize_archive,
        codeload,
        strip_top_level=True,
        subdir=source.get("path") if source.get("source") == "git-subdir" else None,
    )
    return BuiltArchive(data, sha256_hex(data), commit)


async def build_archive(source: Mapping[str, str], *, pinned_ref: str | None = None) -> BuiltArchive:
    if source.get("source") == "archive":
        return await _build_external_archive(source)
    return await _build_github_archive(source, pinned_ref)

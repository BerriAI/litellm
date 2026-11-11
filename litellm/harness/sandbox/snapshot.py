"""Workspace snapshots and FileChange construction.

A snapshot maps a workspace-relative POSIX path to the sha256 of its contents.
Diffing two snapshots tells us which files a turn created, modified or deleted;
`build_file_changes` turns that into `FileChange` events with unified diffs for
small text files.
"""

from __future__ import annotations

import asyncio
import difflib
import functools
import hashlib
import os
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from litellm.constants import HARNESS_MAX_DIFF_BYTES, HARNESS_SNAPSHOT_SKIP_DIRS
from litellm.harness.errors import HarnessError
from litellm.harness.types import FileChange, FileChangeKind

if TYPE_CHECKING:
    from litellm.harness.sandbox.base import Sandbox

# Files larger than this are left out of snapshots entirely.
HARNESS_SNAPSHOT_MAX_FILE_BYTES: Final = 50 * 1024 * 1024
# Upper bound on bytes read by capture_text_contents() for one turn.
HARNESS_SNAPSHOT_MAX_TOTAL_BYTES: Final = 16 * 1024 * 1024
_HASH_CHUNK_BYTES: Final = 1024 * 1024
_NO_NEWLINE_MARKER: Final = "\\ No newline at end of file\n"


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(functools.partial(fh.read, _HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_entry(root: str, dirpath: str, filename: str) -> tuple[str, str] | None:
    full = os.path.join(dirpath, filename)
    try:
        info = os.lstat(full)
    except OSError:
        return None
    if not os.path.isfile(full) or os.path.islink(full):
        return None
    if info.st_size > HARNESS_SNAPSHOT_MAX_FILE_BYTES:
        return None
    try:
        digest = _hash_file(full)
    except OSError:
        return None
    rel = os.path.relpath(full, root).replace(os.sep, "/")
    return rel, digest


def _walk_entries(root: str) -> Iterator[tuple[str, str]]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [  # mutable-ok: os.walk prunes only via in-place mutation of its dirnames list
            d for d in dirnames if d not in HARNESS_SNAPSHOT_SKIP_DIRS
        ]
        for filename in filenames:
            entry = _hash_entry(root, dirpath, filename)
            if entry is not None:
                yield entry


def snapshot_local_sync(root: str) -> Mapping[str, str]:
    """Hash every regular file under root. Symlinks are never followed."""
    return MappingProxyType(dict(_walk_entries(root)))


async def snapshot_local(root: str) -> Mapping[str, str]:
    """Async wrapper around snapshot_local_sync (runs in a worker thread)."""
    return await asyncio.to_thread(snapshot_local_sync, root)


def _change_kind(path: str, before: Mapping[str, str], after: Mapping[str, str]) -> FileChangeKind | None:
    if path not in before:
        return "created"
    if path not in after:
        return "deleted"
    if before[path] != after[path]:
        return "modified"
    return None


def diff_snapshots(
    before: Mapping[str, str], after: Mapping[str, str]
) -> list[tuple[str, FileChangeKind]]:  # mutable-ok: public sandbox helper; callers compare against a list
    """Return (path, kind) for every changed file, sorted by path."""
    kinds = ((path, _change_kind(path, before, after)) for path in sorted(frozenset(before) | frozenset(after)))
    return [  # mutable-ok: public sandbox helper returns a list
        (path, kind) for path, kind in kinds if kind is not None
    ]


def _as_text(data: bytes) -> str | None:
    if len(data) > HARNESS_MAX_DIFF_BYTES or b"\0" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def unified_diff(path: str, old: str | None, new: str | None) -> str:
    """Unified diff between two versions of path; None means the file is absent."""
    from_file = "/dev/null" if old is None else f"a/{path}"
    to_file = "/dev/null" if new is None else f"b/{path}"
    lines = difflib.unified_diff(
        (old or "").splitlines(keepends=True),
        (new or "").splitlines(keepends=True),
        fromfile=from_file,
        tofile=to_file,
    )
    return "".join(line if line.endswith("\n") else line + "\n" + _NO_NEWLINE_MARKER for line in lines)


async def _read_or_none(sandbox: Sandbox, path: str) -> bytes | None:
    try:
        return await sandbox.read(path)
    except (HarnessError, OSError):
        return None


async def capture_text_contents(sandbox: Sandbox, paths_hashes: Mapping[str, str]) -> Mapping[str, bytes]:
    """Read small text files before a turn so "modified"/"deleted" diffs can be built.

    Each kept file is <= HARNESS_MAX_DIFF_BYTES; every byte read (kept or not) counts
    toward HARNESS_SNAPSHOT_MAX_TOTAL_BYTES, after which capture stops.
    """
    captured: dict[str, bytes] = {}  # mutable-ok: async accumulator (awaits per read), frozen on return
    total = 0
    for path in sorted(paths_hashes):
        if total >= HARNESS_SNAPSHOT_MAX_TOTAL_BYTES:
            break
        data = await _read_or_none(sandbox, path)
        if data is None:
            continue
        total += len(data)
        if _as_text(data) is not None:
            captured[path] = data
    return MappingProxyType(captured)


async def _change_for(
    sandbox: Sandbox,
    path: str,
    kind: FileChangeKind,
    before_contents: Mapping[str, bytes],
) -> FileChange:
    old_bytes = before_contents.get(path)
    old = _as_text(old_bytes) if old_bytes is not None else None
    if kind == "deleted":
        diff = unified_diff(path, old, None) if old is not None else None
        return FileChange(path=path, kind=kind, diff=diff)
    new_bytes = await _read_or_none(sandbox, path)
    new = _as_text(new_bytes) if new_bytes is not None else None
    if new is None or (kind == "modified" and old is None):
        return FileChange(path=path, kind=kind, diff=None)
    return FileChange(
        path=path,
        kind=kind,
        diff=unified_diff(path, old if kind == "modified" else None, new),
    )


async def build_file_changes(
    sandbox: Sandbox,
    before: Mapping[str, str],
    after: Mapping[str, str],
    before_contents: Mapping[str, bytes] | None = None,
) -> list[FileChange]:  # mutable-ok: feeds the public Result.files list
    """FileChange per changed path. diff is None when it cannot be built as text."""
    contents: Mapping[str, bytes] = before_contents or MappingProxyType({})
    return [  # mutable-ok: feeds the public Result.files list
        await _change_for(sandbox, path, kind, contents) for path, kind in diff_snapshots(before, after)
    ]

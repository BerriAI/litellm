import io
import json
import stat
import zipfile
from typing import Final

import pytest

from litellm.proxy.anthropic_endpoints.claude_code_endpoints.skill_upload_archive import (
    MAX_SKILL_ARCHIVE_COMPRESSION_RATIO,
    build_uploaded_skill_archive,
)

MANIFEST: Final = b"---\nname: skill\ndescription: Notes\n---\nRead guide.md"


def _zip(files: tuple[tuple[str | zipfile.ZipInfo, bytes], ...], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buffer: Final = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=compression) as archive:
        for name, content in files:
            archive.writestr(name, content)
    return buffer.getvalue()


def _entry(name: str, mode: int, system: int = 3) -> zipfile.ZipInfo:
    entry: Final = zipfile.ZipInfo(name)
    entry.create_system = system
    entry.external_attr = mode << 16
    return entry


@pytest.mark.parametrize("prefix, manifest", [
    ("", MANIFEST), ("folder/", MANIFEST),
    ("folder/", b"---\nnotes: " + b"[" * 1000 + b"0" + b"]" * 1000 + b"\n---\nRead guide.md"),
])
@pytest.mark.parametrize("script, compression", [
    (b"#!/bin/sh\necho ready\n", zipfile.ZIP_DEFLATED), (b"echo ready\n" * 10000, zipfile.ZIP_STORED),
])
def test_installable_archive_preserves_files_and_safe_modes(
    prefix: str, manifest: bytes, script: bytes, compression: int,
) -> None:
    files: Final = (
        ("SKILL.md", manifest, 0o644, 3, 0o644),
        ("guide.md", b"Keep filenames unchanged", 0o666, 3, 0o644),
        ("scripts/run.sh", script, stat.S_IFREG | 0o7777, 3, 0o755),
        ("scripts/owner.sh", script, stat.S_IFREG | 0o700, 3, 0o744),
        ("scripts/no-type.sh", script, 0o755, 3, 0o755),
        ("scripts/dos.sh", script, stat.S_IFREG | 0o7777, 0, 0o644),
        ("scripts/dos-link.sh", script, stat.S_IFLNK | 0o777, 0, 0o644),
    )
    upload: Final = _zip(tuple(
        (_entry(prefix + name, mode, system), data) for name, data, mode, system, _ in files
    ) + ((prefix + ".DS_Store", b"Finder"), ("__MACOSX/._SKILL.md", b"Finder")))
    result: Final = build_uploaded_skill_archive(upload, "registered-name")
    expected: Final = {".claude-plugin/plugin.json": (json.dumps({"name": "registered-name"}).encode(), 0o644)} | {
        f"skills/registered-name/{name}": (data, mode) for name, data, _, _, mode in files
    }
    assert result is not None
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        assert {entry.filename: (archive.read(entry), entry.external_attr >> 16) for entry in archive.infolist()} == expected
        assert {entry.create_system for entry in archive.infolist()} == {3}
        assert {entry.compress_type for entry in archive.infolist()} == {compression}
        assert sum(entry.file_size for entry in archive.infolist()) <= len(result) * MAX_SKILL_ARCHIVE_COMPRESSION_RATIO
    assert build_uploaded_skill_archive(upload, "registered-name") == result

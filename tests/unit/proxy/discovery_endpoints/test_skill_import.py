import hashlib
import io
import stat
import struct
import zipfile
import zlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final, NoReturn

import httpx
import pytest

from litellm.proxy.discovery_endpoints.agent_skills_archive import (
    MAX_ARCHIVE_ENTRIES,
    MAX_ARCHIVE_UNPACKED_BYTES,
    MAX_SKILL_MANIFEST_BYTES,
)
from litellm.proxy.discovery_endpoints.skill_import import (
    MAX_IMPORT_BYTES,
    GitHubSkillSource,
    SkillImportError,
    fetch_skill_source,
    parse_skill_source,
    select_imported_skill,
)

MANIFEST: Final = b"---\nname: pdf-summarizer\ndescription: Summarize PDFs.\n---\n\nRead the PDF.\n"
SOURCE: Final = GitHubSkillSource("author", "repo", "HEAD", "pdf-summarizer", None)
NESTED: Final = "skills/pdf-summarizer"


def repository_zip(entries: tuple[tuple[str | zipfile.ZipInfo, bytes], ...]) -> bytes:
    buffer: Final = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in entries:
            entry: Final = zipfile.ZipInfo(name) if isinstance(name, str) else name
            archive.writestr(entry, content, compress_type=zipfile.ZIP_DEFLATED)
    return buffer.getvalue()


@pytest.mark.parametrize(("url", "skill", "ref", "path", "expected"), (
    ("https://skills.sh/author/repo/pdf-summarizer", None, "HEAD", None, SOURCE),
    ("https://github.com/author/repo.git/", "pdf-summarizer", "release/v2", NESTED,
     GitHubSkillSource("author", "repo", "release/v2", "pdf-summarizer", NESTED)),
    (f"https://github.com/author/repo/tree/release%2Fv2/{NESTED}", None, "HEAD", None,
     GitHubSkillSource("author", "repo", "release/v2", None, NESTED)),
    ("http://github.com/author/repo", None, "HEAD", None, 400),
    ("https://github.com.example/author/repo", None, "HEAD", None, 400),
    ("https://user:secret@github.com/author/repo", None, "HEAD", None, 400),
    ("https://github.com/author/repo?token=secret", None, "HEAD", None, 400),
    ("https://github.com/author/repo#fragment", None, "HEAD", None, 400),
    ("https://github.com/author%2Fother/repo", None, "HEAD", None, 400),
    ("https://[invalid/author/repo", None, "HEAD", None, 400),
    ("https://skills.sh/author/repo", None, "HEAD", None, 400),
    ("https://skills.sh/author/repo/pdf-summarizer", "other", "HEAD", None, 400),
    (f"https://github.com/author/repo/tree/main/{NESTED}", None, "other", None, 400),
    (f"https://github.com/author/repo/tree/main/{NESTED}", None, "HEAD", "other", 400),
))
def test_source_urls(
    url: str, skill: str | None, ref: str, path: str | None, expected: GitHubSkillSource | int
) -> None:
    result: Final = parse_skill_source(url, skill=skill, ref=ref, path=path)
    assert (result.status if isinstance(result, SkillImportError) else result) == expected


@dataclass(frozen=True, slots=True)
class DownloadStream(httpx.AsyncByteStream):
    overflow: bool

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"a" * MAX_IMPORT_BYTES
        if self.overflow:
            yield b"!"
            pytest.fail("The download kept reading after exceeding the byte limit")


@pytest.mark.asyncio
@pytest.mark.parametrize(("mode", "status"), (
    ("bounded", None), ("overflow", 413), ("redirect", 422), ("http-error", 422), ("timeout", 502),
))
async def test_public_download_transport(mode: str, status: int | None) -> None:
    credentials: Final = {
        "Authorization": "Bearer unit-secret",
        "Proxy-Authorization": "Basic unit-secret",
        "X-API-Key": "unit-secret",
        "Cookie": "session=unit-secret",
    }

    def respond(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "https://codeload.github.com/author/repo/zip/release%2Fv2"
        assert not frozenset(key.lower() for key in credentials).intersection(request.headers)
        if mode == "timeout":
            raise httpx.ReadTimeout("simulated timeout", request=request)
        return httpx.Response(
            307 if mode == "redirect" else 404 if mode == "http-error" else 200,
            headers={"Location": "https://untrusted.example/private", "Content-Length": "1"},
            stream=DownloadStream(mode == "overflow"),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(respond), headers=credentials, auth=httpx.BasicAuth("unit-user", "unit-secret"),
        follow_redirects=True, trust_env=False,
    ) as client:
        result: Final = await fetch_skill_source(GitHubSkillSource("author", "repo", "release/v2", None, None), client)
    if status is None:
        assert isinstance(result, bytes)
        actual: Final = (len(result), hashlib.sha256(result).hexdigest())
        expected: Final = (MAX_IMPORT_BYTES, hashlib.sha256(b"a" * MAX_IMPORT_BYTES).hexdigest())
        assert actual == expected
    else:
        assert isinstance(result, SkillImportError) and result.status == status


@pytest.mark.parametrize(("skill", "path", "root_name", "expected"), (
    (None, "", b"root", "root"),
    (None, "/", b"root", "root"),
    (None, "SKILL.md", b"root", 422),
    (None, NESTED, b"root", "pdf-summarizer"),
    ("pdf-summarizer", None, b"root", "pdf-summarizer"),
    ("pdf-summarizer", f"/{NESTED}/", b"pdf-summarizer", "pdf-summarizer"),
    (None, None, b"root", 422),
    ("missing", None, b"root", 422),
    ("pdf-summarizer", "", b"root", 422),
    ("pdf-summarizer", None, b"pdf-summarizer", 422),
    (None, "", None, 422),
))
def test_repository_selection(skill: str | None, path: str | None, root_name: bytes | None, expected: str | int) -> None:
    root: Final = (
        (("repo/SKILL.md", MANIFEST.replace(b"pdf-summarizer", root_name)),)
        if root_name is not None else ()
    )
    result: Final = select_imported_skill(
        repository_zip(root + ((f"repo/{NESTED}/SKILL.md", MANIFEST),)),
        GitHubSkillSource("author", "repo", "HEAD", skill, path),
    )
    assert (result.status if isinstance(result, SkillImportError) else result.declared_name) == expected


def test_corrupt_manifest_compression_returns_an_import_error() -> None:
    content: Final = repository_zip(((f"repo/{NESTED}/SKILL.md", MANIFEST),))
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entry: Final = archive.infolist()[0]
        filename_size, extra_size = struct.unpack_from("<HH", content, entry.header_offset + 26)
        offset: Final = entry.header_offset + 30 + filename_size + extra_size
        invalid: Final = content[:offset] + b"\xff" * entry.compress_size + content[offset + entry.compress_size:]
    result: Final = select_imported_skill(invalid, SOURCE)
    assert isinstance(result, SkillImportError) and result.status == 422


@pytest.mark.parametrize(("violation", "expected"), (
    ("path", 422), ("inside-link", 422), ("outside-link", "pdf-summarizer"), ("entries", 413), ("unpacked-bytes", 413),
))
def test_repository_validation_scope(violation: str, expected: str | int) -> None:
    paths: Final = {"path": "repo/../escape", "inside-link": f"repo/{NESTED}/run.sh"}
    entry: Final = zipfile.ZipInfo(paths.get(violation, "repo/CLAUDE.md"))
    entry.create_system = 3
    entry.external_attr = ((stat.S_IFLNK if violation.endswith("-link") else stat.S_IFREG) | 0o644) << 16
    payload: Final = b"x" * (MAX_ARCHIVE_UNPACKED_BYTES if violation == "unpacked-bytes" else 1)
    extra: Final = (
        tuple((f"repo/unrelated/{index}", b"x") for index in range(MAX_ARCHIVE_ENTRIES))
        if violation == "entries" else ((entry, payload),)
    )
    result: Final = select_imported_skill(
        repository_zip(((f"repo/{NESTED}/SKILL.md", MANIFEST),) + extra), SOURCE
    )
    assert (result.status if isinstance(result, SkillImportError) else result.declared_name) == expected


@pytest.mark.parametrize(("violation", "status"), (("path", 422), ("symlink", 422), ("manifest-bytes", 422)))
def test_candidate_metadata_is_validated_before_manifest_decompression(
    violation: str, status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    entry: Final = zipfile.ZipInfo(
        "repo/../pdf-summarizer/SKILL.md" if violation == "path" else f"repo/{NESTED}/SKILL.md"
    )
    entry.create_system = 3
    entry.external_attr = ((stat.S_IFLNK if violation == "symlink" else stat.S_IFREG) | 0o644) << 16
    content: Final = repository_zip(
        ((entry, b"x" * (MAX_SKILL_MANIFEST_BYTES + 1) if violation == "manifest-bytes" else MANIFEST),)
    )

    def reject_decompression(*args: object, **kwargs: object) -> NoReturn:
        pytest.fail("Invalid candidate metadata reached the decompressor before rejection")

    monkeypatch.setattr(zlib, "decompressobj", reject_decompression)
    result: Final = select_imported_skill(content, SOURCE)
    assert isinstance(result, SkillImportError) and result.status == status

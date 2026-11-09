import hashlib
import io
import stat
import zipfile
from typing import Final

import pytest

from litellm.proxy.discovery_endpoints.agent_skills_archive import (
    MAX_ARCHIVE_ENTRIES,
    agent_skill_frontmatter,
    build_agent_skill_archive,
    build_skill_archive,
)

MANIFEST = b"""---
name: pdf-summarizer
description: Summarize a PDF into an executive brief.
---

Read the PDF, then write the brief.
"""


def zip_bytes(files: dict[str | zipfile.ZipInfo, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def entries_of(content: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_single_top_level_folder_is_stripped_so_skill_md_sits_at_the_root():
    archive = build_skill_archive(
        zip_bytes(
            {
                "pdf-summarizer/SKILL.md": MANIFEST,
                "pdf-summarizer/reference.md": b"page citations",
                "pdf-summarizer/scripts/extract.py": b"print('hi')",
            }
        )
    )

    assert archive is not None
    assert entries_of(archive.content) == {
        "SKILL.md": MANIFEST,
        "reference.md": b"page citations",
        "scripts/extract.py": b"print('hi')",
    }


def test_digest_covers_the_repacked_bytes_and_is_stable_across_builds():
    upload = zip_bytes({"pdf-summarizer/SKILL.md": MANIFEST, "pdf-summarizer/reference.md": b"page citations"})

    first = build_skill_archive(upload)
    second = build_skill_archive(upload)

    assert first is not None and second is not None
    assert first.digest == f"sha256:{hashlib.sha256(first.content).hexdigest()}"
    assert first.content == second.content


def test_an_upload_that_is_already_flat_keeps_every_file_where_it_is():
    archive = build_skill_archive(zip_bytes({"SKILL.md": MANIFEST, "reference.md": b"page citations"}))

    assert archive is not None
    assert sorted(entries_of(archive.content)) == ["SKILL.md", "reference.md"]


def test_manifest_frontmatter_supplies_the_declared_name_and_description():
    archive = build_skill_archive(zip_bytes({"pdf-summarizer/SKILL.md": MANIFEST}))

    assert archive is not None
    assert archive.declared_name == "pdf-summarizer"
    assert archive.declared_description == "Summarize a PDF into an executive brief."


def test_a_manifest_without_frontmatter_declares_nothing():
    archive = build_skill_archive(zip_bytes({"pdf-summarizer/SKILL.md": b"just prose, no frontmatter"}))

    assert archive is not None
    assert archive.declared_name is None
    assert archive.declared_description is None


def test_a_manifest_buried_below_the_stripped_folder_is_not_installable():
    assert build_skill_archive(zip_bytes({"pdf-summarizer/nested/SKILL.md": MANIFEST})) is None


def test_an_upload_with_no_manifest_is_not_installable():
    assert build_skill_archive(zip_bytes({"pdf-summarizer/reference.md": b"page citations"})) is None


def test_a_non_zip_upload_is_not_installable():
    assert build_skill_archive(MANIFEST) is None


def test_a_path_traversal_entry_is_not_installable():
    assert build_skill_archive(zip_bytes({"SKILL.md": MANIFEST, "../escape.md": b"nope"})) is None


def test_an_upload_over_the_entry_cap_is_not_installable():
    files = {"pdf-summarizer/SKILL.md": MANIFEST} | {
        f"pdf-summarizer/file-{index}.md": b"x" for index in range(MAX_ARCHIVE_ENTRIES)
    }

    assert build_skill_archive(zip_bytes(files)) is None


def test_discovery_keeps_its_permission_and_finder_metadata_policy() -> None:
    executable: Final = zipfile.ZipInfo("scripts/run.sh")
    executable.create_system = 3
    executable.external_attr = (stat.S_IFREG | 0o7777) << 16
    result: Final = build_skill_archive(zip_bytes({"SKILL.md": MANIFEST, executable: b"echo ready\n"}))
    assert build_skill_archive(zip_bytes({"folder/SKILL.md": MANIFEST, "__MACOSX/._SKILL.md": b"Finder"})) is None
    assert result is not None
    with zipfile.ZipFile(io.BytesIO(result.content)) as archive:
        assert {entry.create_system for entry in archive.infolist()} == {3}
        assert {entry.external_attr >> 16 for entry in archive.infolist()} == {0o644}


@pytest.mark.parametrize(
    "manifest",
    (
        pytest.param(b"Instructions without frontmatter", id="missing-frontmatter"),
        pytest.param(b"---\n- pdf-summarizer\n- description\n---\n", id="non-mapping"),
        pytest.param(b"---\nname: [unterminated\ndescription: invalid\n---\n", id="invalid-yaml"),
        pytest.param(b"---\nname: pdf-summarizer\ndescription: !!python/object:unsafe {}\n---\n", id="unsafe-yaml-tag"),
        pytest.param(b"---\nname: Bad_Name\ndescription: invalid\n---\n", id="invalid-name"),
        pytest.param(b"---\nname: " + b"a" * 65 + b"\ndescription: too long\n---\n", id="name-too-long"),
        pytest.param(b"---\nname: pdf-summarizer\n---\n", id="missing-description"),
        pytest.param(b"---\nname: pdf-summarizer\ndescription:\n  - item\n---\n", id="non-string-description"),
        pytest.param(b"---\nname: pdf-summarizer\ndescription: '   '\n---\n", id="empty-description"),
        pytest.param(b"---\nname: pdf-summarizer\ndescription: " + b"a" * 1025 + b"\n---\n", id="description-too-long"),
        pytest.param(b"---\nname: pdf-summarizer\nname: other-skill\ndescription: ok\n---\n", id="duplicate-name"),
        pytest.param(b"---\nname: a\ndescription: ok\nlicense: MIT\nlicense: MIT\n---\n", id="duplicate-optional-key"),
    ),
)
def test_portable_upload_rejects_malformed_or_invalid_frontmatter(manifest: bytes) -> None:
    assert build_agent_skill_archive(zip_bytes({"SKILL.md": manifest})) is None


@pytest.mark.parametrize(
    "unsafe_path",
    ("../escape", "scripts/../../escape", "/absolute", r"scripts\escape", "C:/escape", "./scripts/run.sh"),
)
def test_portable_upload_rejects_unsafe_paths(unsafe_path: str) -> None:
    assert build_agent_skill_archive(zip_bytes({"SKILL.md": MANIFEST, unsafe_path: b"unsafe"})) is None


@pytest.mark.parametrize("file_type", (stat.S_IFLNK, stat.S_IFIFO, stat.S_IFCHR))
def test_portable_upload_rejects_links_and_special_files(file_type: int) -> None:
    entry: Final = zipfile.ZipInfo("scripts/run.sh")
    entry.create_system = 3
    entry.external_attr = (file_type | 0o777) << 16
    assert build_agent_skill_archive(zip_bytes({"SKILL.md": MANIFEST, entry: b"../../private"})) is None


@pytest.mark.parametrize("size, accepted", ((10 * 1024 * 1024, True), (10 * 1024 * 1024 + 1, False)))
def test_file_size_limit_applies_to_selected_skill_members(size: int, accepted: bool) -> None:
    content: Final = zip_bytes(
        {
            "selected/SKILL.md": MANIFEST,
            "selected/asset.bin": b"x" * size,
            "unrelated.bin": b"x" * (size + 1),
        }
    )
    assert (build_agent_skill_archive(content, skill_path="selected") is not None) == accepted


def test_portable_upload_rejects_entries_that_normalize_to_the_same_path() -> None:
    content: Final = zip_bytes({"SKILL.md": MANIFEST, "scripts//run.sh": b"first", "scripts/run.sh": b"second"})
    assert build_agent_skill_archive(content) is None


@pytest.mark.parametrize("bom", (b"", b"\xef\xbb\xbf"))
@pytest.mark.parametrize("newline", ("\n", "\r\n"))
@pytest.mark.parametrize(
    ("name", "description", "expected"),
    (
        ("on", "yes", ("on", "yes")),
        ("yes", "on", ("yes", "on")),
        ('"123"', "'true'", ("123", "true")),
        ("!!str 123", "!!str null", ("123", "null")),
        ('"null"', '"false"', ("null", "false")),
        ("date-reader", "2026-11-05", ("date-reader", "2026-11-05")),
        ("block-reader", "|-\n  yes", ("block-reader", "yes")),
    ),
)
def test_frontmatter_preserves_yaml12_strings_bom_newlines_and_extensions(
    bom: bytes, newline: str, name: str, description: str, expected: tuple[str, str]
) -> None:
    frontmatter: Final = (
        f"---\nname: {name}\ndescription: {description}\n"
        "metadata:\n"
        "  tags: [on, yes, 123]\n"
        "x-extension:\n"
        "  nested: {count: 1e3, enabled: false}\n"
        "---\nPreserve this body and its trailing spaces.  \n"
    )
    manifest: Final = bom + frontmatter.replace("\n", newline).encode()
    result: Final = build_agent_skill_archive(zip_bytes({"SKILL.md": manifest}))

    assert result is not None
    assert (result.declared_name, result.declared_description) == expected
    assert result.manifest_content == manifest


@pytest.mark.parametrize("field", ("name", "description"))
@pytest.mark.parametrize("scalar", ("123", "1e3", "012", "0o12", "0x12", "true", "FALSE", "null", "~"))
def test_frontmatter_rejects_non_string_required_scalars(field: str, scalar: str) -> None:
    manifest: Final = f"---\n{field}: {scalar}\n".encode() + (
        b"description: Instructions\n---\n" if field == "name" else b"name: pdf-summarizer\n---\n"
    )

    assert build_agent_skill_archive(zip_bytes({"SKILL.md": manifest})) is None


@pytest.mark.parametrize(
    "extension",
    (
        pytest.param("metadata:\n  owner: first\n  owner: second", id="duplicate-metadata-key"),
        pytest.param("metadata:\n  owners:\n    - {owner: first, owner: second}", id="duplicate-key-in-sequence"),
        pytest.param("metadata:\n  owner: !!python/object:unsafe {}", id="unsafe-nested-tag"),
        pytest.param("metadata:\n  count: !!int invalid", id="invalid-explicit-scalar"),
        pytest.param("metadata:\n  ? [complex, key]\n  : value", id="complex-key"),
        pytest.param("metadata: &loop\n  self: *loop", id="recursive-alias"),
        pytest.param("metadata: " + "[" * 40 + "nested" + "]" * 40, id="excessive-depth"),
        pytest.param("metadata: [unterminated", id="malformed-extension"),
    ),
)
def test_frontmatter_rejects_invalid_nested_metadata(extension: str) -> None:
    manifest: Final = f"---\nname: pdf-summarizer\ndescription: Instructions\n{extension}\n---\n".encode()

    assert build_agent_skill_archive(zip_bytes({"SKILL.md": manifest})) is None


@pytest.mark.parametrize("name", ("résumé", "мой-навык", "技能", "ｆｏｏ"))
def test_hub_registry_retains_its_ascii_name_policy(name: str) -> None:
    manifest: Final = f"---\nname: {name}\ndescription: Instructions\n---\n".encode()

    assert agent_skill_frontmatter(manifest) == {"name": name, "description": "Instructions"}
    assert build_agent_skill_archive(zip_bytes({"SKILL.md": manifest})) is None

import hashlib
import io
import zipfile

from litellm.proxy.discovery_endpoints.agent_skills_archive import (
    MAX_ARCHIVE_ENTRIES,
    build_skill_archive,
)

MANIFEST = b"""---
name: pdf-summarizer
description: Summarize a PDF into an executive brief.
---

Read the PDF, then write the brief.
"""


def zip_bytes(files: dict[str, bytes]) -> bytes:
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

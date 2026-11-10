import hashlib
import os

import pytest

from litellm import sandbox
from litellm.constants import HARNESS_MAX_DIFF_BYTES
from litellm.harness.sandbox.snapshot import (
    build_file_changes,
    capture_text_contents,
    diff_snapshots,
    snapshot_local,
    unified_diff,
)
from litellm.harness.types import FileChange


def test_diff_snapshots_kinds():
    before = {"a": "1", "b": "2", "c": "3"}
    after = {"a": "1", "b": "9", "d": "4"}
    assert diff_snapshots(before, after) == [
        ("b", "modified"),
        ("c", "deleted"),
        ("d", "created"),
    ]


async def test_snapshot_local_hashes_and_skips(tmp_path):
    (tmp_path / "x.txt").write_bytes(b"hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "y.txt").write_bytes(b"y")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "z.pyc").write_bytes(b"z")
    os.symlink(tmp_path / "x.txt", tmp_path / "link.txt")
    snap = await snapshot_local(str(tmp_path))
    assert snap == {
        "x.txt": hashlib.sha256(b"hello").hexdigest(),
        "sub/y.txt": hashlib.sha256(b"y").hexdigest(),
    }


def test_unified_diff_created():
    diff = unified_diff("f.txt", None, "one\n")
    assert diff.startswith("--- /dev/null\n+++ b/f.txt\n")
    assert "+one\n" in diff


async def test_created_modified_deleted_end_to_end(tmp_path):
    box = sandbox.local(tmp_path)
    try:
        await box.write("mod.txt", b"line1\nline2\n")
        await box.write("gone.txt", b"bye\n")
        await box.write("bin.dat", b"\x00\x01\x02")
        before = await box.snapshot()
        contents = await capture_text_contents(box, before)
        assert set(contents) == {"mod.txt", "gone.txt"}

        await box.write("mod.txt", b"line1\nchanged\n")
        await box.write("new.txt", b"fresh\n")
        await box.write("bin.dat", b"\x00\x09")
        os.remove(os.path.join(box.workdir, "gone.txt"))
        after = await box.snapshot()

        changes = await build_file_changes(box, before, after, contents)
        by_path = {c.path: c for c in changes}
        assert all(isinstance(c, FileChange) for c in changes)
        assert [(c.path, c.kind) for c in changes] == [
            ("bin.dat", "modified"),
            ("gone.txt", "deleted"),
            ("mod.txt", "modified"),
            ("new.txt", "created"),
        ]
        assert by_path["bin.dat"].diff is None
        assert "-line2\n" in by_path["mod.txt"].diff
        assert "+changed\n" in by_path["mod.txt"].diff
        assert "+fresh\n" in by_path["new.txt"].diff
        assert "-bye\n" in by_path["gone.txt"].diff
    finally:
        await box.close()


async def test_modified_without_before_contents_has_no_diff(tmp_path):
    box = sandbox.local(tmp_path)
    try:
        await box.write("f.txt", b"a\n")
        before = await box.snapshot()
        await box.write("f.txt", b"b\n")
        after = await box.snapshot()
        changes = await build_file_changes(box, before, after, None)
        assert changes == [FileChange(path="f.txt", kind="modified", diff=None)]
    finally:
        await box.close()


async def test_large_text_file_has_no_diff(tmp_path):
    box = sandbox.local(tmp_path)
    try:
        before = await box.snapshot()
        await box.write("big.txt", b"a" * (HARNESS_MAX_DIFF_BYTES + 1))
        after = await box.snapshot()
        changes = await build_file_changes(box, before, after, {})
        assert changes == [FileChange(path="big.txt", kind="created", diff=None)]
    finally:
        await box.close()


async def test_capture_respects_total_cap(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "litellm.harness.sandbox.snapshot.HARNESS_SNAPSHOT_MAX_TOTAL_BYTES", 10
    )
    box = sandbox.local(tmp_path)
    try:
        await box.write("a.txt", b"x" * 8)
        await box.write("b.txt", b"x" * 8)
        await box.write("c.txt", b"x" * 8)
        captured = await capture_text_contents(box, await box.snapshot())
        assert set(captured) == {"a.txt", "b.txt"}
    finally:
        await box.close()


@pytest.mark.parametrize("data", [b"\xff\xfe bad utf8", b"has\x00nul"])
async def test_capture_skips_binary(tmp_path, data):
    box = sandbox.local(tmp_path)
    try:
        await box.write("f", data)
        assert await capture_text_contents(box, {"f": "h"}) == {}
    finally:
        await box.close()

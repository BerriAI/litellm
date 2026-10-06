"""
Regression tests for the SERVER_ROOT_PATH UI rewrite (issue #44888).

Multiple uvicorn workers import proxy_server concurrently; the rewrite used
an in-place truncate-write, so one worker could read another worker's
half-written file and persist the truncated (zero byte) content. The rewrite
now skips files without the asset prefix and writes atomically via a temp
file + os.replace.
"""

import os

from litellm.proxy.proxy_server import _rewrite_ui_files_for_server_root_path

ASSET_PREFIX = "/litellm-asset-prefix"


def _make_ui_dir(tmp_path):
    ui_dir = tmp_path / "ui"
    (ui_dir / "_next").mkdir(parents=True)
    (ui_dir / "_next" / "chunk.js").write_text(f'fetch("{ASSET_PREFIX}/api")', encoding="utf-8")
    (ui_dir / "index.html").write_text(f'<script src="{ASSET_PREFIX}/_next/chunk.js"></script>', encoding="utf-8")
    (ui_dir / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + ASSET_PREFIX.encode())
    (ui_dir / "plain.txt").write_text("no prefix here", encoding="utf-8")
    return ui_dir


def test_rewrites_asset_prefix(tmp_path):
    ui_dir = _make_ui_dir(tmp_path)
    _rewrite_ui_files_for_server_root_path(str(ui_dir), "/litellm")
    assert (ui_dir / "index.html").read_text(encoding="utf-8") == '<script src="/litellm/_next/chunk.js"></script>'
    assert (ui_dir / "_next" / "chunk.js").read_text(encoding="utf-8") == 'fetch("/litellm/api")'


def test_binary_files_untouched(tmp_path):
    ui_dir = _make_ui_dir(tmp_path)
    _rewrite_ui_files_for_server_root_path(str(ui_dir), "/litellm")
    assert ASSET_PREFIX.encode() in (ui_dir / "logo.png").read_bytes()


def test_no_temp_files_left_behind(tmp_path):
    ui_dir = _make_ui_dir(tmp_path)
    _rewrite_ui_files_for_server_root_path(str(ui_dir), "/litellm")
    leftovers = [p for p in ui_dir.rglob("*") if p.name.endswith(".tmp-server-root-path")]
    assert leftovers == []


def test_second_run_is_a_no_op(tmp_path):
    # Regression for the reported zero-byte UI files: the first boot rewrites,
    # and a restart (second run) must not touch already-processed files at all.
    ui_dir = _make_ui_dir(tmp_path)
    _rewrite_ui_files_for_server_root_path(str(ui_dir), "/litellm")
    snapshot = {p: p.read_bytes() for p in ui_dir.rglob("*") if p.is_file()}
    for path in snapshot:
        os.utime(path, (1000, 1000))  # age every file so a rewrite is detectable
    _rewrite_ui_files_for_server_root_path(str(ui_dir), "/litellm")
    for path, content in snapshot.items():
        assert path.read_bytes() == content
        assert path.stat().st_mtime == 1000  # untouched => not rewritten

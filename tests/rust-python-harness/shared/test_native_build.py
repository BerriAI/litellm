from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Final

import pytest

from . import native_build


def test_needs_rebuild_when_bridge_is_missing() -> None:
    assert native_build.needs_rebuild(None, 1.0)


def test_needs_rebuild_when_sources_are_newer_than_bridge() -> None:
    assert native_build.needs_rebuild(1.0, 2.0)


def test_fresh_bridge_with_older_sources_needs_no_rebuild() -> None:
    assert not native_build.needs_rebuild(2.0, 1.0)


def test_bridge_without_rust_sources_needs_no_rebuild() -> None:
    assert not native_build.needs_rebuild(2.0, None)


def test_newest_source_mtime_tracks_rust_sources_and_skips_target(tmp_path: Final) -> None:
    source: Final = tmp_path / "litellm-rust" / "crates" / "bridge" / "src"
    source.mkdir(parents=True)
    (source / "lib.rs").write_text("fn main() {}\n")
    os.utime(source / "lib.rs", (1_000, 1_000))
    manifest: Final = tmp_path / "litellm-rust" / "crates" / "bridge" / "Cargo.toml"
    manifest.write_text("[package]\n")
    os.utime(manifest, (2_000, 2_000))
    lockfile: Final = tmp_path / "litellm-rust" / "Cargo.lock"
    lockfile.write_text("")
    os.utime(lockfile, (1_500, 1_500))
    target: Final = tmp_path / "litellm-rust" / "target" / "debug" / "junk.rs"
    target.parent.mkdir(parents=True)
    target.write_text("fn main() {}\n")
    os.utime(target, (9_999, 9_999))

    assert native_build._newest_source_mtime(tmp_path) == 2_000.0


def test_newest_source_mtime_is_none_without_rust_workspace(tmp_path: Final) -> None:
    assert native_build._newest_source_mtime(tmp_path) is None


def test_ensure_trace_bridge_rebuilds_when_stale(
    tmp_path: Final, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    native: Final = tmp_path / "_native.abi3.so"
    native.write_bytes(b"")
    os.utime(native, (1_000, 1_000))
    source: Final = tmp_path / "litellm-rust" / "crates" / "bridge" / "src" / "lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("fn main() {}\n")
    os.utime(source, (2_000, 2_000))
    state: Final = SimpleNamespace(rebuilt=False)

    def fake_rebuild(repo_root: object) -> tuple[bool, str]:
        state.rebuilt = True
        return True, ""

    monkeypatch.setattr(native_build, "_native_module_path", lambda: native)
    monkeypatch.setattr(native_build, "_rebuild", fake_rebuild)
    monkeypatch.setattr(native_build, "_drop_imported_bridge", lambda: None)
    monkeypatch.setattr(native_build, "get_native_bridge", lambda: SimpleNamespace(_trace=object()))

    assert native_build.ensure_trace_bridge(tmp_path) is None
    assert state.rebuilt is True
    assert "Rebuilding native Rust bridge" in capsys.readouterr().out


def test_ensure_trace_bridge_reports_failed_rebuild(tmp_path: Final, monkeypatch: pytest.MonkeyPatch) -> None:
    source: Final = tmp_path / "litellm-rust" / "crates" / "bridge" / "src" / "lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("fn main() {}\n")

    monkeypatch.setattr(native_build, "_native_module_path", lambda: None)
    monkeypatch.setattr(native_build, "_rebuild", lambda repo_root: (False, "boom"))

    message: Final = native_build.ensure_trace_bridge(tmp_path)

    assert message is not None
    assert "rebuild failed" in message
    assert "boom" in message


def test_ensure_trace_bridge_flags_missing_trace_feature_without_rebuild(
    tmp_path: Final, monkeypatch: pytest.MonkeyPatch
) -> None:
    native: Final = tmp_path / "_native.abi3.so"
    native.write_bytes(b"")
    os.utime(native, (9_999, 9_999))
    source: Final = tmp_path / "litellm-rust" / "crates" / "bridge" / "src" / "lib.rs"
    source.parent.mkdir(parents=True)
    source.write_text("fn main() {}\n")
    os.utime(source, (1_000, 1_000))
    state: Final = SimpleNamespace(rebuilt=False)

    def fake_rebuild(repo_root: object) -> tuple[bool, str]:
        state.rebuilt = True
        return True, ""

    monkeypatch.setattr(native_build, "_native_module_path", lambda: native)
    monkeypatch.setattr(native_build, "_rebuild", fake_rebuild)
    monkeypatch.setattr(native_build, "get_native_bridge", lambda: SimpleNamespace(_trace=None))

    message: Final = native_build.ensure_trace_bridge(tmp_path)

    assert message is not None
    assert "_trace" in message
    assert state.rebuilt is False

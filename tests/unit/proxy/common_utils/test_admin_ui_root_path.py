import os
import shutil
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.testclient import TestClient

from litellm.proxy.common_utils.admin_ui_root_path import (
    RENDERED_UI_DIRNAME,
    RenderedUI,
    RenderUnavailable,
    UIRenderResult,
    render_ui_for_root_path,
    restructure_ui_html_files,
)

CONFIG_PLACEHOLDER: Final = "/litellm/.well-known/litellm-ui-config"
ASSET_PLACEHOLDER: Final = "/litellm-asset-prefix"
FAVICON_BYTES: Final = bytes(range(256)) + b"/litellm-asset-prefix\xff\xfe"
WOFF2_BYTES: Final = b"wOF2" + ASSET_PLACEHOLDER.encode() + b"\x00\xff"
APP_JS: Final = f'a="{ASSET_PLACEHOLDER}";b=`${{r}}{CONFIG_PLACEHOLDER}`;c="{CONFIG_PLACEHOLDER}";d="{ASSET_PLACEHOLDER}/_next";e="{ASSET_PLACEHOLDER}/.well-known/litellm-ui-config"'
EXPECTED_FOO_LITELLM_APP_JS: Final = (
    'a="/foo/litellm";b=`${r}/foo/litellm/.well-known/litellm-ui-config`;c="/foo/litellm/.well-known/litellm-ui-config";'
    'd="/foo/litellm/_next";e="/foo/litellm/.well-known/litellm-ui-config"'
)


def _write(path: Path, content: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if isinstance(content, bytes) else content.encode())


def _build_source(root: Path) -> Path:
    source: Final = root / "out"
    _write(source / "index.html", f'<script src="{ASSET_PLACEHOLDER}/_next/static/chunks/app.js"></script>')
    _write(source / "login.html", f'<link href="{ASSET_PLACEHOLDER}/_next/login.css">')
    _write(source / "_next" / "static" / "chunks" / "app.js", APP_JS)
    _write(source / "favicon.ico", FAVICON_BYTES)
    _write(source / "fonts" / "a.woff2", WOFF2_BYTES)
    return source


def _snapshot(directory: Path) -> dict[str, bytes]:
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file() and RENDERED_UI_DIRNAME not in path.relative_to(directory).parts
    }


def _rendered_directory(result: UIRenderResult) -> Path:
    assert isinstance(result, RenderedUI), result
    return result.directory


def test_restructure_ui_html_files_handles_nested_routes(tmp_path: Path):
    ui_root: Final = tmp_path / "ui"
    _write(ui_root / "home.html", "home")
    _write(ui_root / "mcp" / "oauth" / "callback.html", "callback")
    _write(ui_root / "existing" / "index.html", "keep")
    _write(ui_root / "_next" / "ignore.html", "asset")
    _write(ui_root / "litellm-asset-prefix" / "ignore.html", "asset")

    restructure_ui_html_files(ui_root)

    assert not (ui_root / "home.html").exists()
    assert (ui_root / "home" / "index.html").read_text() == "home"
    assert not (ui_root / "mcp" / "oauth" / "callback.html").exists()
    assert (ui_root / "mcp" / "oauth" / "callback" / "index.html").read_text() == "callback"
    assert (ui_root / "existing" / "index.html").read_text() == "keep"
    assert (ui_root / "_next" / "ignore.html").read_text() == "asset"
    assert (ui_root / "litellm-asset-prefix" / "ignore.html").read_text() == "asset"


def test_ui_extensionless_route_requires_restructure(tmp_path: Path):
    ui_root: Final = tmp_path / "ui"
    _write(ui_root / "index.html", "index")
    _write(ui_root / "login.html", "login")

    fastapi_app: Final = FastAPI()
    fastapi_app.mount("/ui", StaticFiles(directory=str(ui_root), html=True), name="ui")
    client: Final = TestClient(fastapi_app)

    assert client.get("/ui/login.html").status_code == 200
    assert client.get("/ui/login").status_code == 404

    restructure_ui_html_files(ui_root)

    response: Final = client.get("/ui/login")
    assert response.status_code == 200
    assert "login" in response.text


def test_concurrent_workers_render_the_same_complete_directory(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    worker_count: Final = 16
    barrier: Final = threading.Barrier(worker_count)

    def worker() -> tuple[UIRenderResult, dict[str, bytes]]:
        barrier.wait()
        result: Final = render_ui_for_root_path(source, "/foo/litellm")
        return result, _snapshot(_rendered_directory(result))

    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        outcomes: Final = tuple(f.result() for f in tuple(pool.submit(worker) for _ in range(worker_count)))

    directories: Final = {_rendered_directory(result) for result, _ in outcomes}
    assert len(directories) == 1
    expected_render: Final = {
        "index.html": b'<script src="/foo/litellm/_next/static/chunks/app.js"></script>',
        "login/index.html": b'<link href="/foo/litellm/_next/login.css">',
        "_next/static/chunks/app.js": EXPECTED_FOO_LITELLM_APP_JS.encode(),
        "favicon.ico": FAVICON_BYTES,
        "fonts/a.woff2": WOFF2_BYTES,
    }
    assert all(snapshot == expected_render for _, snapshot in outcomes)
    assert [path.name for path in (source / RENDERED_UI_DIRNAME).iterdir()] == [next(iter(directories)).name]


def test_root_path_ending_in_litellm_is_not_prefixed_twice(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    source_before: Final = _snapshot(source)

    first: Final = _rendered_directory(render_ui_for_root_path(source, "/foo/litellm"))
    second: Final = _rendered_directory(render_ui_for_root_path(source, "/foo/litellm"))

    app_js: Final = (first / "_next" / "static" / "chunks" / "app.js").read_text()
    assert first == second
    assert app_js == EXPECTED_FOO_LITELLM_APP_JS
    assert _snapshot(source) == source_before


def test_changing_root_path_renders_from_the_untouched_source(tmp_path: Path):
    source: Final = _build_source(tmp_path)

    dir_a: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))
    dir_b: Final = _rendered_directory(render_ui_for_root_path(source, "/b"))

    index_b: Final = (dir_b / "index.html").read_text()
    assert dir_a != dir_b
    assert "/b/_next" in index_b
    assert "/a/_next" not in index_b
    assert "/a/_next" in (dir_a / "index.html").read_text()
    assert ASSET_PLACEHOLDER in (source / "index.html").read_text()


@pytest.mark.parametrize("root_path", ["", "/"])
def test_empty_or_slash_root_path_serves_source_without_writing(tmp_path: Path, root_path: str):
    source: Final = _build_source(tmp_path)
    source_before: Final = _snapshot(source)

    result: Final = render_ui_for_root_path(source, root_path)

    assert result == RenderedUI(source)
    assert _snapshot(source) == source_before
    assert not (source / RENDERED_UI_DIRNAME).exists()


def test_rendered_directory_serves_extensionless_routes(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    directory: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))
    fastapi_app: Final = FastAPI()
    fastapi_app.mount("/ui", StaticFiles(directory=str(directory), html=True), name="ui")
    client: Final = TestClient(fastapi_app)

    response: Final = client.get("/ui/login")

    assert response.status_code == 200
    assert "/a/_next/login.css" in response.text
    assert (source / "login.html").is_file()


def test_changed_source_file_renders_into_a_new_directory(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    before: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))

    _write(source / "_next" / "static" / "chunks" / "app.js", f'{APP_JS};z="{ASSET_PLACEHOLDER}/new"')
    after: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))

    assert before != after
    assert 'z="/a/new"' in (after / "_next" / "static" / "chunks" / "app.js").read_text()
    assert 'z="/a/new"' not in (before / "_next" / "static" / "chunks" / "app.js").read_text()


def test_restructure_leaves_in_source_render_untouched(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    rendered: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))
    before: Final = _snapshot(rendered)

    restructure_ui_html_files(source)

    assert _snapshot(rendered) == before
    assert (source / "login" / "index.html").is_file()


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need elevated privileges on Windows")
def test_symlinked_directory_is_rendered_with_its_content(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    real_next: Final = source / "real_next"
    _write(real_next / "chunk.js", f'x="{ASSET_PLACEHOLDER}/y"')
    (source / "linked").symlink_to(real_next, target_is_directory=True)

    directory: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))

    assert (directory / "linked" / "chunk.js").read_text() == 'x="/a/y"'


def test_binary_files_are_copied_verbatim_while_text_is_substituted(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    _write(source / "img.PNG", FAVICON_BYTES)

    directory: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))

    assert (directory / "img.PNG").read_bytes() == FAVICON_BYTES
    assert (directory / "fonts" / "a.woff2").read_bytes() == WOFF2_BYTES
    assert "/a/_next" in (directory / "index.html").read_text()


def test_vanishing_source_file_during_key_computation_is_reported_unavailable(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    (source / "dangling.js").symlink_to(source / "missing")

    result: Final = render_ui_for_root_path(source, "/a")

    assert isinstance(result, RenderUnavailable)


def test_render_lives_inside_the_source_directory_and_is_reused(tmp_path: Path):
    source: Final = _build_source(tmp_path)

    first: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))
    second: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))

    assert first == second
    assert first.parent == source / RENDERED_UI_DIRNAME
    assert not (first / RENDERED_UI_DIRNAME).exists()
    assert {path.name for path in tmp_path.iterdir()} == {"out"}


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need elevated privileges on Windows")
def test_symlinks_pointing_outside_the_source_are_not_published(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    private: Final = tmp_path / "private"
    _write(private / "secret.txt", "secret")
    _write(tmp_path / "outside.txt", "outside")
    (source / "linked").symlink_to(private, target_is_directory=True)
    (source / "leak.txt").symlink_to(tmp_path / "outside.txt")

    directory: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))

    assert set(_snapshot(directory)) == {
        "index.html",
        "login/index.html",
        "_next/static/chunks/app.js",
        "favicon.ico",
        "fonts/a.woff2",
    }


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need elevated privileges on Windows")
def test_symlink_planted_at_the_render_target_is_not_served(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    key: Final = _rendered_directory(render_ui_for_root_path(source, "/a")).name
    shutil.rmtree(source / RENDERED_UI_DIRNAME)
    attacker: Final = tmp_path / "attacker"
    _write(attacker / "index.html", "attacker")
    (source / RENDERED_UI_DIRNAME).mkdir(mode=0o755)
    (source / RENDERED_UI_DIRNAME / key).symlink_to(attacker, target_is_directory=True)

    result: Final = render_ui_for_root_path(source, "/a")

    assert isinstance(result, RenderUnavailable)
    assert (attacker / "index.html").read_text() == "attacker"


@pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="permission bits are not enforced on Windows or for root",
)
def test_group_or_world_writable_render_root_is_refused(tmp_path: Path):
    source: Final = _build_source(tmp_path)
    output_root: Final = source / RENDERED_UI_DIRNAME
    output_root.mkdir()
    output_root.chmod(0o777)

    result: Final = render_ui_for_root_path(source, "/a")

    assert isinstance(result, RenderUnavailable)
    assert list(output_root.iterdir()) == []


@pytest.mark.skipif(sys.platform == "win32", reason="permission bits are not enforced on Windows")
def test_rendered_ui_is_not_group_or_world_writable_under_a_permissive_umask(tmp_path: Path):
    previous_umask: Final = os.umask(0o002)
    try:
        source: Final = _build_source(tmp_path)
        rendered: Final = _rendered_directory(render_ui_for_root_path(source, "/a"))
    finally:
        os.umask(previous_umask)

    paths: Final = (rendered, *rendered.rglob("*"))
    assert len(paths) > 5
    assert [path for path in paths if path.stat().st_mode & 0o022] == []
    assert [path for path in paths if path.stat().st_mode & 0o444 != 0o444] == []

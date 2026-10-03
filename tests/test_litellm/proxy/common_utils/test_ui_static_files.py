"""
Unit tests for UiStaticFiles, the /ui static mount with <route>.html fallback.

Regression tests for https://github.com/BerriAI/litellm/issues/24037: a Next.js
export without ``trailingSlash: true`` ships ``chat.html`` plus an index-less
``chat/`` data directory, which vanilla StaticFiles(html=True) turns into a 404
on direct navigation to /ui/chat. The fallback must serve that layout without
writing anything to disk, so read-only deployments work.
"""

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from litellm.proxy.common_utils.ui_static_files import UiStaticFiles


def make_client(ui_dir) -> TestClient:
    app = Starlette()
    app.mount("/ui", UiStaticFiles(directory=str(ui_dir), html=True), name="ui")
    return TestClient(app)


@pytest.fixture
def flat_export(tmp_path):
    (tmp_path / "index.html").write_text("<h1>root</h1>")
    (tmp_path / "chat.html").write_text("<h1>chat page</h1>")
    chat_dir = tmp_path / "chat"
    chat_dir.mkdir()
    (chat_dir / "__next.chat.__PAGE__.txt").write_text("rsc payload")
    (tmp_path / "model_hub.html").write_text("<h1>model hub page</h1>")
    return tmp_path


def test_route_shadowed_by_indexless_directory_serves_html(flat_export):
    client = make_client(flat_export)

    response = client.get("/ui/chat")

    assert response.status_code == 200
    assert "chat page" in response.text


def test_route_with_trailing_slash_serves_html(flat_export):
    client = make_client(flat_export)

    response = client.get("/ui/chat/")

    assert response.status_code == 200
    assert "chat page" in response.text


def test_route_without_directory_serves_html(flat_export):
    client = make_client(flat_export)

    response = client.get("/ui/model_hub")

    assert response.status_code == 200
    assert "model hub page" in response.text


def test_serving_flat_export_writes_nothing_to_disk(flat_export):
    client = make_client(flat_export)
    before = sorted(p.relative_to(flat_export) for p in flat_export.rglob("*"))

    client.get("/ui/chat")
    client.get("/ui/model_hub")

    after = sorted(p.relative_to(flat_export) for p in flat_export.rglob("*"))
    assert after == before


def test_restructured_export_still_served(tmp_path):
    (tmp_path / "index.html").write_text("<h1>root</h1>")
    chat_dir = tmp_path / "chat"
    chat_dir.mkdir()
    (chat_dir / "index.html").write_text("<h1>restructured chat</h1>")
    client = make_client(tmp_path)

    response = client.get("/ui/chat")

    assert response.status_code == 200
    assert "restructured chat" in response.text


def test_root_serves_index(flat_export):
    client = make_client(flat_export)

    response = client.get("/ui/")

    assert response.status_code == 200
    assert "root" in response.text


def test_unknown_route_returns_404(flat_export):
    client = make_client(flat_export)

    response = client.get("/ui/does-not-exist")

    assert response.status_code == 404


def test_asset_files_still_served_directly(flat_export):
    (flat_export / "next.svg").write_text("<svg></svg>")
    client = make_client(flat_export)

    response = client.get("/ui/next.svg")

    assert response.status_code == 200
    assert response.text == "<svg></svg>"

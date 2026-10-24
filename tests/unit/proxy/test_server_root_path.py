import os
from unittest import mock

from litellm.proxy import utils


# Test the utility function logic
def test_get_server_root_path_unset():
    """
    Test that get_server_root_path returns empty string when SERVER_ROOT_PATH is unset
    """
    with mock.patch.dict(os.environ, {}, clear=True):
        # We need to make sure SERVER_ROOT_PATH is not in env
        if "SERVER_ROOT_PATH" in os.environ:
            del os.environ["SERVER_ROOT_PATH"]

        root_path = utils.get_server_root_path()
        assert root_path == "", "Should return empty string when unset to allow X-Forwarded-Prefix"


def test_get_server_root_path_set():
    """
    Test that get_server_root_path returns the value when SERVER_ROOT_PATH is set
    """
    with mock.patch.dict(os.environ, {"SERVER_ROOT_PATH": "/my-path"}, clear=True):
        root_path = utils.get_server_root_path()
        assert root_path == "/my-path", "Should return the set value"


def test_get_server_root_path_empty_string():
    """
    Test that get_server_root_path returns empty string when SERVER_ROOT_PATH is explicitly empty
    """
    with mock.patch.dict(os.environ, {"SERVER_ROOT_PATH": ""}, clear=True):
        root_path = utils.get_server_root_path()
        assert root_path == "", "Should return empty string when explicitly set to empty"


# Integration test simulation for FastAPI app initialization
def test_fastapi_app_initialization_mock():
    """
    Simulate how proxy_server.py initializes FastAPI app with the root_path.
    We don't import proxy_server because it has global side effects/singletons.
    Instead we verify the logic flow.
    """
    from fastapi import FastAPI

    # CASE 1: Proxy Mode (Unset)
    with mock.patch.dict(os.environ, {}, clear=True):
        if "SERVER_ROOT_PATH" in os.environ:
            del os.environ["SERVER_ROOT_PATH"]

        server_root_path = utils.get_server_root_path()
        app = FastAPI(root_path=server_root_path)
        assert app.root_path == ""

    # CASE 2: Direct Mode (Set)
    with mock.patch.dict(os.environ, {"SERVER_ROOT_PATH": "/custom-root"}, clear=True):
        server_root_path = utils.get_server_root_path()
        app = FastAPI(root_path=server_root_path)
        assert app.root_path == "/custom-root"


SAMPLE_PRISTINE_UI_CONTENT = (
    "let t=h?`${h}/litellm/.well-known/litellm-ui-config`:"
    '"/litellm/.well-known/litellm-ui-config";'
    '<link rel="preload" href="/litellm-asset-prefix/_next/static/css/app.css" />'
)


def test_rewrite_ui_content_first_rewrite():
    """Test A: First rewrite from pristine path with /foo/litellm."""
    result = utils.rewrite_ui_content(
        content=SAMPLE_PRISTINE_UI_CONTENT,
        server_root_path="/foo/litellm",
    )
    assert "`${h}/foo/litellm/.well-known/litellm-ui-config`" in result
    assert ':"/foo/litellm/.well-known/litellm-ui-config"' in result
    assert 'href="/foo/litellm/_next/static/css/app.css"' in result


def test_rewrite_ui_content_repeated_rewrite():
    """
    Test B: Repeated rewrite must be idempotent.
    Applying rewrite twice with /foo/litellm must NOT produce /foo/foo/litellm.
    """
    first = utils.rewrite_ui_content(
        content=SAMPLE_PRISTINE_UI_CONTENT,
        server_root_path="/foo/litellm",
    )
    second = utils.rewrite_ui_content(
        content=first,
        server_root_path="/foo/litellm",
    )
    assert second == first
    assert "/foo/foo/litellm" not in second
    assert "`${h}/foo/litellm/.well-known/litellm-ui-config`" in second
    assert ':"/foo/litellm/.well-known/litellm-ui-config"' in second


def test_rewrite_ui_content_default_path():
    """Test C: SERVER_ROOT_PATH=/litellm behavior must remain unchanged."""
    first = utils.rewrite_ui_content(
        content=SAMPLE_PRISTINE_UI_CONTENT,
        server_root_path="/litellm",
    )
    second = utils.rewrite_ui_content(
        content=first,
        server_root_path="/litellm",
    )
    assert "`${h}/litellm/.well-known/litellm-ui-config`" in first
    assert ':"/litellm/.well-known/litellm-ui-config"' in first
    assert second == first


def test_rewrite_ui_content_nested_and_edge_paths():
    """Test other supported root paths: /api/litellm, nested, custom, and slash normalization."""
    for root_path in [
        "/api/litellm",
        "/app/gateway/litellm",
        "/custom-proxy",
        "/nested/litellm/",
        "prefix/litellm",
    ]:
        first = utils.rewrite_ui_content(
            content=SAMPLE_PRISTINE_UI_CONTENT,
            server_root_path=root_path,
        )
        second = utils.rewrite_ui_content(
            content=first,
            server_root_path=root_path,
        )
        assert second == first, f"Failed idempotency for {root_path}"

        clean_path = "/" + root_path.strip("/")
        expected_endpoint = f"{clean_path}/.well-known/litellm-ui-config"
        assert f':"{expected_endpoint}"' in first, f"Expected endpoint {expected_endpoint} not in output"


def test_rewrite_ui_content_healing_and_reconfiguration():
    """
    Test healing previously corrupted paths (e.g. from buggy restarts)
    and smoothly switching between different configured root paths.
    """
    corrupted = (
        'let t=h?`${h}/foo/foo/litellm/.well-known/litellm-ui-config`:"/foo/foo/litellm/.well-known/litellm-ui-config";'
    )
    healed = utils.rewrite_ui_content(
        content=corrupted,
        server_root_path="/foo/litellm",
    )
    assert "`${h}/foo/litellm/.well-known/litellm-ui-config`" in healed
    assert ':"/foo/litellm/.well-known/litellm-ui-config"' in healed

    reconfigured = utils.rewrite_ui_content(
        content=healed,
        server_root_path="/bar/litellm",
    )
    assert "`${h}/bar/litellm/.well-known/litellm-ui-config`" in reconfigured
    assert ':"/bar/litellm/.well-known/litellm-ui-config"' in reconfigured
    assert utils.rewrite_ui_content(reconfigured, "/bar/litellm") == reconfigured


def test_rewrite_ui_content_actual_packaged_ui_chunk():
    """Verify idempotency on the actual packaged UI chunk file if available."""
    from pathlib import Path

    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parents[2]
    packaged_ui_path = repo_root / "litellm" / "proxy" / "_experimental" / "out"

    if not packaged_ui_path.exists():
        return

    for chunk in packaged_ui_path.rglob("*.js"):
        content = chunk.read_text(encoding="utf-8", errors="ignore")
        if ".well-known/litellm-ui-config" in content:
            once = utils.rewrite_ui_content(content, "/foo/litellm")
            twice = utils.rewrite_ui_content(once, "/foo/litellm")
            assert once == twice
            assert "/foo/foo/litellm" not in twice
            assert "/foo/litellm/.well-known/litellm-ui-config" in twice


def test_rewrite_ui_content_asset_reconfiguration_and_prefix_edge_cases():
    """Verify asset URL rewriting handles root path changes and root paths containing asset prefix."""
    sample_html = '<link rel="stylesheet" href="/litellm-asset-prefix/_next/static/css/app.css" />'

    first = utils.rewrite_ui_content(sample_html, "/foo/litellm")
    assert 'href="/foo/litellm/_next/static/css/app.css"' in first
    second = utils.rewrite_ui_content(first, "/foo/litellm")
    assert second == first

    reconfigured = utils.rewrite_ui_content(first, "/bar/litellm")
    assert 'href="/bar/litellm/_next/static/css/app.css"' in reconfigured
    assert utils.rewrite_ui_content(reconfigured, "/bar/litellm") == reconfigured

    root_with_prefix = "/litellm-asset-prefix/custom"
    with_prefix = utils.rewrite_ui_content(sample_html, root_with_prefix)
    assert f'href="{root_with_prefix}/_next/static/css/app.css"' in with_prefix
    assert utils.rewrite_ui_content(with_prefix, root_with_prefix) == with_prefix


def test_rewrite_ui_content_empty_or_root_path():
    assert utils.rewrite_ui_content(SAMPLE_PRISTINE_UI_CONTENT, "") == SAMPLE_PRISTINE_UI_CONTENT
    assert utils.rewrite_ui_content(SAMPLE_PRISTINE_UI_CONTENT, "/") == SAMPLE_PRISTINE_UI_CONTENT

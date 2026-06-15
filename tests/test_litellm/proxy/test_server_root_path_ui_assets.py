"""
Tests for SERVER_ROOT_PATH rewriting of absolute /ui/assets/ paths in JS bundles.

Regression test for https://github.com/BerriAI/litellm/issues/25283

When LiteLLM is served under a sub-path (SERVER_ROOT_PATH=/web-llmgateway/v1),
some JS chunks emit absolute paths like "/ui/assets/logos/" that the browser
resolves without the sub-path prefix, causing 404s for logo assets.
The startup path-rewriting logic must patch these absolute paths.
"""

import os


def _apply_server_root_path_replacements(
    ui_path: str,
    server_root_path: str,
    litellm_asset_prefix: str = "/litellm-asset-prefix",
) -> None:
    """Replicate the replacement logic from proxy_server.py for testing."""
    skip_extensions = (
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".ico",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
    )
    for root, _, files in os.walk(ui_path):
        for filename in files:
            if filename.endswith(skip_extensions):
                continue
            file_path = os.path.join(root, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                modified = content.replace(litellm_asset_prefix, server_root_path)
                modified = modified.replace(
                    '"/ui/assets/', f'"{server_root_path}/ui/assets/'
                )
                modified = modified.replace(
                    "'/ui/assets/", f"'{server_root_path}/ui/assets/"
                )
                modified = modified.replace(
                    "/litellm/.well-known/litellm-ui-config",
                    f"{server_root_path}/.well-known/litellm-ui-config",
                )
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified)
            except (UnicodeDecodeError, PermissionError, OSError):
                continue


def test_absolute_ui_asset_paths_rewritten_with_server_root_path(tmp_path):
    """
    JS bundles that use absolute "/ui/assets/logos/" paths should have those
    paths rewritten to include SERVER_ROOT_PATH when it is set.

    Reproduces: browser requests /{SERVER_ROOT_PATH}/assets/logos/... (404)
    instead of /{SERVER_ROOT_PATH}/ui/assets/logos/... (200).
    """
    ui_path = tmp_path / "ui"
    ui_path.mkdir()

    # Simulate a JS chunk that uses an absolute /ui/assets/ path.
    # This mirrors the pattern seen in the packaged UI output:
    #   _next/static/chunks/0d219667baa010f5.js
    js_file = ui_path / "_next" / "static" / "chunks" / "logos.js"
    js_file.parent.mkdir(parents=True)
    js_file.write_text('let eq="/ui/assets/logos/",r={github:`${eq}github.svg`}')

    server_root_path = "/web-llmgateway/v1"
    _apply_server_root_path_replacements(str(ui_path), server_root_path)

    result = js_file.read_text()
    assert (
        f'"{server_root_path}/ui/assets/logos/"' in result
    ), f"Absolute /ui/assets/ path not rewritten to include SERVER_ROOT_PATH. Got: {result}"
    assert (
        '"/ui/assets/logos/"' not in result
    ), "Old absolute /ui/assets/ path should have been replaced"


def test_litellm_asset_prefix_still_rewritten(tmp_path):
    """Existing litellm-asset-prefix replacement must still work alongside the new fix."""
    ui_path = tmp_path / "ui"
    ui_path.mkdir()

    js_file = ui_path / "bundle.js"
    js_file.write_text('let t="/litellm-asset-prefix/_next/",eq="/ui/assets/logos/"')

    server_root_path = "/myapp"
    _apply_server_root_path_replacements(str(ui_path), server_root_path)

    result = js_file.read_text()
    assert f'"{server_root_path}/_next/"' in result
    assert f'"{server_root_path}/ui/assets/logos/"' in result
    assert '"/litellm-asset-prefix/_next/"' not in result
    assert '"/ui/assets/logos/"' not in result


def test_no_replacement_when_no_server_root_path(tmp_path):
    """When server_root_path is empty, paths should remain unchanged."""
    ui_path = tmp_path / "ui"
    ui_path.mkdir()

    original = 'let eq="/ui/assets/logos/",r={}'
    js_file = ui_path / "bundle.js"
    js_file.write_text(original)

    # Simulate the guard: only replace when server_root_path is non-empty
    server_root_path = ""
    if server_root_path and server_root_path != "/":
        _apply_server_root_path_replacements(str(ui_path), server_root_path)

    assert js_file.read_text() == original


def test_binary_files_are_skipped(tmp_path):
    """Image and font files must not be touched by the replacement logic."""
    ui_path = tmp_path / "ui"
    ui_path.mkdir()

    png_file = ui_path / "logo.png"
    png_file.write_bytes(b"\x89PNG\r\n\x1a\n")

    _apply_server_root_path_replacements(str(ui_path), "/root")

    assert png_file.read_bytes() == b"\x89PNG\r\n\x1a\n"


def test_single_quoted_ui_asset_paths_rewritten(tmp_path):
    """
    Single-quoted '/ui/assets/' paths must also be rewritten.
    Some HTML/CSS files use single quotes for attribute values.
    """
    ui_path = tmp_path / "ui"
    ui_path.mkdir()

    js_file = ui_path / "bundle.js"
    js_file.write_text("let eq='/ui/assets/logos/',r={}")

    server_root_path = "/myapp"
    _apply_server_root_path_replacements(str(ui_path), server_root_path)

    result = js_file.read_text()
    assert f"'{server_root_path}/ui/assets/logos/'" in result
    assert "'/ui/assets/logos/'" not in result


def test_invalid_server_root_path_raises_value_error():
    """
    get_server_root_path() must raise ValueError for values that could
    be injected into served JS/HTML files (XSS prevention).
    """
    import os
    from unittest.mock import patch

    from litellm.proxy.utils import get_server_root_path

    malicious_paths = [
        '"><script>alert(1)</script>',
        "/myapp/../../etc/passwd",
        "/myapp with spaces",
        "/myapp?query=1",
    ]
    for path in malicious_paths:
        with patch.dict(os.environ, {"SERVER_ROOT_PATH": path}):
            try:
                result = get_server_root_path()
                assert False, f"Expected ValueError for path {path!r}, got {result!r}"
            except ValueError:
                pass  # expected


def test_valid_server_root_path_accepted():
    """Valid SERVER_ROOT_PATH values must be accepted without error."""
    import os
    from unittest.mock import patch

    from litellm.proxy.utils import get_server_root_path

    valid_paths = [
        "",
        "/myapp",
        "/myapp/v1",
        "/web-llmgateway/v1",
        "/a-b_c",
    ]
    for path in valid_paths:
        with patch.dict(os.environ, {"SERVER_ROOT_PATH": path}):
            result = get_server_root_path()
            assert result == path

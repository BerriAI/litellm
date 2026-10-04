"""SERVER_ROOT_PATH rewrite of the UI's config request (#44517)."""

from __future__ import annotations

import pytest

from litellm.proxy.proxy_server import _rewrite_ui_config_path

# The two shapes the config path takes in the packaged UI bundle
PACKAGED_UI_CHUNK = (
    'fetch(`${g}/litellm/.well-known/litellm-ui-config`);const u={url:"/litellm/.well-known/litellm-ui-config"}'
)


@pytest.mark.parametrize("server_root_path", ["/foo", "/litellm", "/foo/litellm"])
def test_restarts_keep_a_single_root_path_prefix(server_root_path: str) -> None:
    rewritten = PACKAGED_UI_CHUNK
    for _ in range(3):  # the proxy rewrites the same UI files on every start
        rewritten = _rewrite_ui_config_path(rewritten, server_root_path)

    assert rewritten == (
        f"fetch(`${{g}}{server_root_path}/.well-known/litellm-ui-config`);"
        f'const u={{url:"{server_root_path}/.well-known/litellm-ui-config"}}'
    )

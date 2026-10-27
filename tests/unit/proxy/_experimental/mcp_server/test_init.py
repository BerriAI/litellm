import importlib
import sys

import pytest

import litellm.proxy.mcp_server.server
import litellm.proxy.mcp_server.utils

OLD_PACKAGE = "litellm.proxy._experimental.mcp_server"


def test_old_package_warns_and_deep_imports_resolve_to_proxy_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name.startswith(OLD_PACKAGE):
            monkeypatch.delitem(sys.modules, name)

    with pytest.warns(DeprecationWarning, match="litellm.proxy.mcp_server"):
        importlib.import_module(OLD_PACKAGE)

    assert importlib.import_module(f"{OLD_PACKAGE}.server") is litellm.proxy.mcp_server.server
    assert importlib.import_module(f"{OLD_PACKAGE}.utils") is litellm.proxy.mcp_server.utils

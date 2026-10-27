import importlib
import sys

import pytest

import litellm.proxy.mcp_server.auth.user_api_key_auth_mcp
import litellm.proxy.mcp_server.server
import litellm.proxy.mcp_server.stdio_gate
import litellm.proxy.mcp_server.utils

OLD_PACKAGE = "litellm.proxy._experimental.mcp_server"


def test_old_package_warns_and_deep_imports_resolve_to_proxy_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name.startswith(OLD_PACKAGE):
            monkeypatch.delitem(sys.modules, name)

    with pytest.warns(DeprecationWarning, match="litellm.proxy.mcp_server"):
        old = importlib.import_module(OLD_PACKAGE)

    assert importlib.import_module(f"{OLD_PACKAGE}.server") is litellm.proxy.mcp_server.server
    assert importlib.import_module(f"{OLD_PACKAGE}.utils") is litellm.proxy.mcp_server.utils
    assert (
        importlib.import_module(f"{OLD_PACKAGE}.auth.user_api_key_auth_mcp")
        is litellm.proxy.mcp_server.auth.user_api_key_auth_mcp
    )
    assert getattr(old, "server") is litellm.proxy.mcp_server.server
    assert getattr(old, "auth") is litellm.proxy.mcp_server.auth
    assert litellm.proxy.mcp_server.server.__spec__ is not None
    assert litellm.proxy.mcp_server.server.__spec__.name == "litellm.proxy.mcp_server.server"
    with pytest.raises(AttributeError):
        getattr(old, "no_such_module")


def test_old_stdio_gate_import_leaves_the_mcp_server_unloaded(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name.startswith(OLD_PACKAGE):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.delitem(sys.modules, "litellm.proxy.mcp_server.server")

    with pytest.warns(DeprecationWarning):
        old_stdio_gate = importlib.import_module(f"{OLD_PACKAGE}.stdio_gate")

    assert old_stdio_gate is litellm.proxy.mcp_server.stdio_gate
    assert "litellm.proxy.mcp_server.server" not in sys.modules

import importlib
import sys
from collections.abc import Mapping
from typing import Final

import pytest

import litellm.proxy.mcp_server.auth.user_api_key_auth_mcp
import litellm.proxy.mcp_server.server
import litellm.proxy.mcp_server.utils

OLD_PACKAGE = "litellm.proxy._experimental.mcp_server"


def test_old_package_warns_and_deep_imports_resolve_to_proxy_mcp_server(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name.startswith(OLD_PACKAGE):
            monkeypatch.delitem(sys.modules, name)

    monkeypatch.delattr(litellm.proxy.mcp_server.auth, "user_api_key_auth_mcp")
    monkeypatch.delitem(sys.modules, "litellm.proxy.mcp_server.auth.user_api_key_auth_mcp")

    with pytest.warns(DeprecationWarning, match="litellm.proxy.mcp_server"):
        old = importlib.import_module(OLD_PACKAGE)

    assert old.server is litellm.proxy.mcp_server.server
    assert old.auth is litellm.proxy.mcp_server.auth
    assert old.auth.user_api_key_auth_mcp is importlib.import_module(
        "litellm.proxy.mcp_server.auth.user_api_key_auth_mcp"
    )
    assert importlib.import_module(f"{OLD_PACKAGE}.server") is litellm.proxy.mcp_server.server
    assert importlib.import_module(f"{OLD_PACKAGE}.utils") is litellm.proxy.mcp_server.utils
    assert (
        importlib.import_module(f"{OLD_PACKAGE}.auth.user_api_key_auth_mcp")
        is litellm.proxy.mcp_server.auth.user_api_key_auth_mcp
    )
    attrs: Mapping[str, object] = vars(old)
    assert attrs["server"] is litellm.proxy.mcp_server.server
    assert attrs["auth"] is litellm.proxy.mcp_server.auth


def test_old_stdio_helper_import_works_without_optional_mcp_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(litellm.proxy, "mcp_server")
    old_parent: Final = importlib.import_module("litellm.proxy._experimental")
    monkeypatch.delattr(old_parent, "mcp_server", raising=False)
    monkeypatch.delattr(litellm, "mcp_client", raising=False)
    prefixes: Final = (OLD_PACKAGE, "litellm.proxy.mcp_server", "litellm.mcp_client", "mcp")
    for name in tuple(sys.modules):
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, "mcp", None)

    with pytest.warns(DeprecationWarning, match="litellm.proxy.mcp_server"):
        old: Final = importlib.import_module(f"{OLD_PACKAGE}.stdio_gate")
    new: Final = importlib.import_module("litellm.proxy.mcp_server.stdio_gate")

    monkeypatch.setenv("LITELLM_ENABLE_MCP_STDIO", "true")
    assert old.is_mcp_stdio_enabled() is True
    monkeypatch.setenv("LITELLM_ENABLE_MCP_STDIO", "false")
    assert old.is_mcp_stdio_enabled() is False
    assert old is new


def test_legacy_child_import_keeps_canonical_module_reload_working(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in tuple(sys.modules):
        if name.startswith(OLD_PACKAGE):
            monkeypatch.delitem(sys.modules, name)

    helper: Final = importlib.import_module("litellm.proxy.mcp_server.stdio_gate")
    with pytest.warns(DeprecationWarning, match="litellm.proxy.mcp_server"):
        old: Final = importlib.import_module(f"{OLD_PACKAGE}.stdio_gate")

    monkeypatch.setattr(helper, "MCP_STDIO_ENABLED_ENV_VAR", "RELOAD_PROBE")
    monkeypatch.setenv("LITELLM_ENABLE_MCP_STDIO", "true")
    monkeypatch.setenv("RELOAD_PROBE", "false")
    importlib.reload(helper)

    assert old is helper
    assert old.is_mcp_stdio_enabled() is True

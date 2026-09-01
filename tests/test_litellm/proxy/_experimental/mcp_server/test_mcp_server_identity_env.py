"""Regression tests for the configurable MCP gateway identity.

``LITELLM_MCP_SERVER_NAME`` and ``LITELLM_MCP_SERVER_DESCRIPTION`` are read from
the environment at import time in
``litellm.proxy._experimental.mcp_server.utils`` and must flow through to every
consumer, including the well-known registry entry built in
``mcp_management_endpoints``. The env values are reloaded into the modules and
restored afterwards so the override does not leak into other tests.
"""

import contextlib
import importlib
import os

import pytest

pytest.importorskip("mcp")

UTILS_MODULE = "litellm.proxy._experimental.mcp_server.utils"
MGMT_MODULE = "litellm.proxy.management_endpoints.mcp_management_endpoints"


@contextlib.contextmanager
def _env_and_reload(**env):
    saved = {key: os.environ.get(key) for key in env}
    utils_module = importlib.import_module(UTILS_MODULE)
    mgmt_module = importlib.import_module(MGMT_MODULE)
    # Restore the pre-reload module attributes afterwards instead of reloading
    # a third time: a reload re-creates every class in the module, so modules
    # that imported names like MCPMissingUserEnvVarsError before this test
    # would keep raising the old class while pytest.raises in later tests
    # matches the new one
    snapshots = {module: dict(vars(module)) for module in (utils_module, mgmt_module)}

    def _apply_env(values):
        for key, value in values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def _reload():
        utils = importlib.reload(utils_module)
        mgmt = importlib.reload(mgmt_module)
        return utils, mgmt

    try:
        _apply_env(env)
        yield _reload()
    finally:
        _apply_env(saved)
        for module, snapshot in snapshots.items():
            for key in [key for key in vars(module) if key not in snapshot]:
                delattr(module, key)
            vars(module).update(snapshot)


def test_defaults_used_when_env_unset():
    with _env_and_reload(
        LITELLM_MCP_SERVER_NAME=None, LITELLM_MCP_SERVER_DESCRIPTION=None
    ) as (utils, _mgmt):
        assert utils.LITELLM_MCP_SERVER_NAME == "litellm-mcp-server"
        assert utils.LITELLM_MCP_SERVER_DESCRIPTION == "MCP Server for LiteLLM"


def test_env_overrides_server_identity():
    with _env_and_reload(
        LITELLM_MCP_SERVER_NAME="acme-gateway",
        LITELLM_MCP_SERVER_DESCRIPTION="Acme internal MCP gateway",
    ) as (utils, _mgmt):
        assert utils.LITELLM_MCP_SERVER_NAME == "acme-gateway"
        assert utils.LITELLM_MCP_SERVER_DESCRIPTION == "Acme internal MCP gateway"


def test_env_override_propagates_to_registry_entry():
    with _env_and_reload(
        LITELLM_MCP_SERVER_NAME="acme-gateway",
        LITELLM_MCP_SERVER_DESCRIPTION="Acme internal MCP gateway",
    ) as (_utils, mgmt):
        entry = mgmt._build_builtin_registry_entry("http://localhost:4000")

    assert entry["name"] == "acme-gateway"
    assert entry["title"] == "acme-gateway"
    assert entry["description"] == "Acme internal MCP gateway"

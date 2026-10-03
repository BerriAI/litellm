from pathlib import Path

import pytest

from scripts.check_mcp_operation_boundary import main, violations


@pytest.mark.parametrize(
    "source",
    (
        "from mcp.server.auth.middleware.auth_context import auth_context_var as hidden",
        "from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode as mode",
        "caller = legacy.get_active_auth_context()",
        "owners = transport._stateful_session_owners",
        "from weakref import WeakKeyDictionary",
        "from litellm.proxy._experimental.mcp_server.server import get_auth_context",
    ),
)
def test_shared_operation_boundary_rejects_ambient_state(source):
    assert violations(Path("operations.py"), source)


def test_legacy_adapter_may_resolve_context_but_policy_must_receive_it():
    source = "from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode"
    assert violations(Path("server.py"), source) == ()
    assert violations(Path("legacy_callbacks.py"), source) == ()
    assert violations(Path("operations.py"), "def execute(context):\n    return context.client_ip") == ()
    assert violations(Path("mcp_server_manager.py"), "def _mcp_registry_key(server):\n    return server.name") == ()


def test_boundary_command_rejects_shared_state_and_accepts_explicit_context(tmp_path, monkeypatch, capsys):
    import subprocess
    import sys

    package = tmp_path / "litellm/proxy/_experimental/mcp_server"
    package.mkdir(parents=True)
    module = package / "operations.py"
    module.write_text("from mcp.server.auth.middleware.auth_context import auth_context_var as hidden\n")
    command = [sys.executable, str(Path(__file__).resolve().parents[2] / "scripts/check_mcp_operation_boundary.py")]
    monkeypatch.chdir(tmp_path)
    assert main() == 1
    assert "operations.py:1:" in capsys.readouterr().err
    rejected = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert rejected.returncode == 1
    assert "operations.py:1: MCP request/session state belongs in a legacy adapter" in rejected.stderr

    module.write_text("def execute(context):\n    return context.client_ip\n")
    assert main() == 0
    assert "MCP operation boundary: passed" in capsys.readouterr().out
    accepted = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, check=False)
    assert accepted.returncode == 0
    assert "MCP operation boundary: passed" in accepted.stdout

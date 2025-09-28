"""
Purpose
- Ensure litellm (and experimental modules) import cleanly even if MCP is absent.

Scope
- DOES: simulate missing mcp/mcp.types; assert imports do not crash at import-time.
- DOES NOT: run any runtime logic depending on MCP.

Run
- `pytest tests/smoke -k test_mcp_absent_import_guard -q`
"""
import importlib
import sys
import types
import pytest


def test_mcp_absent_import_guard(monkeypatch):
    # Simulate MCP not installed
    for name in list(sys.modules.keys()):
        if name.startswith('mcp'):
            sys.modules.pop(name, None)
    monkeypatch.setitem(sys.modules, 'mcp', None)
    monkeypatch.setitem(sys.modules, 'mcp.types', None)

    # Base litellm must import without MCP
    importlib.invalidate_caches()
    litellm = importlib.import_module('litellm')
    assert litellm is not None

    # Experimental modules should not crash at import; they may lazily error on use
    try:
        importlib.import_module('litellm.experimental_mcp_client.tools')
        importlib.import_module('litellm.experimental_mcp_client.client')
    except Exception as e:
        pytest.fail(f"Experimental MCP modules should not crash on import: {e}")

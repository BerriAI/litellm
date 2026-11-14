"""
Tests for OpenAPI to MCP generator, focusing on security and edge cases.

This test suite ensures that:
1. Parameter names with invalid Python identifiers are handled safely
2. No exec() is used (security)
3. All edge cases (hyphens, dots, keywords, special chars) work correctly
4. Path traversal attacks are prevented
5. Path parameters are properly URL encoded
"""

import pytest


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


# ---------------------------------------------------------------------------
# Tool name sanitization for OpenAPI -> MCP
# Repro: GitHub's REST OpenAPI uses tag-namespaced operationIds like
# "actions/download-job-logs-for-workflow-run". Without sanitization the
# generated MCP tool name contains '/', which Anthropic/OpenAI/Bedrock all
# reject (^[a-zA-Z0-9_-]+$). This block guards the registration + preview
# paths against that.
# ---------------------------------------------------------------------------


def test_openapi_generator_import_does_not_require_mcp_sdk() -> None:
    import subprocess
    import sys

    script = """
import builtins
original_import = builtins.__import__
def without_mcp(name, *args, **kwargs):
    if name == 'mcp' or name.startswith('mcp.'):
        raise ModuleNotFoundError('MCP SDK unavailable')
    return original_import(name, *args, **kwargs)
builtins.__import__ = without_mcp
import litellm.proxy.mcp_server.openapi_to_mcp_generator
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

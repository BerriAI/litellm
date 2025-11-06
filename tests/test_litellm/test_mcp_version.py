"""
Test to verify MCP package version is 1.20.0 or higher.
"""
import pytest


def test_mcp_version():
    """
    Verify that the MCP package version is 1.20.0 or higher.
    This test ensures the MCP dependency upgrade was successful.
    """
    try:
        import mcp

        # Get version from package metadata
        try:
            from importlib.metadata import version

            mcp_version = version("mcp")
        except ImportError:
            # Fallback for older Python versions
            import pkg_resources

            mcp_version = pkg_resources.get_distribution("mcp").version

        # Parse version string and check
        version_parts = mcp_version.split(".")
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0

        # Version should be 1.20.0 or higher
        assert major >= 1, f"MCP major version should be >= 1, got {major}"
        if major == 1:
            assert (
                minor >= 20
            ), f"MCP minor version should be >= 20 for major version 1, got {minor}"

        print(f"✓ MCP version {mcp_version} meets requirement (>= 1.20.0)")

    except ImportError:
        pytest.skip("MCP package not installed")


def test_mcp_imports():
    """
    Verify that all required MCP imports are still available after the version update.
    This ensures backward compatibility of the MCP package.
    """
    try:
        # Test all imports used in the litellm codebase
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client
        from mcp.types import CallToolRequestParams as MCPCallToolRequestParams
        from mcp.types import CallToolResult as MCPCallToolResult
        from mcp.types import TextContent
        from mcp.types import Tool as MCPTool

        # Verify all imports are not None
        assert ClientSession is not None
        assert StdioServerParameters is not None
        assert sse_client is not None
        assert stdio_client is not None
        assert streamablehttp_client is not None
        assert MCPCallToolRequestParams is not None
        assert MCPCallToolResult is not None
        assert TextContent is not None
        assert MCPTool is not None

        print("✓ All MCP imports are available and backward compatible")

    except ImportError as e:
        pytest.fail(f"MCP import failed: {e}")

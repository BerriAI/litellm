from litellm._version import get_distribution_name

try:
    from .tools import call_openai_tool, load_mcp_tools
except ModuleNotFoundError as exc:
    if exc.name not in ("mcp", "httpx2"):
        raise
    raise ImportError(
        f"MCP client dependencies are missing. Install them with: pip install '{get_distribution_name()}[mcp]'"
    ) from exc

__all__ = ["call_openai_tool", "load_mcp_tools"]

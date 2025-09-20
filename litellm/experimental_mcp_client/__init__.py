# Soft-dep guard: avoid importing MCP tools at package import time
try:
    from .tools import call_openai_tool, load_mcp_tools  # type: ignore
    __all__ = ["load_mcp_tools", "call_openai_tool"]
except Exception:  # pragma: no cover - optional MCP stack may be absent
    __all__ = []


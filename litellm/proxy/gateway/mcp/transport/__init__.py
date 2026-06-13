"""transport — MCP Gateway v2

S1 · terminate client connections (/mcp, /sse), parse scope, dispatch. Routes stay thin: parse -> call one service -> serialize.
"""

"""LiteLLM MCP Gateway — the clean-room v2 gateway, grafted into v1 behind a flag.

Written v2-native; nothing here imports from v1 (`litellm.proxy._experimental.mcp_server.*`).
v1 only ever reaches this package through a thin adapter, never the other way round.
"""

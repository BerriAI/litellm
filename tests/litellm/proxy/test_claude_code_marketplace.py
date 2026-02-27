import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_claude_code_plugin_table_schema_exists():
    """
    Test that LiteLLM_ClaudeCodePluginTable is defined in schema.prisma.
    Regression test for https://github.com/BerriAI/litellm/issues/21310
    """
    with open("schema.prisma", "r") as f:
        schema = f.read()
    assert "LiteLLM_ClaudeCodePluginTable" in schema, (
        "LiteLLM_ClaudeCodePluginTable model missing from schema.prisma - "
        "this causes AttributeError on all /claude-code/plugins endpoints"
    )

    with open("litellm/proxy/schema.prisma", "r") as f:
        proxy_schema = f.read()
    assert "LiteLLM_ClaudeCodePluginTable" in proxy_schema, (
        "LiteLLM_ClaudeCodePluginTable model missing from litellm/proxy/schema.prisma"
    )
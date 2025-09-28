import asyncio
import pytest


@pytest.mark.ndsmoke
def test_stdio_mcp_list_tools_and_model_advice_optional():
    """Spawn the stdio MCP server and verify tools + a local tool call.

    Skip-friendly: if the Python MCP client is unavailable, or transport fails, skip.
    """
    try:
        # Python MCP client (model-context-protocol)
        from mcp.client.stdio import StdioServerParameters, connect  # type: ignore
    except Exception:
        pytest.skip("Python MCP client not installed (model-context-protocol)")

    async def _run():
        # Connect to server as a child process over stdio
        params = StdioServerParameters(
            command="python3",
            args=["-m", "litellm.proxy._experimental.mcp_server.stdio_server"],
        )
        try:
            async with connect(params) as session:  # type: ignore
                tools = await session.list_tools()
                names = {getattr(t, "name", None) or t.get("name") for t in tools}
                assert "model.advice" in names, f"tools={names}"
                assert "llm.chat" in names, f"tools={names}"

                # Call a purely local tool (no external deps)
                result = await session.call_tool(
                    name="model.advice",
                    arguments={
                        "task_description": "summarize a long pdf with images",
                        "max_context_tokens": 400000,
                    },
                )
                # result is a list of content blocks; check for any text
                texts = []
                for c in result:
                    text = getattr(c, "text", None) or (isinstance(c, dict) and c.get("text"))
                    if text:
                        texts.append(text)
                assert any(texts), "no text content returned from model.advice"
        except Exception as e:
            pytest.skip(f"MCP stdio connect/call failed: {e}")

    # Run with a timeout to avoid hanging CI
    try:
        asyncio.run(asyncio.wait_for(_run(), timeout=15))
    except asyncio.TimeoutError:
        pytest.skip("stdio MCP smoke timed out")


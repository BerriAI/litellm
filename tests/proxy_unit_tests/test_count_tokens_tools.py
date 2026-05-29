import pytest
from litellm.proxy._types import TokenCountRequest


@pytest.mark.asyncio
async def test_count_tokens_anthropic_tools_not_ignored():
    """Test that Anthropic-style tools are counted in fallback path"""
    from litellm.proxy.proxy_server import token_counter as proxy_token_counter

    request_with_tools = TokenCountRequest(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "hello"}],
        tools=[{
            "name": "example_tool",
            "description": "A tool " + "with long description " * 100,
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {"type": "string", "description": "A parameter"}
                }
            }
        }]
    )

    request_without_tools = TokenCountRequest(
        model="claude-opus-4-6",
        messages=[{"role": "user", "content": "hello"}],
    )

    response_with = await proxy_token_counter(request=request_with_tools, call_endpoint=False)
    response_without = await proxy_token_counter(request=request_without_tools, call_endpoint=False)

    assert response_with.total_tokens > response_without.total_tokens, (
        f"Tools should add tokens: with={response_with.total_tokens}, without={response_without.total_tokens}"
    )

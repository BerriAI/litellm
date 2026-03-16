"""
E2E tests for Claude Agent SDK with LiteLLM Proxy using Bedrock models.

Tests streaming messages across different Bedrock models:
- Regular Bedrock Claude Sonnet 4.5
- Bedrock Converse Claude Sonnet 4.5
- AWS Nova Pro
"""

import os
import pytest
import asyncio
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions


# Test models from test_config.yaml
# Note: bedrock-converse-claude-sonnet-4.5 removed temporarily as the Bedrock Converse API
# for Claude Sonnet 4.5 may not be available in all regions/accounts
# Note: bedrock-nova-premier requires provisioned throughput (not standard cross-region
# inference profile) and is not reliably available in CI accounts. Using nova-pro instead.
TEST_MODELS = [
    ("bedrock-claude-sonnet-4.5", "Bedrock Invoke API"),
    ("bedrock-converse-claude-sonnet-4.5", "Bedrock Converse API"),
    ("bedrock-nova-pro", "AWS Nova Pro"),
]


@pytest.fixture(scope="module")
def litellm_proxy_config():
    """Configure connection to LiteLLM proxy"""
    proxy_url = os.getenv("LITELLM_PROXY_URL", "http://localhost:4000")
    api_key = os.getenv("LITELLM_API_KEY", "sk-1234")
    
    # Set environment variables for Claude Agent SDK
    os.environ["ANTHROPIC_BASE_URL"] = proxy_url.rstrip('/')
    os.environ["ANTHROPIC_API_KEY"] = api_key
    
    return {
        "proxy_url": proxy_url,
        "api_key": api_key,
    }


MAX_RETRIES = 3


async def _run_streaming_test(model_name: str) -> tuple[list[str], str]:
    """
    Run a single streaming test attempt for the given model.

    Returns (received_chunks, full_response).
    """
    options = ClaudeAgentOptions(
        system_prompt=(
            "You are a helpful AI assistant. "
            "Always follow the user's instructions exactly."
        ),
        model=model_name,
        max_turns=5,
    )

    test_query = (
        "Respond with exactly the following text and nothing else:\n"
        "Hello from LiteLLM!"
    )

    received_chunks: list[str] = []
    full_response = ""

    async with ClaudeSDKClient(options=options) as client:
        await client.query(test_query)

        async for msg in client.receive_response():
            if hasattr(msg, 'type'):
                if msg.type == 'content_block_delta':
                    if hasattr(msg, 'delta') and hasattr(msg.delta, 'text'):
                        chunk_text = msg.delta.text
                        received_chunks.append(chunk_text)
                        full_response += chunk_text
                elif msg.type == 'content_block_start':
                    if hasattr(msg, 'content_block') and hasattr(msg.content_block, 'text'):
                        chunk_text = msg.content_block.text
                        received_chunks.append(chunk_text)
                        full_response += chunk_text

            # Fallback to content handling
            if hasattr(msg, 'content'):
                for content_block in msg.content:
                    if hasattr(content_block, 'text'):
                        chunk_text = content_block.text
                        received_chunks.append(chunk_text)
                        full_response += chunk_text

    return received_chunks, full_response


@pytest.mark.asyncio
@pytest.mark.parametrize("model_name,model_description", TEST_MODELS)
async def test_claude_agent_sdk_streaming(litellm_proxy_config, model_name, model_description):
    """
    Test streaming messages with Claude Agent SDK through LiteLLM proxy.

    This validates:
    1. Claude Agent SDK can connect to LiteLLM proxy
    2. Streaming works correctly
    3. Different Bedrock models (Invoke, Converse, Nova) work end-to-end
    """
    print(f"\n{'='*60}")
    print(f"Testing: {model_name} ({model_description})")
    print(f"{'='*60}")

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            received_chunks, full_response = await _run_streaming_test(model_name)

            # Assertions
            print(f"\n✅ Received {len(received_chunks)} chunks")
            print(f"📝 Full response: {full_response[:100]}...")

            # Verify we got a response
            assert len(full_response) > 0, f"No response received from {model_name}"

            # Verify streaming (should have multiple chunks for most responses)
            # Note: Very short responses might come in 1 chunk, so we just verify we got content
            assert len(received_chunks) > 0, f"No chunks received from {model_name}"

            # Verify response contains expected content (case insensitive)
            assert "hello" in full_response.lower(), (
                f"Response doesn't contain expected greeting: {full_response}"
            )

            print(f"✅ Test passed for {model_name} (attempt {attempt})")
            return  # Success

        except Exception as e:
            last_error = e
            print(f"⚠️ Attempt {attempt}/{MAX_RETRIES} failed for {model_name}: {e}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2)

    pytest.fail(
        f"Test failed for {model_name} ({model_description}) after {MAX_RETRIES} attempts: {last_error}"
    )


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

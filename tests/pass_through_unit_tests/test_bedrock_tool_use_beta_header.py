"""
Simple E2E test for Bedrock with advanced-tool-use beta header.

Tests that LiteLLM correctly filters out the advanced-tool-use-2025-11-20 beta header
for Bedrock Invoke API, which doesn't support it and returns a 400 "invalid beta flag" error.
"""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath("../.."))

import litellm


@pytest.mark.asyncio
async def test_bedrock_sonnet_4_5_with_advanced_tool_use_beta_header():
    """
    Simple E2E test: Call Bedrock Sonnet 4.5 with advanced-tool-use beta header.

    This should work without throwing "invalid beta flag" error because LiteLLM
    filters out the advanced-tool-use beta header for Bedrock Invoke API.
    """
    litellm._turn_on_debug()
    response = await litellm.anthropic.messages.acreate(
        model="bedrock/invoke/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        max_tokens=100,
        provider_specific_header={
            "custom_llm_provider": "bedrock",
            "extra_headers": {
                "anthropic-beta": "advanced-tool-use-2025-11-20",
            },
        },
    )

    # Verify response
    assert response is not None
    assert "content" in response
    print(f"✅ Test passed! Response: {response}")


# @pytest.mark.asyncio
# async def test_bedrock_claude_3_5_with_advanced_tool_use_beta_header_filtered():
#     """
#     Simple E2E test: Call Bedrock Claude 3.5 with advanced-tool-use beta header.

#     This should work because the beta header is filtered out by LiteLLM before
#     sending the request to Bedrock Invoke API.
#     """

#     response = await litellm.anthropic.messages.acreate(
#         model="bedrock/invoke/us.anthropic.claude-3-5-sonnet-20240620-v1:0",
#         messages=[{"role": "user", "content": "What is 2+2?"}],
#         max_tokens=100,
#         provider_specific_header={
#             "custom_llm_provider": "bedrock",
#             "extra_headers": {
#                 "anthropic-beta": "advanced-tool-use-2025-11-20",
#             },
#         },
#     )

#     # Verify response
#     assert response is not None
#     assert "content" in response
#     print(f"✅ Test passed! Claude 3.5 response (beta header filtered): {response}")



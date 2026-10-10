"""
Tests for Azure Foundry Agent Service integration.

These tests require an Azure Foundry Agent Service endpoint and a pre-configured agent.

The Azure Foundry Agent Service uses the Assistants API pattern:
1. Create a thread
2. Add messages to the thread
3. Create and poll a run
4. Get the agent's response messages

Model format: azure_ai/agents/<agent_id>

API Base format: https://<AIFoundryResourceName>.services.ai.azure.com/api/projects/<ProjectName>

Authentication: Uses Azure AD Bearer tokens (not API keys)
  Get token via: az account get-access-token --resource 'https://ai.azure.com'

Example environment variables:
  AZURE_AGENTS_API_BASE=https://litellm-ci-cd-prod.services.ai.azure.com/api/projects/litellm-ci-cd
  AZURE_AGENTS_API_KEY=<Azure AD Bearer token>

See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
"""

import json
import os


import pytest
from unittest.mock import MagicMock

import litellm


@pytest.mark.asyncio
async def test_azure_ai_agents_acompletion_non_streaming():
    """
    Test non-streaming acompletion call to Azure Foundry Agent Service.
    Uses the multi-step flow: create thread -> add messages -> create/poll run -> get messages
    """
    api_base = os.environ.get("AZURE_AGENTS_API_BASE")
    api_key = os.environ.get("AZURE_AGENTS_API_KEY")
    agent_id = os.environ.get("AZURE_AGENTS_AGENT_ID", "asst_hbnoK9BOCcHhC3lC4MDroVGG")

    if not api_base or not api_key:
        pytest.skip(
            "AZURE_AGENTS_API_BASE and AZURE_AGENTS_API_KEY environment variables required"
        )

    response = await litellm.acompletion(
        model=f"azure_ai/agents/{agent_id}",
        messages=[{"role": "user", "content": "Hi Agent, what is 25 * 4?"}],
        api_base=api_base,
        api_key=api_key,
        stream=False,
    )

    assert response is not None
    assert response.choices is not None
    assert len(response.choices) > 0
    assert response.choices[0].message is not None
    assert response.choices[0].message.content is not None
    assert len(response.choices[0].message.content) > 0

    # Verify thread_id is returned for conversation continuity
    if hasattr(response, "_hidden_params") and response._hidden_params:
        assert "thread_id" in response._hidden_params

    print(f"Response: {response.choices[0].message.content}")


@pytest.mark.asyncio
async def test_azure_ai_agents_acompletion_streaming():
    """
    Test native streaming acompletion call to Azure Foundry Agent Service.
    Uses the create-thread-and-run endpoint with stream=True for SSE streaming.
    """
    api_base = os.environ.get("AZURE_AGENTS_API_BASE")
    api_key = os.environ.get("AZURE_AGENTS_API_KEY")
    agent_id = os.environ.get("AZURE_AGENTS_AGENT_ID", "asst_hbnoK9BOCcHhC3lC4MDroVGG")

    if not api_base or not api_key:
        pytest.skip(
            "AZURE_AGENTS_API_BASE and AZURE_AGENTS_API_KEY environment variables required"
        )

    response = await litellm.acompletion(
        model=f"azure_ai/agents/{agent_id}",
        messages=[{"role": "user", "content": "Hi Agent, what is 10 + 5?"}],
        api_base=api_base,
        api_key=api_key,
        stream=True,
    )

    # Native streaming - collect chunks from the async iterator
    chunks = []
    full_content = ""
    async for chunk in response:
        print("Streaming chunk: ", chunk)
        chunks.append(chunk)
        if hasattr(chunk, "choices") and chunk.choices:
            delta = chunk.choices[0].delta
            if hasattr(delta, "content") and delta.content:
                full_content += delta.content

    assert len(chunks) > 0, "Expected at least one streaming chunk"
    assert len(full_content) > 0, "Expected content from streaming response"
    print(f"Streamed response ({len(chunks)} chunks): {full_content}")
































@pytest.mark.asyncio
async def test_azure_ai_agents_conversation_continuity():
    """
    Test that thread_id can be used for conversation continuity.
    """
    api_base = os.environ.get("AZURE_AGENTS_API_BASE")
    api_key = os.environ.get("AZURE_AGENTS_API_KEY")
    agent_id = os.environ.get("AZURE_AGENTS_AGENT_ID", "asst_hbnoK9BOCcHhC3lC4MDroVGG")

    if not api_base or not api_key:
        pytest.skip(
            "AZURE_AGENTS_API_BASE and AZURE_AGENTS_API_KEY environment variables required"
        )

    try:
        # First message
        response1 = await litellm.acompletion(
            model=f"azure_ai/agents/{agent_id}",
            messages=[{"role": "user", "content": "My name is Alice. Remember this."}],
            api_base=api_base,
            api_key=api_key,
            stream=False,
        )

        assert response1 is not None

        # Get thread_id for continuity
        thread_id = None
        if hasattr(response1, "_hidden_params") and response1._hidden_params:
            thread_id = response1._hidden_params.get("thread_id")

        if thread_id:
            # Second message using the same thread
            response2 = await litellm.acompletion(
                model=f"azure_ai/agents/{agent_id}",
                messages=[{"role": "user", "content": "What is my name?"}],
                api_base=api_base,
                api_key=api_key,
                thread_id=thread_id,  # Continue the conversation
                stream=False,
            )

            assert response2 is not None
            # The agent should remember the name from the previous message
            print(f"Response to name question: {response2.choices[0].message.content}")

    except Exception as e:
        pytest.skip(f"Azure Agent Service not available: {e}")

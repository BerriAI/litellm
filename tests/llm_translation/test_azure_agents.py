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

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

import pytest

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
        pytest.skip("AZURE_AGENTS_API_BASE and AZURE_AGENTS_API_KEY environment variables required")

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
        pytest.skip("AZURE_AGENTS_API_BASE and AZURE_AGENTS_API_KEY environment variables required")

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



def test_azure_ai_agents_is_agents_route():
    """
    Test the is_azure_ai_agents_route detection method.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    # Should be recognized as agents route
    assert AzureAIAgentsConfig.is_azure_ai_agents_route("azure_ai/agents/asst_123") is True
    assert AzureAIAgentsConfig.is_azure_ai_agents_route("agents/asst_123") is True
    
    # Should NOT be recognized as agents route
    assert AzureAIAgentsConfig.is_azure_ai_agents_route("azure_ai/gpt-4") is False
    assert AzureAIAgentsConfig.is_azure_ai_agents_route("gpt-4") is False


def test_azure_ai_get_azure_ai_route():
    """
    Test the get_azure_ai_route dispatch method.
    """
    from litellm.llms.azure_ai.common_utils import AzureFoundryModelInfo

    # Should return "agents" for agents routes
    assert AzureFoundryModelInfo.get_azure_ai_route("agents/asst_123") == "agents"
    assert AzureFoundryModelInfo.get_azure_ai_route("azure_ai/agents/asst_abc") == "agents"
    
    # Should return "default" for non-agents routes
    assert AzureFoundryModelInfo.get_azure_ai_route("gpt-4") == "default"
    assert AzureFoundryModelInfo.get_azure_ai_route("claude-3-sonnet") == "default"
    assert AzureFoundryModelInfo.get_azure_ai_route("azure_ai/gpt-4o") == "default"


def test_azure_ai_agents_get_agent_id_from_model():
    """
    Test agent ID extraction from model name.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    # Test with full model name
    agent_id = AzureAIAgentsConfig.get_agent_id_from_model("azure_ai/agents/asst_abc123")
    assert agent_id == "asst_abc123"

    # Test with just agents/id
    agent_id = AzureAIAgentsConfig.get_agent_id_from_model("agents/asst_xyz789")
    assert agent_id == "asst_xyz789"

    # Test with just agent ID (fallback)
    agent_id = AzureAIAgentsConfig.get_agent_id_from_model("asst_plain")
    assert agent_id == "asst_plain"


def test_azure_ai_agents_config_get_agent_id():
    """
    Test agent ID extraction via config method.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    config = AzureAIAgentsConfig()

    # Test with full model name
    agent_id = config._get_agent_id("azure_ai/agents/asst_abc123", {})
    assert agent_id == "asst_abc123"

    # Test with optional_params override
    agent_id = config._get_agent_id("azure_ai/agents/asst_abc123", {"agent_id": "asst_override"})
    assert agent_id == "asst_override"

    # Test with assistant_id in optional_params
    agent_id = config._get_agent_id("azure_ai/agents/asst_abc123", {"assistant_id": "asst_assistant"})
    assert agent_id == "asst_assistant"


def test_azure_ai_agents_config_get_complete_url():
    """
    Test that AzureAIAgentsConfig correctly generates base URLs.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    config = AzureAIAgentsConfig()

    # Test URL generation
    url = config.get_complete_url(
        api_base="https://test-project.services.ai.azure.com",
        api_key=None,
        model="agents/asst_123",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url == "https://test-project.services.ai.azure.com"

    # Test URL with trailing slash
    url_with_slash = config.get_complete_url(
        api_base="https://test-project.services.ai.azure.com/",
        api_key=None,
        model="agents/asst_123",
        optional_params={},
        litellm_params={},
        stream=False,
    )
    assert url_with_slash == "https://test-project.services.ai.azure.com"


def test_azure_ai_agents_config_transform_request():
    """
    Test that AzureAIAgentsConfig correctly transforms requests.
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    config = AzureAIAgentsConfig()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2 + 2?"},
    ]

    request = config.transform_request(
        model="azure_ai/agents/asst_123",
        messages=messages,
        optional_params={},
        litellm_params={"stream": False},
        headers={},
    )

    assert request["agent_id"] == "asst_123"
    assert "messages" in request
    assert len(request["messages"]) == 2
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][1]["role"] == "user"
    assert "api_version" in request
    assert request["api_version"] == "2025-05-01"


def test_azure_ai_agents_provider_detection():
    """
    Test that the azure_ai provider is correctly detected from model name.
    """
    from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

    model, provider, api_key, api_base = get_llm_provider(
        model="azure_ai/agents/asst_abc123",
        api_base="https://test.services.ai.azure.com",
    )

    assert provider == "azure_ai"
    assert model == "agents/asst_abc123"


def test_azure_ai_agents_validate_environment():
    """
    Test that headers are correctly set up with Bearer token authentication.
    
    Azure Foundry Agents uses Bearer token authentication (Azure AD tokens).
    """
    from litellm.llms.azure_ai.agents.transformation import AzureAIAgentsConfig

    config = AzureAIAgentsConfig()

    headers = config.validate_environment(
        headers={},
        model="agents/asst_123",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-azure-ad-token",
        api_base="https://test.services.ai.azure.com/api/projects/test-project",
    )

    assert headers["Content-Type"] == "application/json"
    assert headers["Authorization"] == "Bearer test-azure-ad-token"


def test_azure_ai_agents_handler_url_builders():
    """
    Test the URL building methods in the handler.
    
    Azure Foundry Agents API uses direct paths without /openai/ prefix.
    See: https://learn.microsoft.com/en-us/azure/ai-foundry/agents/quickstart
    """
    from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler

    handler = AzureAIAgentsHandler()
    api_base = "https://test.services.ai.azure.com/api/projects/test-project"
    api_version = "2025-05-01"
    thread_id = "thread_abc123"
    run_id = "run_xyz789"

    # Test thread URL - direct path without /openai/ prefix
    thread_url = handler._build_thread_url(api_base, api_version)
    assert thread_url == f"{api_base}/threads?api-version={api_version}"

    # Test messages URL
    messages_url = handler._build_messages_url(api_base, thread_id, api_version)
    assert messages_url == f"{api_base}/threads/{thread_id}/messages?api-version={api_version}"

    # Test runs URL
    runs_url = handler._build_runs_url(api_base, thread_id, api_version)
    assert runs_url == f"{api_base}/threads/{thread_id}/runs?api-version={api_version}"

    # Test run status URL
    status_url = handler._build_run_status_url(api_base, thread_id, run_id, api_version)
    assert status_url == f"{api_base}/threads/{thread_id}/runs/{run_id}?api-version={api_version}"


def test_azure_ai_agents_extract_content_from_messages():
    """
    Test content extraction from Azure Agents message response.
    """
    from litellm.llms.azure_ai.agents.handler import AzureAIAgentsHandler

    handler = AzureAIAgentsHandler()

    # Test typical message response
    messages_data = {
        "data": [
            {
                "id": "msg_123",
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": {"value": "The answer is 100."}
                    }
                ]
            },
            {
                "id": "msg_122",
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": {"value": "What is 25 * 4?"}
                    }
                ]
            }
        ]
    }

    content = handler._extract_content_from_messages(messages_data)
    assert content == "The answer is 100."

    # Test empty response
    empty_data = {"data": []}
    content = handler._extract_content_from_messages(empty_data)
    assert content == ""


@pytest.mark.asyncio
async def test_azure_ai_agents_conversation_continuity():
    """
    Test that thread_id can be used for conversation continuity.
    """
    api_base = os.environ.get("AZURE_AGENTS_API_BASE")
    api_key = os.environ.get("AZURE_AGENTS_API_KEY")
    agent_id = os.environ.get("AZURE_AGENTS_AGENT_ID", "asst_hbnoK9BOCcHhC3lC4MDroVGG")

    if not api_base or not api_key:
        pytest.skip("AZURE_AGENTS_API_BASE and AZURE_AGENTS_API_KEY environment variables required")

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

"""
Tests whether Langfuse is dynamically discovered and can be initialized without real network calls.
"""
from unittest.mock import patch
from litellm.proxy.prompts.prompt_registry import get_prompt_initializer_from_integrations
from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec

def test_langfuse_discovery_and_init():
    """
    Validates the fix in __init__.py by ensuring Langfuse is registered and initialized (mocked).
    """
    # 1. Verify if the integration was registered in the global dictionary
    registry = get_prompt_initializer_from_integrations()
    assert "langfuse" in registry, "Error: Langfuse wasn't discovered by the prompt_registry!"

    # 2. Test initialization using MOCK to avoid real network calls
    # Patch the function that creates the real Langfuse client
    with patch("litellm.integrations.langfuse.langfuse_prompt_management.langfuse_client_init") as mocked_client:
        mocked_client.return_value = None  # No real client needed for this test
        
        init_func = registry["langfuse"]
        params = PromptLiteLLMParams(
            prompt_integration="langfuse", 
            langfuse_public_key="test-key",
            langfuse_secret="test-secret"
        )
        spec = PromptSpec(prompt_id="test-id", litellm_params=params)
        
        # Initialize the class (this would call the network, but is now mocked)
        obj = init_func(params, spec)
        
        assert obj is not None, "Initiation function returned None!"
        assert obj.integration_name == "langfuse"

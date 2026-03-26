import pytest
from litellm.proxy.prompts.prompt_registry import get_prompt_initializer_from_integrations
from litellm.types.prompts.init_prompts import PromptLiteLLMParams, PromptSpec

def test_langfuse_discovery_and_init():
    """
    Tests whether Langfuse is dynamically discovered and can be initialized.
    This validates the fix in __init__.py at litellm/integrations/langfuse/
    """

    registry = get_prompt_initializer_from_integrations()
    assert "langfuse" in registry, "Error: Langfuse wasn't discovered by the prompt_registry!"

    init_func = registry["langfuse"]
    params = PromptLiteLLMParams(
        prompt_integration="langfuse", 
        langfuse_public_key="test-key",
        langfuse_secret="test-secret"
    )
    spec = PromptSpec(prompt_id="test-id", litellm_params=params)
    
    obj = init_func(params, spec)
    assert obj is not None, "Initiation function returned None!"
    assert obj.integration_name == "langfuse"
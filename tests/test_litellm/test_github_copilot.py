"""
Unit tests for GitHub Copilot model registration in LiteLLM.
Validates that models listed in the GitHub Copilot API are correctly
registered in the model_prices_and_context_window.json registry.
"""

import pytest
import litellm


# Models newly added by this PR (gpt-4o was already registered)
EXPECTED_MODELS = [
    "github_copilot/claude-3-5-sonnet",
    "github_copilot/claude-3-5-haiku",
    "github_copilot/gemini-1-5-pro",
    "github_copilot/gpt-5.4",
    "github_copilot/gpt-5.4-mini",
]

REASONING_MODELS = [
    "github_copilot/gpt-5-mini",
    "github_copilot/gpt-5.4",
    "github_copilot/gpt-5.4-mini",
    "github_copilot/gpt-5.5",
]


@pytest.mark.parametrize("model", EXPECTED_MODELS)
def test_github_copilot_model_registered(model):
    """Each GitHub Copilot model should be resolvable via get_model_info."""
    model_info = litellm.get_model_info(model)
    assert model_info is not None, f"Model '{model}' not found in registry"
    assert model_info["litellm_provider"] == "github_copilot", (
        f"Expected litellm_provider='github_copilot' for '{model}', "
        f"got '{model_info['litellm_provider']}'"
    )
    assert model_info["mode"] == "chat", (
        f"Expected mode='chat' for '{model}', got '{model_info['mode']}'"
    )


@pytest.mark.parametrize("model", EXPECTED_MODELS)
def test_github_copilot_model_has_token_limits(model):
    """Each model should define max_tokens."""
    model_info = litellm.get_model_info(model)
    assert model_info is not None
    assert "max_tokens" in model_info, f"'{model}' is missing max_tokens"
    assert model_info["max_tokens"] > 0, (
        f"'{model}' has invalid max_tokens={model_info['max_tokens']}"
    )


def test_github_copilot_claude_3_5_sonnet_context_window():
    """claude-3-5-sonnet should reflect the 200k context window."""
    model_info = litellm.get_model_info("github_copilot/claude-3-5-sonnet")
    assert model_info is not None
    assert model_info["max_input_tokens"] == 200000


def test_github_copilot_claude_3_5_haiku_context_window():
    """claude-3-5-haiku should reflect the 200k context window."""
    model_info = litellm.get_model_info("github_copilot/claude-3-5-haiku")
    assert model_info is not None
    assert model_info["max_input_tokens"] == 200000


@pytest.mark.parametrize("model", EXPECTED_MODELS)
def test_github_copilot_models_support_function_calling(model):
    """Each newly added model should support function calling."""
    model_info = litellm.get_model_info(model)
    assert model_info is not None
    assert model_info.get("supports_function_calling") is True, (
        f"'{model}' should support function calling"
    )


@pytest.mark.parametrize("model", REASONING_MODELS)
def test_github_copilot_reasoning_models(model):
    """gpt-5-mini, gpt-5.4, and gpt-5.4-mini should have supports_reasoning=True."""
    model_info = litellm.get_model_info(model)
    assert model_info is not None, f"Model '{model}' not found in registry"
    assert model_info.get("supports_reasoning") is True, (
        f"'{model}' should have supports_reasoning=True"
    )

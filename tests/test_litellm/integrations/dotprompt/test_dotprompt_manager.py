import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


from unittest.mock import MagicMock, patch

import litellm
from litellm.integrations.dotprompt import DotpromptManager
from litellm.types.utils import StandardCallbackDynamicParams


def test_dotprompt_manager_initialization():
    """Test basic DotpromptManager initialization."""
    prompt_dir = os.path.dirname(__file__)  # Directory containing this test file
    manager = DotpromptManager(prompt_dir)

    assert manager.integration_name == "dotprompt"
    assert manager.prompt_directory == prompt_dir


def test_should_run_prompt_management():
    """Test should_run_prompt_management method."""
    prompt_dir = os.path.dirname(__file__)
    manager = DotpromptManager(prompt_dir)

    # Test with existing prompt
    assert (
        manager.should_run_prompt_management(
            "sample_prompt", StandardCallbackDynamicParams()
        )
        == True
    )

    # Test with non-existing prompt
    assert (
        manager.should_run_prompt_management(
            "nonexistent_prompt", StandardCallbackDynamicParams()
        )
        == False
    )


def test_convert_to_messages_simple():
    """Test converting simple text to messages."""
    prompt_dir = os.path.dirname(__file__)
    manager = DotpromptManager(prompt_dir)

    # Test simple text
    messages = manager._convert_to_messages("Hello world!")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello world!"


def test_convert_to_messages_with_roles():
    """Test converting text with role prefixes to messages."""
    prompt_dir = os.path.dirname(__file__)
    manager = DotpromptManager(prompt_dir)

    # Test text with role prefixes
    content = """System: You are a helpful assistant.

User: What is the capital of France?"""

    messages = manager._convert_to_messages(content)
    assert len(messages) == 2

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."

    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is the capital of France?"


def test_compile_prompt_helper():
    """Test the _compile_prompt_helper method."""
    prompt_dir = os.path.dirname(__file__)
    manager = DotpromptManager(prompt_dir)

    # Test compiling a simple prompt
    result = manager._compile_prompt_helper(
        prompt_id="sample_prompt",
        prompt_variables={"text": "This is a test article."},
        dynamic_callback_params=StandardCallbackDynamicParams(),
    )

    assert result["prompt_id"] == "sample_prompt"
    assert result["prompt_template_model"] == "gemini/gemini-1.5-pro"
    assert len(result["prompt_template"]) >= 1
    assert "This is a test article." in result["prompt_template"][0]["content"]


def test_compile_prompt_helper_with_chat_format():
    """Test compiling a prompt that generates role-based messages."""
    prompt_dir = os.path.dirname(__file__)
    manager = DotpromptManager(prompt_dir)

    # Test with chat_prompt that has system context
    result = manager._compile_prompt_helper(
        prompt_id="chat_prompt",
        prompt_variables={
            "user_message": "Hello there!",
            "system_context": "You are a helpful assistant.",
        },
        dynamic_callback_params=StandardCallbackDynamicParams(),
    )

    assert result["prompt_id"] == "chat_prompt"
    assert result["prompt_template_model"] == "gpt-4"
    assert len(result["prompt_template"]) == 2

    # Should have system message first
    assert result["prompt_template"][0]["role"] == "system"
    assert "You are a helpful assistant." in result["prompt_template"][0]["content"]

    # Then user message
    assert result["prompt_template"][1]["role"] == "user"
    assert "Hello there!" in result["prompt_template"][1]["content"]


def test_extract_optional_params():
    """Test extracting optional parameters from template metadata."""
    prompt_dir = os.path.dirname(__file__)
    manager = DotpromptManager(prompt_dir)

    # Get a template with optional params
    template = manager.prompt_manager.get_prompt("chat_prompt")
    params = manager._extract_optional_params(template)

    assert "temperature" in params
    assert params["temperature"] == 0.7
    assert "max_tokens" in params
    assert params["max_tokens"] == 150


def test_error_handling():
    """Test error handling for invalid prompts."""
    prompt_dir = os.path.dirname(__file__)
    manager = DotpromptManager(prompt_dir)

    # Test with non-existent prompt
    with pytest.raises(ValueError, match="Prompt 'nonexistent' not found"):
        manager._compile_prompt_helper(
            prompt_id="nonexistent",
            prompt_variables={},
            dynamic_callback_params=StandardCallbackDynamicParams(),
        )


def test_integration_with_prompt_management():
    """Test integration with the prompt management system."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a test prompt
        prompt_file = Path(temp_dir) / "test_integration.prompt"
        prompt_file.write_text(
            """---
model: gpt-3.5-turbo
temperature: 0.5
---
System: You are a {{role}}.

User: {{question}}"""
        )

        manager = DotpromptManager(temp_dir)

        # Test should_run_prompt_management
        assert (
            manager.should_run_prompt_management(
                "test_integration", StandardCallbackDynamicParams()
            )
            == True
        )

        # Test compile_prompt_helper
        result = manager._compile_prompt_helper(
            prompt_id="test_integration",
            prompt_variables={"role": "helpful assistant", "question": "What is AI?"},
            dynamic_callback_params=StandardCallbackDynamicParams(),
        )

        assert result["prompt_template_model"] == "gpt-3.5-turbo"
        assert result["prompt_template_optional_params"]["temperature"] == 0.5
        assert len(result["prompt_template"]) == 2

        assert result["prompt_template"][0]["role"] == "system"
        assert "helpful assistant" in result["prompt_template"][0]["content"]

        assert result["prompt_template"][1]["role"] == "user"
        assert "What is AI?" in result["prompt_template"][1]["content"]


def test_set_prompt_directory():
    """Test setting and changing prompt directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = DotpromptManager(temp_dir)

        # Initially should be empty
        assert not manager.should_run_prompt_management(
            "test_prompt", StandardCallbackDynamicParams()
        )

        # Create a prompt file
        prompt_file = Path(temp_dir) / "test_prompt.prompt"
        prompt_file.write_text("Hello {{name}}!")

        # Set directory to force reload
        manager.set_prompt_directory(temp_dir)

        # Now should find the prompt
        assert manager.should_run_prompt_management(
            "test_prompt", StandardCallbackDynamicParams()
        )


def test_no_prompt_directory_error():
    """Test error when no prompt directory is set."""
    manager = DotpromptManager(None)

    # should_run_prompt_management returns False when there's an error
    result = manager.should_run_prompt_management(
        "any_prompt", StandardCallbackDynamicParams()
    )
    assert result == False

    # But accessing prompt_manager property should raise an error
    with pytest.raises(ValueError, match="prompt_directory must be set"):
        _ = manager.prompt_manager

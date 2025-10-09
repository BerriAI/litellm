import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.integrations.bitbucket import BitBucketPromptManager


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_integration_with_litellm(mock_client_class):
    """Test BitBucket prompt integration with LiteLLM completion."""
    # Mock the BitBucket client
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
temperature: 0.7
max_tokens: 150
---
System: You are a helpful assistant.

User: {{user_message}}"""
    mock_client_class.return_value = mock_client

    # Configure BitBucket
    bitbucket_config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    # Set global BitBucket configuration
    litellm.set_global_bitbucket_config(bitbucket_config)

    # Test that the configuration was set
    assert litellm.global_bitbucket_config == bitbucket_config


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_initialization(mock_client_class):
    """Test BitBucketPromptManager initialization."""
    # Mock the BitBucket client
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
temperature: 0.7
---
Hello {{name}}!"""
    mock_client_class.return_value = mock_client

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config, prompt_id="test_prompt")

    # Should have loaded the prompt
    assert "test_prompt" in manager.prompt_manager.prompts
    template = manager.prompt_manager.prompts["test_prompt"]
    assert template.model == "gpt-4"
    assert template.temperature == 0.7

    # Test rendering
    rendered = manager.prompt_manager.render_template("test_prompt", {"name": "World"})
    assert rendered == "Hello World!"


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_error_handling(mock_client_class):
    """Test BitBucketPromptManager error handling."""
    # Mock the BitBucket client to raise an error
    mock_client = MagicMock()
    mock_client.get_file_content.side_effect = Exception("BitBucket API error")
    mock_client_class.return_value = mock_client

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    with pytest.raises(Exception, match="Failed to load prompt 'test_prompt' from BitBucket"):
        manager = BitBucketPromptManager(config, prompt_id="test_prompt")
        _ = manager.prompt_manager  # This triggers the error


def test_bitbucket_prompt_manager_config_validation():
    """Test BitBucketPromptManager configuration validation."""
    # Test missing required fields - validation happens when prompt_manager is accessed
    with pytest.raises(ValueError, match="workspace, repository, and access_token are required"):
        manager = BitBucketPromptManager({})
        _ = manager.prompt_manager  # This triggers validation

    with pytest.raises(ValueError, match="workspace, repository, and access_token are required"):
        manager = BitBucketPromptManager({"workspace": "test"})
        _ = manager.prompt_manager  # This triggers validation

    with pytest.raises(ValueError, match="workspace, repository, and access_token are required"):
        manager = BitBucketPromptManager({"repository": "test"})
        _ = manager.prompt_manager  # This triggers validation

    with pytest.raises(ValueError, match="workspace, repository, and access_token are required"):
        manager = BitBucketPromptManager({"access_token": "test"})
        _ = manager.prompt_manager  # This triggers validation


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_complex_prompt(mock_client_class):
    """Test BitBucketPromptManager with complex prompt structure."""
    # Mock the BitBucket client
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
temperature: 0.3
max_tokens: 500
input:
  schema:
    user_question: string
    context?: string
    language: string
---
System: You are a helpful {{language}} programming assistant.

{% if context %}Context: {{context}}

{% endif %}User: {{user_question}}

Please provide a detailed response in {{language}}."""
    mock_client_class.return_value = mock_client

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config, prompt_id="complex_prompt")

    # Should have loaded the prompt
    assert "complex_prompt" in manager.prompt_manager.prompts
    template = manager.prompt_manager.prompts["complex_prompt"]
    assert template.model == "gpt-4"
    assert template.temperature == 0.3
    assert template.max_tokens == 500
    assert template.input_schema == {
        "user_question": "string",
        "context?": "string",
        "language": "string"
    }

    # Test rendering with all variables
    rendered = manager.prompt_manager.render_template(
        "complex_prompt",
        {
            "user_question": "How do I create a class?",
            "context": "Python programming",
            "language": "Python"
        }
    )

    assert "You are a helpful Python programming assistant." in rendered
    assert "Context: Python programming" in rendered
    assert "How do I create a class?" in rendered
    assert "Please provide a detailed response in Python." in rendered

    # Test rendering without optional context
    rendered_no_context = manager.prompt_manager.render_template(
        "complex_prompt",
        {
            "user_question": "What is inheritance?",
            "language": "Java"
        }
    )

    assert "You are a helpful Java programming assistant." in rendered_no_context
    assert "Context:" not in rendered_no_context
    assert "What is inheritance?" in rendered_no_context


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_message_parsing(mock_client_class):
    """Test BitBucketPromptManager message parsing for different prompt formats."""
    # Mock the BitBucket client
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
---
System: You are a helpful assistant.

User: {{user_message}}

Assistant: I'll help you with that."""
    mock_client_class.return_value = mock_client

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config, prompt_id="conversation_prompt")

    # Test message parsing
    messages = manager._parse_prompt_to_messages(
        "System: You are a helpful assistant.\n\nUser: Hello!\n\nAssistant: Hi there!"
    )

    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello!"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "Hi there!"


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_pre_call_hook_integration(mock_client_class):
    """Test BitBucketPromptManager pre_call_hook integration."""
    # Mock the BitBucket client
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = """---
model: gpt-4
temperature: 0.8
max_tokens: 200
---
System: You are a helpful assistant.

User: {{user_message}}"""
    mock_client_class.return_value = mock_client

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config, prompt_id="test_prompt")

    # Test pre_call_hook
    original_messages = [{"role": "user", "content": "This will be ignored"}]
    litellm_params = {"api_key": "test-key"}

    result_messages, result_params = manager.pre_call_hook(
        user_id="test_user",
        messages=original_messages,
        litellm_params=litellm_params,
        prompt_id="test_prompt",
        prompt_variables={"user_message": "What is AI?"}
    )

    # Should have parsed the prompt into messages
    assert len(result_messages) == 2
    assert result_messages[0]["role"] == "system"
    assert result_messages[0]["content"] == "You are a helpful assistant."
    assert result_messages[1]["role"] == "user"
    assert result_messages[1]["content"] == "What is AI?"

    # Should have updated litellm_params
    assert result_params["model"] == "gpt-4"
    assert result_params["temperature"] == 0.8
    assert result_params["max_tokens"] == 200
    assert result_params["api_key"] == "test-key"  # Original params preserved


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_post_call_hook(mock_client_class):
    """Test BitBucketPromptManager post_call_hook."""
    # Mock the BitBucket client
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "Simple prompt: {{message}}"
    mock_client_class.return_value = mock_client

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config, prompt_id="test_prompt")

    # Mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = "Test response"

    # Test post_call_hook
    result = manager.post_call_hook(
        user_id="test_user",
        response=mock_response,
        input_messages=[{"role": "user", "content": "test"}],
        litellm_params={},
        prompt_id="test_prompt"
    )

    # Should return the response unchanged
    assert result == mock_response


def test_bitbucket_prompt_manager_integration_name():
    """Test BitBucketPromptManager integration name."""
    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)
    assert manager.integration_name == "bitbucket"


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_get_template(mock_client_class):
    """Test BitBucketPromptManager get_template method."""
    # Mock the BitBucket client
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "Test content"
    mock_client_class.return_value = mock_client

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config, prompt_id="test_prompt")

    # Test getting existing template
    template = manager.prompt_manager.get_template("test_prompt")
    assert template is not None
    assert template.template_id == "test_prompt"

    # Test getting non-existing template
    template = manager.prompt_manager.get_template("nonexistent")
    assert template is None


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_list_templates(mock_client_class):
    """Test BitBucketPromptManager list_templates method."""
    # Mock the BitBucket client
    mock_client = MagicMock()
    mock_client.get_file_content.return_value = "Test content"
    mock_client_class.return_value = mock_client

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config, prompt_id="test_prompt")

    # Test listing templates
    templates = manager.prompt_manager.list_templates()
    assert isinstance(templates, list)
    assert "test_prompt" in templates

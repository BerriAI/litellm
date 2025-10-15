import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.integrations.bitbucket.bitbucket_client import BitBucketClient
from litellm.integrations.bitbucket.bitbucket_prompt_manager import (
    BitBucketPromptManager,
    BitBucketPromptTemplate,
)


def test_bitbucket_prompt_template_creation():
    """Test BitBucketPromptTemplate creation and metadata extraction."""
    metadata = {
        "model": "gpt-4",
        "temperature": 0.7,
        "input": {"schema": {"text": "string"}},
        "output": {"format": "json"},
    }

    template = BitBucketPromptTemplate(
        template_id="test_template",
        content="Hello {{name}}!",
        metadata=metadata,
    )

    assert template.template_id == "test_template"
    assert template.content == "Hello {{name}}!"
    assert template.model == "gpt-4"
    assert template.optional_params["temperature"] == 0.7
    assert template.input_schema == {"text": "string"}


def test_bitbucket_client_initialization():
    """Test BitBucketClient initialization with different auth methods."""
    # Test token-based auth
    config_token = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
        "branch": "main",
    }

    client = BitBucketClient(config_token)
    assert client.workspace == "test-workspace"
    assert client.repository == "test-repo"
    assert client.access_token == "test-token"
    assert client.branch == "main"
    assert client.auth_method == "token"

    # Test basic auth
    config_basic = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-password",
        "username": "test-user",
        "auth_method": "basic",
    }

    client_basic = BitBucketClient(config_basic)
    assert client_basic.auth_method == "basic"
    assert client_basic.username == "test-user"


def test_bitbucket_client_missing_required_fields():
    """Test BitBucketClient initialization with missing required fields."""
    with pytest.raises(ValueError, match="workspace, repository, and access_token are required"):
        BitBucketClient({"workspace": "test"})

    with pytest.raises(ValueError, match="workspace, repository, and access_token are required"):
        BitBucketClient({"repository": "test"})

    with pytest.raises(ValueError, match="workspace, repository, and access_token are required"):
        BitBucketClient({"access_token": "test"})


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_bitbucket_client_get_file_content_success(mock_get):
    """Test successful file content retrieval from BitBucket."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.text = "file content"
    mock_response.headers = {"content-type": "text/plain"}
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    client = BitBucketClient(config)
    content = client.get_file_content("test.prompt")

    assert content == "file content"
    mock_get.assert_called_once()


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_bitbucket_client_get_file_content_not_found(mock_get):
    """Test file content retrieval when file doesn't exist."""
    # Mock 404 response
    import httpx
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("404 Not Found", request=MagicMock(), response=mock_response)
    mock_response.status_code = 404
    mock_response.response = mock_response
    mock_get.return_value = mock_response

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    client = BitBucketClient(config)
    content = client.get_file_content("nonexistent.prompt")

    assert content is None


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_bitbucket_client_get_file_content_access_denied(mock_get):
    """Test file content retrieval with access denied error."""
    # Mock 403 response
    import httpx
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("403 Forbidden", request=MagicMock(), response=mock_response)
    mock_response.status_code = 403
    mock_response.response = mock_response
    mock_get.return_value = mock_response

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    client = BitBucketClient(config)

    with pytest.raises(Exception, match="Access denied to file 'test.prompt'"):
        client.get_file_content("test.prompt")


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_bitbucket_client_get_file_content_auth_failed(mock_get):
    """Test file content retrieval with authentication failure."""
    # Mock 401 response
    import httpx
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError("401 Unauthorized", request=MagicMock(), response=mock_response)
    mock_response.status_code = 401
    mock_response.response = mock_response
    mock_get.return_value = mock_response

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    client = BitBucketClient(config)

    with pytest.raises(Exception, match="Authentication failed"):
        client.get_file_content("test.prompt")


@patch("litellm.llms.custom_httpx.http_handler.HTTPHandler.get")
def test_bitbucket_client_list_files_success(mock_get):
    """Test successful file listing from BitBucket."""
    # Mock successful response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "values": [
            {"type": "commit_file", "path": "prompts/test1.prompt"},
            {"type": "commit_file", "path": "prompts/test2.prompt"},
            {"type": "commit_file", "path": "prompts/other.txt"},
        ]
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    client = BitBucketClient(config)
    files = client.list_files("prompts", ".prompt")

    assert len(files) == 2
    assert "prompts/test1.prompt" in files
    assert "prompts/test2.prompt" in files


def test_bitbucket_prompt_manager_parse_prompt_file():
    """Test parsing .prompt file content with YAML frontmatter."""
    prompt_content = """---
model: gpt-4
temperature: 0.7
max_tokens: 150
input:
  schema:
    user_message: string
    system_context?: string
---

{% if system_context %}System: {{system_context}}

{% endif %}User: {{user_message}}"""

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)
    template = manager.prompt_manager._parse_prompt_file(prompt_content, "test_prompt")

    assert template.template_id == "test_prompt"
    assert template.model == "gpt-4"
    assert template.temperature == 0.7
    assert template.max_tokens == 150
    assert template.input_schema == {"user_message": "string", "system_context?": "string"}
    assert "{% if system_context %}" in template.content


def test_bitbucket_prompt_manager_parse_prompt_file_no_frontmatter():
    """Test parsing .prompt file content without YAML frontmatter."""
    prompt_content = "Simple prompt: {{message}}"

    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)
    template = manager.prompt_manager._parse_prompt_file(prompt_content, "simple_prompt")

    assert template.template_id == "simple_prompt"
    assert template.content == "Simple prompt: {{message}}"
    assert template.metadata == {}


def test_bitbucket_prompt_manager_render_template():
    """Test template rendering with variables."""
    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)
    
    # Add a test template
    template = BitBucketPromptTemplate(
        template_id="test_template",
        content="Hello {{name}}! Welcome to {{place}}.",
        metadata={"model": "gpt-4"},
    )
    manager.prompt_manager.prompts["test_template"] = template

    rendered = manager.prompt_manager.render_template("test_template", {"name": "World", "place": "Earth"})
    assert rendered == "Hello World! Welcome to Earth."


def test_bitbucket_prompt_manager_render_template_not_found():
    """Test template rendering when template doesn't exist."""
    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)

    with pytest.raises(ValueError, match="Template 'nonexistent' not found"):
        manager.prompt_manager.render_template("nonexistent", {"some": "variable"})


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_integration(mock_client_class):
    """Test BitBucketPromptManager integration with BitBucketClient."""
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


def test_bitbucket_prompt_manager_parse_prompt_to_messages():
    """Test parsing prompt content into messages."""
    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)

    # Test simple user message
    simple_prompt = "Hello, how can I help you?"
    messages = manager._parse_prompt_to_messages(simple_prompt)
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello, how can I help you?"

    # Test multi-role conversation
    multi_role_prompt = """System: You are a helpful assistant.

User: What is the capital of France?

Assistant: The capital of France is Paris."""
    
    messages = manager._parse_prompt_to_messages(multi_role_prompt)
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a helpful assistant."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "What is the capital of France?"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "The capital of France is Paris."


def test_bitbucket_prompt_manager_pre_call_hook():
    """Test the pre_call_hook method."""
    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)
    
    # Add a test template
    template = BitBucketPromptTemplate(
        template_id="test_prompt",
        content="System: You are a helpful assistant.\n\nUser: {{user_message}}",
        metadata={"model": "gpt-4", "temperature": 0.7},
    )
    manager.prompt_manager.prompts["test_prompt"] = template

    # Test pre_call_hook
    messages = [{"role": "user", "content": "This will be ignored"}]
    litellm_params = {}
    
    result_messages, result_params = manager.pre_call_hook(
        user_id="test_user",
        messages=messages,
        litellm_params=litellm_params,
        prompt_id="test_prompt",
        prompt_variables={"user_message": "Hello!"}
    )

    # Should have parsed the prompt into messages
    assert len(result_messages) == 2
    assert result_messages[0]["role"] == "system"
    assert result_messages[0]["content"] == "You are a helpful assistant."
    assert result_messages[1]["role"] == "user"
    assert result_messages[1]["content"] == "Hello!"

    # Should have updated litellm_params
    assert result_params["model"] == "gpt-4"
    assert result_params["temperature"] == 0.7


def test_bitbucket_prompt_manager_pre_call_hook_no_prompt_id():
    """Test pre_call_hook when no prompt_id is provided."""
    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)
    
    messages = [{"role": "user", "content": "Hello"}]
    litellm_params = {}
    
    result_messages, result_params = manager.pre_call_hook(
        user_id="test_user",
        messages=messages,
        litellm_params=litellm_params,
        prompt_id=None,
    )

    # Should return original messages and params unchanged
    assert result_messages == messages
    assert result_params == litellm_params


def test_bitbucket_prompt_manager_get_available_prompts():
    """Test getting list of available prompts."""
    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)
    
    # Add some test templates
    template1 = BitBucketPromptTemplate("prompt1", "content1", {})
    template2 = BitBucketPromptTemplate("prompt2", "content2", {})
    manager.prompt_manager.prompts["prompt1"] = template1
    manager.prompt_manager.prompts["prompt2"] = template2

    available_prompts = manager.get_available_prompts()
    assert set(available_prompts) == {"prompt1", "prompt2"}


@patch("litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient")
def test_bitbucket_prompt_manager_reload_prompts(mock_client_class):
    """Test reloading prompts from BitBucket."""
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
    
    # Mock the prompt manager to test reload
    with patch.object(manager, '_prompt_manager', None):
        manager.reload_prompts()
        # Should trigger reload by accessing prompt_manager property
        _ = manager.prompt_manager


def test_bitbucket_prompt_manager_yaml_parsing_fallback():
    """Test YAML parsing fallback when PyYAML is not available."""
    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)
    
    # Test basic YAML parsing fallback
    yaml_content = """model: gpt-4
temperature: 0.7
max_tokens: 150"""
    
    parsed = manager.prompt_manager._parse_yaml_basic(yaml_content)
    assert parsed["model"] == "gpt-4"
    assert parsed["temperature"] == 0.7
    assert parsed["max_tokens"] == 150


def test_bitbucket_prompt_manager_yaml_parsing_with_types():
    """Test YAML parsing with different data types."""
    config = {
        "workspace": "test-workspace",
        "repository": "test-repo",
        "access_token": "test-token",
    }

    manager = BitBucketPromptManager(config)
    
    yaml_content = """model: gpt-4
temperature: 0.7
max_tokens: 150
enabled: true
disabled: false
count: 42
rate: 0.5"""
    
    parsed = manager.prompt_manager._parse_yaml_basic(yaml_content)
    assert parsed["model"] == "gpt-4"
    assert parsed["temperature"] == 0.7
    assert parsed["max_tokens"] == 150
    assert parsed["enabled"] is True
    assert parsed["disabled"] is False
    assert parsed["count"] == 42
    assert parsed["rate"] == 0.5

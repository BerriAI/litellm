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
from litellm.integrations.dotprompt.prompt_manager import PromptManager, PromptTemplate


def test_prompt_manager_initialization():
    """Test basic PromptManager initialization and loading."""
    # Test with the existing prompts directory
    prompt_dir = Path(
        __file__
    ).parent  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_directory=str(prompt_dir))

    # Should have loaded at least the sample prompts
    assert len(manager.prompts) >= 3
    assert "sample_prompt" in manager.prompts
    assert "chat_prompt" in manager.prompts
    assert "coding_assistant" in manager.prompts


def test_prompt_template_creation():
    """Test PromptTemplate creation and metadata extraction."""
    metadata = {
        "model": "gpt-4",
        "temperature": 0.7,
        "input": {"schema": {"text": "string"}},
        "output": {"format": "json"},
    }

    template = PromptTemplate(
        content="Hello {{name}}!", metadata=metadata, template_id="test_template"
    )

    assert template.content == "Hello {{name}}!"
    assert template.model == "gpt-4"
    assert template.optional_params["temperature"] == 0.7
    assert template.input_schema == {"text": "string"}
    assert template.output_format == "json"


def test_render_simple_template():
    """Test rendering a simple template with variables."""
    prompt_dir = Path(
        __file__
    ).parent  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_directory=str(prompt_dir))

    # Test sample_prompt rendering
    rendered = manager.render(
        "sample_prompt", {"text": "This is a test article about AI."}
    )

    expected_content = "Extract the requested information from the given text. If a piece of information is not present, omit that field from the output.\n\nText: This is a test article about AI."
    assert rendered == expected_content


def test_render_chat_prompt():
    """Test rendering the chat prompt with conditional content."""
    prompt_dir = Path(
        __file__
    ).parent  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_directory=str(prompt_dir))

    # Test with system context
    rendered = manager.render(
        "chat_prompt",
        {
            "user_message": "Hello there!",
            "system_context": "You are a helpful assistant.",
        },
    )

    assert "System: You are a helpful assistant." in rendered
    assert "User: Hello there!" in rendered

    # Test without system context
    rendered_no_system = manager.render("chat_prompt", {"user_message": "Hello there!"})

    assert "System:" not in rendered_no_system
    assert "User: Hello there!" in rendered_no_system


def test_render_coding_assistant():
    """Test rendering the coding assistant prompt with complex logic."""
    prompt_dir = Path(
        __file__
    ).parent  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_directory=str(prompt_dir))

    rendered = manager.render(
        "coding_assistant",
        {
            "language": "Python",
            "task": "Create a function to calculate fibonacci numbers",
            "code": "def fib(n):\n    pass",
            "requirements": ["Use recursion", "Handle edge cases", "Add documentation"],
        },
    )

    assert "Focus on Python programming." in rendered
    assert "Create a function to calculate fibonacci numbers" in rendered
    assert "def fib(n):" in rendered
    assert "Use recursion" in rendered
    assert "Handle edge cases" in rendered
    assert "Add documentation" in rendered


def test_input_validation():
    """Test input validation against schema."""
    # Create a temporary directory with a test prompt
    with tempfile.TemporaryDirectory() as temp_dir:
        prompt_file = Path(temp_dir) / "test_validation.prompt"
        prompt_file.write_text(
            """---
input:
  schema:
    name: string
    age: integer
    active: boolean
---
Hello {{name}}, you are {{age}} years old and {'active' if active else 'inactive'}."""
        )

        manager = PromptManager(prompt_directory=str(temp_dir))

        # Valid input should work
        rendered = manager.render(
            "test_validation", {"name": "Alice", "age": 30, "active": True}
        )
        assert "Hello Alice, you are 30 years old" in rendered

        # Invalid type should raise error
        with pytest.raises(ValueError, match="Invalid type for field 'age'"):
            manager.render(
                "test_validation",
                {
                    "name": "Alice",
                    "age": "thirty",  # string instead of int
                    "active": True,
                },
            )


def test_prompt_not_found():
    """Test error handling for non-existent prompts."""
    prompt_dir = Path(
        __file__
    ).parent  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_directory=str(prompt_dir))

    with pytest.raises(KeyError, match="Prompt 'nonexistent' not found"):
        manager.render("nonexistent", {"some": "variable"})


def test_list_prompts():
    """Test listing available prompts."""
    prompt_dir = Path(
        __file__
    ).parent  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_directory=str(prompt_dir))

    prompts = manager.list_prompts()
    assert isinstance(prompts, list)
    assert "sample_prompt" in prompts
    assert "chat_prompt" in prompts
    assert "coding_assistant" in prompts


def test_get_prompt_metadata():
    """Test retrieving prompt metadata."""
    prompt_dir = Path(
        __file__
    ).parent  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_directory=str(prompt_dir))

    metadata = manager.get_prompt_metadata("sample_prompt")
    assert metadata is not None
    assert metadata["model"] == "gemini/gemini-1.5-pro"
    assert "input" in metadata
    assert "output" in metadata


def test_get_prompt_with_version():
    """Test that get_prompt correctly retrieves versioned prompts."""
    prompt_dir = Path(__file__).parent
    manager = PromptManager(prompt_directory=str(prompt_dir))

    # Get base prompt (no version)
    base_prompt = manager.get_prompt(prompt_id="chat_prompt")
    assert base_prompt is not None
    assert "User: {{user_message}}" in base_prompt.content

    # Get version 1
    v1_prompt = manager.get_prompt(prompt_id="chat_prompt", version=1)
    assert v1_prompt is not None
    assert "Version 1:" in v1_prompt.content
    assert v1_prompt.model == "gpt-3.5-turbo"

    # Get version 2
    v2_prompt = manager.get_prompt(prompt_id="chat_prompt", version=2)
    assert v2_prompt is not None
    assert "Version 2:" in v2_prompt.content
    assert v2_prompt.model == "gpt-4"


def test_add_prompt_programmatically():
    """Test adding prompts programmatically."""
    prompt_dir = Path(
        __file__
    ).parent  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_directory=str(prompt_dir))

    initial_count = len(manager.prompts)

    manager.add_prompt(
        "dynamic_prompt",
        "Hello {{name}}! Welcome to {{place}}.",
        {"model": "gpt-3.5-turbo", "temperature": 0.5},
    )

    assert len(manager.prompts) == initial_count + 1
    assert "dynamic_prompt" in manager.prompts

    rendered = manager.render("dynamic_prompt", {"name": "World", "place": "Earth"})
    assert rendered == "Hello World! Welcome to Earth."


def test_frontmatter_parsing():
    """Test YAML frontmatter parsing."""
    # Create a temporary directory with a test prompt
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with frontmatter
        prompt_with_frontmatter = Path(temp_dir) / "with_frontmatter.prompt"
        prompt_with_frontmatter.write_text(
            """---
model: gpt-4
temperature: 0.8
input:
  schema:
    topic: string
---
Write about {{topic}}."""
        )

        # Test without frontmatter
        prompt_without_frontmatter = Path(temp_dir) / "without_frontmatter.prompt"
        prompt_without_frontmatter.write_text("Simple template: {{message}}")

        manager = PromptManager(prompt_directory=str(temp_dir))

        # Check frontmatter was parsed correctly
        with_meta = manager.get_prompt("with_frontmatter")
        assert with_meta is not None
        assert with_meta.model == "gpt-4"
        assert with_meta.optional_params["temperature"] == 0.8

        # Check template without frontmatter still works
        without_meta = manager.get_prompt("without_frontmatter")
        assert without_meta is not None
        assert without_meta.metadata == {}

        rendered = manager.render("without_frontmatter", {"message": "Hello!"})
        assert rendered == "Simple template: Hello!"


def test_prompt_manager_json_initialization():
    """Test PromptManager initialization with JSON data instead of directory."""
    prompt_data = {
        "json_test_prompt": {
            "content": "Hello {{name}}! Welcome to {{service}}.",
            "metadata": {"model": "gpt-4", "temperature": 0.8, "max_tokens": 150},
        },
        "simple_prompt": {
            "content": "This is a simple prompt: {{message}}",
            "metadata": {},
        },
    }

    # Initialize PromptManager with JSON data only (no directory)
    manager = PromptManager(prompt_data=prompt_data)

    # Should have loaded the JSON prompts
    assert len(manager.prompts) == 2
    assert "json_test_prompt" in manager.prompts
    assert "simple_prompt" in manager.prompts

    # Test prompt properties
    json_prompt = manager.get_prompt("json_test_prompt")
    assert json_prompt.content == "Hello {{name}}! Welcome to {{service}}."
    assert json_prompt.model == "gpt-4"
    assert json_prompt.optional_params["temperature"] == 0.8
    assert json_prompt.optional_params["max_tokens"] == 150


def test_prompt_manager_mixed_initialization():
    """Test PromptManager with both directory and JSON data."""
    # Use existing directory
    prompt_dir = Path(__file__).parent

    # Add JSON data
    json_data = {
        "json_only_prompt": {
            "content": "This prompt only exists in JSON: {{data}}",
            "metadata": {"model": "gpt-3.5-turbo"},
        }
    }

    manager = PromptManager(prompt_directory=str(prompt_dir), prompt_data=json_data)

    # Should have prompts from both directory and JSON
    assert "sample_prompt" in manager.prompts  # From directory
    assert "json_only_prompt" in manager.prompts  # From JSON

    # Test rendering both types
    json_rendered = manager.render("json_only_prompt", {"data": "test"})
    assert json_rendered == "This prompt only exists in JSON: test"


def test_load_prompts_from_json_data():
    """Test loading additional prompts from JSON data after initialization."""
    # Start with directory-based manager
    prompt_dir = Path(__file__).parent
    manager = PromptManager(prompt_directory=str(prompt_dir))

    initial_count = len(manager.prompts)

    # Load additional prompts from JSON
    additional_prompts = {
        "dynamic_json_prompt": {
            "content": "Dynamic prompt: {{dynamic_content}}",
            "metadata": {"model": "claude-3", "temperature": 0.5},
        },
        "another_json_prompt": {
            "content": "Another prompt with {{variable}}",
            "metadata": {"model": "gpt-4"},
        },
    }

    manager.load_prompts_from_json_data(additional_prompts)

    # Should have added the new prompts
    assert len(manager.prompts) == initial_count + 2
    assert "dynamic_json_prompt" in manager.prompts
    assert "another_json_prompt" in manager.prompts

    # Test rendering the new prompts
    rendered = manager.render("dynamic_json_prompt", {"dynamic_content": "test"})
    assert rendered == "Dynamic prompt: test"


def test_prompt_file_to_json_conversion():
    """Test converting .prompt files to JSON format."""
    # Create a temporary prompt file with frontmatter
    with tempfile.TemporaryDirectory() as temp_dir:
        prompt_file = Path(temp_dir) / "test_conversion.prompt"
        prompt_file.write_text(
            """---
model: gpt-4
temperature: 0.7
max_tokens: 200
input:
  schema:
    user_input: string
    context: string
output:
  format: json
---
You are an AI assistant. Given the context: {{context}}

Please respond to: {{user_input}}"""
        )

        manager = PromptManager()
        json_data = manager.prompt_file_to_json(prompt_file)

        # Check the conversion
        assert "content" in json_data
        assert "metadata" in json_data

        expected_content = """You are an AI assistant. Given the context: {{context}}

Please respond to: {{user_input}}"""
        assert json_data["content"] == expected_content

        metadata = json_data["metadata"]
        assert metadata["model"] == "gpt-4"
        assert metadata["temperature"] == 0.7
        assert metadata["max_tokens"] == 200
        assert metadata["input"]["schema"]["user_input"] == "string"
        assert metadata["output"]["format"] == "json"


def test_json_to_prompt_file_conversion():
    """Test converting JSON data back to .prompt file format."""
    json_data = {
        "content": "Hello {{name}}! How can I help you with {{task}}?",
        "metadata": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.8,
            "max_tokens": 100,
            "input": {"schema": {"name": "string", "task": "string"}},
        },
    }

    manager = PromptManager()
    prompt_content = manager.json_to_prompt_file(json_data)

    # Should have YAML frontmatter and content
    assert prompt_content.startswith("---\n")
    assert "---\n" in prompt_content[4:]  # Second --- delimiter
    assert "Hello {{name}}! How can I help you with {{task}}?" in prompt_content
    assert "model: gpt-3.5-turbo" in prompt_content
    assert "temperature: 0.8" in prompt_content


def test_json_to_prompt_file_without_metadata():
    """Test converting JSON with no metadata to .prompt format."""
    json_data = {
        "content": "Simple prompt without metadata: {{message}}",
        "metadata": {},
    }

    manager = PromptManager()
    prompt_content = manager.json_to_prompt_file(json_data)

    # Should return just the content without frontmatter
    assert prompt_content == "Simple prompt without metadata: {{message}}"
    assert "---" not in prompt_content


def test_get_all_prompts_as_json():
    """Test exporting all prompts to JSON format."""
    prompt_data = {
        "prompt1": {
            "content": "First prompt: {{var1}}",
            "metadata": {"model": "gpt-4"},
        },
        "prompt2": {
            "content": "Second prompt: {{var2}}",
            "metadata": {"model": "claude-3", "temperature": 0.5},
        },
    }

    manager = PromptManager(prompt_data=prompt_data)
    all_prompts_json = manager.get_all_prompts_as_json()

    assert len(all_prompts_json) == 2
    assert "prompt1" in all_prompts_json
    assert "prompt2" in all_prompts_json

    # Check structure
    prompt1_data = all_prompts_json["prompt1"]
    assert prompt1_data["content"] == "First prompt: {{var1}}"
    assert prompt1_data["metadata"]["model"] == "gpt-4"

    prompt2_data = all_prompts_json["prompt2"]
    assert prompt2_data["content"] == "Second prompt: {{var2}}"
    assert prompt2_data["metadata"]["model"] == "claude-3"


def test_json_prompt_rendering_with_validation():
    """Test rendering JSON-based prompts with input validation."""
    prompt_data = {
        "validated_prompt": {
            "content": "Process {{data}} for user {{user_id}}",
            "metadata": {
                "model": "gpt-4",
                "input": {"schema": {"data": "string", "user_id": "integer"}},
            },
        }
    }

    manager = PromptManager(prompt_data=prompt_data)

    # Valid input should work
    rendered = manager.render("validated_prompt", {"data": "test data", "user_id": 123})
    assert rendered == "Process test data for user 123"

    # Invalid input should raise error
    with pytest.raises(ValueError, match="Invalid type for field 'user_id'"):
        manager.render(
            "validated_prompt", {"data": "test data", "user_id": "not_an_int"}
        )


def test_round_trip_conversion():
    """Test converting .prompt file to JSON and back to .prompt file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create original prompt file
        original_file = Path(temp_dir) / "original.prompt"
        original_content = """---
model: gpt-4
temperature: 0.6
---
Original prompt content: {{variable}}"""
        original_file.write_text(original_content)

        manager = PromptManager()

        # Convert to JSON
        json_data = manager.prompt_file_to_json(original_file)

        # Convert back to prompt file format
        converted_content = manager.json_to_prompt_file(json_data)

        # Create new file with converted content
        converted_file = Path(temp_dir) / "converted.prompt"
        converted_file.write_text(converted_content)

        # Load both files and compare
        manager_original = PromptManager(prompt_directory=str(temp_dir))

        original_template = manager_original.get_prompt("original")
        converted_template = manager_original.get_prompt("converted")

        # Content should be the same
        assert original_template.content == converted_template.content
        assert original_template.model == converted_template.model
        assert (
            original_template.optional_params["temperature"]
            == converted_template.optional_params["temperature"]
        )


def test_prompt_main():
    """
    Integration test placeholder for litellm completion integration.
    This would be implemented once the PromptManager is integrated with litellm.
    """
    # TODO: Implement once PromptManager is integrated with litellm completion
    pass


@pytest.mark.asyncio
async def test_dotprompt_auto_detection_with_model_only():
    """
    Test that dotprompt prompts can be auto-detected when passing model="gpt-4" and prompt_id,
    without needing to specify model="dotprompt/gpt-4".
    """
    from litellm.integrations.dotprompt import DotpromptManager

    prompt_dir = Path(__file__).parent
    dotprompt_manager = DotpromptManager(prompt_directory=str(prompt_dir))
    
    # Register the dotprompt manager in callbacks
    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [dotprompt_manager]
    
    try:
        # Mock the HTTP handler to avoid actual API calls
        with patch("litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler.post") as mock_post:
            mock_response_data = litellm.ModelResponse(
                choices=[
                    litellm.Choices(
                        message=litellm.Message(content="Hello!"),
                        index=0,
                        finish_reason="stop",
                    )
                ]
            ).model_dump()
            
            # Create a proper mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = json.dumps(mock_response_data)
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = mock_response_data
            
            mock_post.return_value = mock_response
            
            # Call with model="gpt-4" (no "dotprompt/" prefix) and prompt_id
            await litellm.acompletion(
                model="gpt-4",
                prompt_id="chat_prompt",
                prompt_variables={"user_message": "Hello world"},
                messages=[{"role": "user", "content": "This will be ignored"}],
            )
            
            mock_post.assert_called_once()
            
            # Get request body from the call (it's passed as 'data' parameter as JSON string)
            data_str = mock_post.call_args.kwargs.get("data", "{}")
            request_body = json.loads(data_str)
            
            print(f"Request body: {json.dumps(request_body, indent=2)}")
            
            # Verify the prompt was auto-detected and used
            # The chat_prompt.prompt has metadata: model: gpt-4, temperature: 0.7, max_tokens: 150
            assert request_body["model"] == "gpt-4"
            
            # Note: OpenAI API might strip out temperature/max_tokens if they're not in the request
            # The key test is that the messages were transformed
            
            # Verify the messages were transformed using the prompt template
            # chat_prompt template: "User: {{user_message}}"
            messages = request_body["messages"]
            assert len(messages) >= 1
            
            # The first message should be from the prompt template with the variable substituted
            # Template is: "User: {{user_message}}" with user_message="Hello world"
            first_message_content = messages[0]["content"]
            print(f"First message content: {first_message_content}")
            assert "Hello world" in first_message_content
    
    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks


@pytest.mark.asyncio
async def test_dotprompt_with_prompt_version():
    """
    Test that dotprompt can load and use specific prompt versions.
    Versions are stored as separate files with .v{version}.prompt naming convention.
    """
    from litellm.integrations.dotprompt import DotpromptManager
    from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

    prompt_dir = Path(__file__).parent
    dotprompt_manager = DotpromptManager(prompt_directory=str(prompt_dir))
    
    # Register the dotprompt manager in callbacks
    original_callbacks = litellm.callbacks.copy()
    litellm.callbacks = [dotprompt_manager]
    
    try:
        # Mock the HTTP handler to avoid actual API calls
        with patch("litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler.post") as mock_post:
            mock_response_data = litellm.ModelResponse(
                choices=[
                    litellm.Choices(
                        message=litellm.Message(content="Hello!"),
                        index=0,
                        finish_reason="stop",
                    )
                ]
            ).model_dump()
            
            # Create a proper mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = json.dumps(mock_response_data)
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = mock_response_data
            
            mock_post.return_value = mock_response
        
            # Test version 1
            await litellm.acompletion(
                model="gpt-3.5-turbo",
                prompt_id="chat_prompt",
                prompt_version=1,
                prompt_variables={"user_message": "Test v1"},
                messages=[],
            )
            
            assert mock_post.call_count >= 1
            data_str = mock_post.call_args.kwargs.get("data", "{}")
            request_body = json.loads(data_str)
            
            print(f"Version 1 request body: {json.dumps(request_body, indent=2)}")
            
            # Verify version 1 prompt was used
            # chat_prompt.v1.prompt has: model: gpt-3.5-turbo, temperature: 0.5, max_tokens: 100
            assert request_body["model"] == "gpt-3.5-turbo"
            
            # Verify the message contains "Version 1:" prefix from v1 template
            messages = request_body["messages"]
            assert len(messages) >= 1
            first_message_content = messages[0]["content"]
            print(f"Version 1 message: {first_message_content}")
            assert "Version 1:" in first_message_content
            assert "Test v1" in first_message_content
            
            # Reset mock for version 2 test
            mock_post.reset_mock()
        
        # Test version 2
        with patch("litellm.llms.custom_httpx.llm_http_handler.AsyncHTTPHandler.post") as mock_post:
            mock_response_data = litellm.ModelResponse(
                choices=[
                    litellm.Choices(
                        message=litellm.Message(content="Hello!"),
                        index=0,
                        finish_reason="stop",
                    )
                ]
            ).model_dump()
            
            # Create a proper mock response
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = json.dumps(mock_response_data)
            mock_response.headers = {"Content-Type": "application/json"}
            mock_response.json.return_value = mock_response_data
            
            mock_post.return_value = mock_response
            
            await litellm.acompletion(
                model="gpt-4",
                prompt_id="chat_prompt",
                prompt_version=2,
                prompt_variables={"user_message": "Test v2"},
                messages=[],
            )
            
            mock_post.assert_called_once()
            data_str = mock_post.call_args.kwargs.get("data", "{}")
            request_body = json.loads(data_str)
            
            print(f"Version 2 request body: {json.dumps(request_body, indent=2)}")
            
            # Verify version 2 prompt was used
            # chat_prompt.v2.prompt has: model: gpt-4, temperature: 0.9, max_tokens: 200
            assert request_body["model"] == "gpt-4"
            
            # Verify the message contains "Version 2:" prefix from v2 template
            messages = request_body["messages"]
            assert len(messages) >= 1
            first_message_content = messages[0]["content"]
            print(f"Version 2 message: {first_message_content}")
            assert "Version 2:" in first_message_content
            assert "Test v2" in first_message_content
    
    finally:
        # Restore original callbacks
        litellm.callbacks = original_callbacks

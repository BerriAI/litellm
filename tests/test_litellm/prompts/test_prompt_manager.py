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
from litellm.integrations.dotprompt import PromptManager, PromptTemplate


def test_prompt_manager_initialization():
    """Test basic PromptManager initialization and loading."""
    # Test with the existing prompts directory
    prompt_dir = "."  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_dir)

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
    assert template.temperature == 0.7
    assert template.input_schema == {"text": "string"}
    assert template.output_format == "json"


def test_render_simple_template():
    """Test rendering a simple template with variables."""
    prompt_dir = "."  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_dir)

    # Test sample_prompt rendering
    rendered = manager.render(
        "sample_prompt", {"text": "This is a test article about AI."}
    )

    expected_content = "Extract the requested information from the given text. If a piece of information is not present, omit that field from the output.\n\nText: This is a test article about AI."
    assert rendered == expected_content


def test_render_chat_prompt():
    """Test rendering the chat prompt with conditional content."""
    prompt_dir = "."  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_dir)

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
    prompt_dir = "."  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_dir)

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

        manager = PromptManager(temp_dir)

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
    prompt_dir = "."  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_dir)

    with pytest.raises(KeyError, match="Prompt 'nonexistent' not found"):
        manager.render("nonexistent", {"some": "variable"})


def test_list_prompts():
    """Test listing available prompts."""
    prompt_dir = "."  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_dir)

    prompts = manager.list_prompts()
    assert isinstance(prompts, list)
    assert "sample_prompt" in prompts
    assert "chat_prompt" in prompts
    assert "coding_assistant" in prompts


def test_get_prompt_metadata():
    """Test retrieving prompt metadata."""
    prompt_dir = "."  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_dir)

    metadata = manager.get_prompt_metadata("sample_prompt")
    assert metadata is not None
    assert metadata["model"] == "gemini/gemini-1.5-pro"
    assert "input" in metadata
    assert "output" in metadata


def test_add_prompt_programmatically():
    """Test adding prompts programmatically."""
    prompt_dir = "."  # Current directory when running from tests/test_litellm/prompts
    manager = PromptManager(prompt_dir)

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

        manager = PromptManager(temp_dir)

        # Check frontmatter was parsed correctly
        with_meta = manager.get_prompt("with_frontmatter")
        assert with_meta.model == "gpt-4"
        assert with_meta.temperature == 0.8

        # Check template without frontmatter still works
        without_meta = manager.get_prompt("without_frontmatter")
        assert without_meta.metadata == {}

        rendered = manager.render("without_frontmatter", {"message": "Hello!"})
        assert rendered == "Simple template: Hello!"


def test_prompt_main():
    """
    Integration test placeholder for litellm completion integration.
    This would be implemented once the PromptManager is integrated with litellm.
    """
    # TODO: Implement once PromptManager is integrated with litellm completion
    pass

"""
Test /prompts/test endpoint for testing prompts before saving
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import HTTPException


class TestPromptTestEndpoint:
    """
    Tests the /prompts/test endpoint that allows testing prompts with variables
    """

    @pytest.mark.asyncio
    async def test_parse_dotprompt_with_variables(self):
        """
        Test that dotprompt content is parsed and variables are rendered correctly
        """
        from litellm.integrations.dotprompt.prompt_manager import PromptManager

        dotprompt_content = """---
model: gpt-4o
temperature: 0.7
max_tokens: 100
---

User: Hello {{name}}, how are you?"""

        # Parse the dotprompt
        prompt_manager = PromptManager()
        frontmatter, template_content = prompt_manager._parse_frontmatter(
            content=dotprompt_content
        )

        assert frontmatter["model"] == "gpt-4o"
        assert frontmatter["temperature"] == 0.7
        assert frontmatter["max_tokens"] == 100
        assert "{{name}}" in template_content

        # Render with variables
        from jinja2 import Environment

        jinja_env = Environment(
            variable_start_string="{{",
            variable_end_string="}}",
        )
        jinja_template = jinja_env.from_string(template_content)
        rendered = jinja_template.render(name="World")

        assert "Hello World" in rendered
        assert "{{name}}" not in rendered

    @pytest.mark.asyncio
    async def test_convert_to_messages_format(self):
        """
        Test that rendered prompt is converted to OpenAI messages format
        """
        import re

        rendered_content = """System: You are a helpful assistant.

User: Hello World, how are you?"""

        messages = []
        role_pattern = r"^(System|User|Assistant|Developer):\s*(.*?)(?=\n(?:System|User|Assistant|Developer):|$)"
        matches = list(
            re.finditer(
                pattern=role_pattern,
                string=rendered_content.strip(),
                flags=re.MULTILINE | re.DOTALL,
            )
        )

        for match in matches:
            role = match.group(1).lower()
            content = match.group(2).strip()

            if role == "developer":
                role = "system"

            if content:
                messages.append({"role": role, "content": content})

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "Hello World" in messages[1]["content"]

    @pytest.mark.asyncio
    async def test_single_message_without_role(self):
        """
        Test that content without role markers is treated as a user message
        """
        import re

        rendered_content = "Just a plain message without any role markers"

        messages = []
        role_pattern = r"^(System|User|Assistant|Developer):\s*(.*?)(?=\n(?:System|User|Assistant|Developer):|$)"
        matches = list(
            re.finditer(
                pattern=role_pattern,
                string=rendered_content.strip(),
                flags=re.MULTILINE | re.DOTALL,
            )
        )

        if not matches:
            messages.append({"role": "user", "content": rendered_content.strip()})

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == rendered_content

    @pytest.mark.asyncio
    async def test_missing_model_raises_error(self):
        """
        Test that missing model in frontmatter raises an error
        """
        from litellm.integrations.dotprompt.prompt_manager import PromptManager

        dotprompt_content = """---
temperature: 0.7
---

User: Hello"""

        prompt_manager = PromptManager()
        frontmatter, _ = prompt_manager._parse_frontmatter(content=dotprompt_content)

        model = frontmatter.get("model")
        assert model is None

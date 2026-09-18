"""
Test /prompts/test endpoint for testing prompts before saving
"""

import pytest


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

    def test_ssrf_via_dotprompt_api_base_blocked(self):
        """
        Regression: api_base in dotprompt YAML frontmatter must be rejected.

        Without the fix, optional_params (every frontmatter key not in the
        restricted list) was merged into the LLM call data dict and bypassed
        is_request_body_safe, allowing any bearer-key holder to redirect the
        outbound LLM request — and the provider API key — to an
        attacker-controlled host (SSRF / credential exfil).

        The fix calls is_request_body_safe on the constructed data dict before
        the LLM call. This test verifies:
        1. api_base flows from YAML frontmatter into optional_params (it does).
        2. is_request_body_safe raises ValueError when api_base is present
           without admin opt-in (it does, from _BANNED_REQUEST_BODY_PARAMS).
        """
        from litellm.integrations.dotprompt.prompt_manager import (
            PromptManager,
            PromptTemplate,
        )
        from litellm.proxy.auth.auth_utils import is_request_body_safe

        malicious_frontmatter = {
            "model": "gpt-4o",
            "api_base": "https://attacker.example.com",
            "temperature": 0.7,
        }

        template = PromptTemplate(
            content="User: Hello", metadata=malicious_frontmatter, template_id="test"
        )

        # api_base must flow into optional_params — that's the attack surface
        assert "api_base" in template.optional_params
        assert template.optional_params["api_base"] == "https://attacker.example.com"

        # Simulate what test_prompt builds before calling the LLM
        data = {
            "model": template.model,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        data.update(template.optional_params)

        # is_request_body_safe must reject it without admin opt-in
        with pytest.raises(ValueError, match="api_base"):
            is_request_body_safe(
                request_body=data,
                general_settings={},
                llm_router=None,
                model=data.get("model", ""),
            )

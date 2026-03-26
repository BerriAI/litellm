"""
Tests that all prompt managers use Jinja2 SandboxedEnvironment to prevent SSTI attacks.

Verifies:
1. Normal template variable substitution still works
2. Malicious templates attempting to access unsafe attributes are blocked

Covers:
- litellm/integrations/dotprompt/prompt_manager.py
- litellm/integrations/arize/arize_phoenix_prompt_manager.py
- litellm/integrations/bitbucket/bitbucket_prompt_manager.py
- litellm/integrations/gitlab/gitlab_prompt_manager.py
- litellm/proxy/prompts/prompt_endpoints.py (uses dotprompt PromptManager.jinja_env)
"""

from unittest.mock import patch

import pytest
from jinja2.sandbox import SandboxedEnvironment, SecurityError

from litellm.integrations.dotprompt.prompt_manager import PromptManager


class TestDotpromptSandboxedEnvironment:
    """Test that DotpromptManager's jinja_env blocks SSTI payloads."""

    def setup_method(self):
        self.manager = PromptManager()

    def test_jinja_env_is_sandboxed(self):
        """jinja_env must be a SandboxedEnvironment instance."""
        assert isinstance(self.manager.jinja_env, SandboxedEnvironment)

    def test_normal_variable_substitution(self):
        """Normal {{ variable }} rendering should work."""
        template = self.manager.jinja_env.from_string(
            "Hello {{ name }}, you are {{ role }}."
        )
        result = template.render(name="Alice", role="admin")
        assert result == "Hello Alice, you are admin."

    def test_empty_variables(self):
        """Template with no variables should render as-is."""
        template = self.manager.jinja_env.from_string("No variables here.")
        result = template.render()
        assert result == "No variables here."

    def test_ssti_dunder_class_blocked(self):
        """Accessing __class__ on objects should be blocked."""
        template = self.manager.jinja_env.from_string(
            "{{ config.__class__.__init__.__globals__['os'].popen('id').read() }}"
        )
        with pytest.raises(SecurityError):
            template.render(config={})

    def test_ssti_dunder_mro_blocked(self):
        """Accessing __mro__ should be blocked."""
        template = self.manager.jinja_env.from_string(
            "{{ ''.__class__.__mro__[1].__subclasses__() }}"
        )
        with pytest.raises(SecurityError):
            template.render()

    def test_ssti_dunder_globals_blocked(self):
        """Accessing __globals__ should be blocked."""
        template = self.manager.jinja_env.from_string(
            "{{ request.__class__.__init__.__globals__ }}"
        )
        with pytest.raises(SecurityError):
            template.render(request="test")

    def test_ssti_getattr_bypass_blocked(self):
        """Attempting getattr-based bypass should be blocked."""
        template = self.manager.jinja_env.from_string(
            "{{ lipsum.__globals__['os'].popen('id').read() }}"
        )
        with pytest.raises(SecurityError):
            template.render()


class TestArizePhoenixSandboxedEnvironment:
    """Test that ArizePhoenixTemplateManager's jinja_env blocks SSTI payloads."""

    def setup_method(self):
        from litellm.integrations.arize.arize_phoenix_prompt_manager import (
            ArizePhoenixTemplateManager,
        )

        with patch(
            "litellm.integrations.arize.arize_phoenix_prompt_manager.ArizePhoenixClient"
        ):
            self.manager = ArizePhoenixTemplateManager(
                api_key="fake-key", api_base="https://fake.arize.com"
            )

    def test_jinja_env_is_sandboxed(self):
        assert isinstance(self.manager.jinja_env, SandboxedEnvironment)

    def test_normal_variable_substitution(self):
        template = self.manager.jinja_env.from_string("Hello {{ name }}.")
        result = template.render(name="World")
        assert result == "Hello World."

    def test_ssti_blocked(self):
        template = self.manager.jinja_env.from_string(
            "{{ ''.__class__.__mro__[1].__subclasses__() }}"
        )
        with pytest.raises(SecurityError):
            template.render()


class TestBitBucketSandboxedEnvironment:
    """Test that BitBucketTemplateManager's jinja_env blocks SSTI payloads."""

    def setup_method(self):
        from litellm.integrations.bitbucket.bitbucket_prompt_manager import (
            BitBucketTemplateManager,
        )

        with patch(
            "litellm.integrations.bitbucket.bitbucket_prompt_manager.BitBucketClient"
        ):
            self.manager = BitBucketTemplateManager(
                bitbucket_config={
                    "bitbucket_api_base": "https://api.bitbucket.org",
                    "bitbucket_workspace": "test",
                    "bitbucket_repo_slug": "test",
                }
            )

    def test_jinja_env_is_sandboxed(self):
        assert isinstance(self.manager.jinja_env, SandboxedEnvironment)

    def test_normal_variable_substitution(self):
        template = self.manager.jinja_env.from_string("Hello {{ name }}.")
        result = template.render(name="World")
        assert result == "Hello World."

    def test_ssti_blocked(self):
        template = self.manager.jinja_env.from_string(
            "{{ ''.__class__.__mro__[1].__subclasses__() }}"
        )
        with pytest.raises(SecurityError):
            template.render()


class TestGitLabSandboxedEnvironment:
    """Test that GitLabTemplateManager's jinja_env blocks SSTI payloads."""

    def setup_method(self):
        from litellm.integrations.gitlab.gitlab_prompt_manager import (
            GitLabTemplateManager,
        )

        with patch(
            "litellm.integrations.gitlab.gitlab_prompt_manager.GitLabClient"
        ):
            self.manager = GitLabTemplateManager(
                gitlab_config={
                    "project": "test/repo",
                    "access_token": "fake-token",
                }
            )

    def test_jinja_env_is_sandboxed(self):
        assert isinstance(self.manager.jinja_env, SandboxedEnvironment)

    def test_normal_variable_substitution(self):
        template = self.manager.jinja_env.from_string("Hello {{ name }}.")
        result = template.render(name="World")
        assert result == "Hello World."

    def test_ssti_blocked(self):
        template = self.manager.jinja_env.from_string(
            "{{ ''.__class__.__mro__[1].__subclasses__() }}"
        )
        with pytest.raises(SecurityError):
            template.render()

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.fireworks_ai.messages.transformation import (
    FireworksAIMessagesConfig,
)


@pytest.fixture
def config():
    return FireworksAIMessagesConfig()


class TestValidateAnthropicMessagesEnvironment:
    """Tests for validate_anthropic_messages_environment."""

    def test_should_set_api_key_from_explicit_param(self, config):
        headers, api_base = config.validate_anthropic_messages_environment(
            headers={},
            model="claude-3-5-sonnet",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="explicit-key",
        )
        assert headers["Authorization"] == "Bearer explicit-key"

    def test_should_set_api_key_from_fireworks_api_key_env(self, config):
        with patch(
            "litellm.llms.fireworks_ai.messages.transformation.get_secret_str",
            side_effect=lambda key: "env-key" if key == "FIREWORKS_API_KEY" else None,
        ):
            headers, _ = config.validate_anthropic_messages_environment(
                headers={},
                model="claude-3-5-sonnet",
                messages=[],
                optional_params={},
                litellm_params={},
            )
            assert headers["Authorization"] == "Bearer env-key"

    def test_should_set_api_key_from_fireworks_ai_api_key_env(self, config):
        with patch(
            "litellm.llms.fireworks_ai.messages.transformation.get_secret_str",
            side_effect=lambda key: (
                "ai-env-key" if key == "FIREWORKS_AI_API_KEY" else None
            ),
        ):
            headers, _ = config.validate_anthropic_messages_environment(
                headers={},
                model="claude-3-5-sonnet",
                messages=[],
                optional_params={},
                litellm_params={},
            )
            assert headers["Authorization"] == "Bearer ai-env-key"

    def test_should_not_overwrite_existing_authorization(self, config):
        headers, _ = config.validate_anthropic_messages_environment(
            headers={"Authorization": "Bearer pre-existing"},
            model="claude-3-5-sonnet",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="new-key",
        )
        assert headers["Authorization"] == "Bearer pre-existing"

    def test_should_set_default_content_type(self, config):
        headers, _ = config.validate_anthropic_messages_environment(
            headers={},
            model="claude-3-5-sonnet",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="key",
        )
        assert headers["content-type"] == "application/json"
        assert "anthropic-version" not in headers

    def test_should_not_overwrite_existing_content_type(self, config):
        headers, _ = config.validate_anthropic_messages_environment(
            headers={"content-type": "application/xml"},
            model="claude-3-5-sonnet",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="key",
        )
        assert headers["content-type"] == "application/xml"

    def test_should_pass_through_api_base(self, config):
        _, api_base = config.validate_anthropic_messages_environment(
            headers={},
            model="claude-3-5-sonnet",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="key",
            api_base="https://custom.example.com",
        )
        assert api_base == "https://custom.example.com"


class TestGetCompleteUrl:
    """Tests for get_complete_url."""

    def test_should_use_default_base_url(self, config):
        with patch(
            "litellm.llms.fireworks_ai.messages.transformation.get_secret_str",
            return_value=None,
        ):
            url = config.get_complete_url(
                api_base=None,
                api_key=None,
                model="claude-3-5-sonnet",
                optional_params={},
                litellm_params={},
            )
            assert url == "https://api.fireworks.ai/inference/v1/messages"

    def test_should_append_messages_to_v1_base(self, config):
        url = config.get_complete_url(
            api_base="https://custom.example.com/v1",
            api_key=None,
            model="claude-3-5-sonnet",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.example.com/v1/messages"

    def test_should_append_v1_messages_to_bare_base(self, config):
        url = config.get_complete_url(
            api_base="https://custom.example.com/api",
            api_key=None,
            model="claude-3-5-sonnet",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.example.com/api/v1/messages"

    def test_should_not_duplicate_v1_messages_suffix(self, config):
        url = config.get_complete_url(
            api_base="https://custom.example.com/v1/messages",
            api_key=None,
            model="claude-3-5-sonnet",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://custom.example.com/v1/messages"

    def test_should_strip_trailing_slash(self, config):
        url = config.get_complete_url(
            api_base="https://api.fireworks.ai/inference/v1/",
            api_key=None,
            model="claude-3-5-sonnet",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://api.fireworks.ai/inference/v1/messages"

    def test_should_use_fireworks_api_base_env(self, config):
        with patch(
            "litellm.llms.fireworks_ai.messages.transformation.get_secret_str",
            side_effect=lambda key: (
                "https://env-base.example.com/v1"
                if key == "FIREWORKS_API_BASE"
                else None
            ),
        ):
            url = config.get_complete_url(
                api_base=None,
                api_key=None,
                model="claude-3-5-sonnet",
                optional_params={},
                litellm_params={},
            )
            assert url == "https://env-base.example.com/v1/messages"

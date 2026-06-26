"""Unit tests for system role message normalization in AnthropicMessagesConfig.

Covers the should_normalize_system_role_messages() hook, _as_system_content_blocks(),
and _normalize_system_role_messages() methods.
"""

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)


class TestAsSystemContentBlocks:
    """Test the _as_system_content_blocks helper."""

    def test_none_returns_empty_list(self):
        result = AnthropicMessagesConfig._as_system_content_blocks(None)
        assert result == []

    def test_string_returns_text_block(self):
        result = AnthropicMessagesConfig._as_system_content_blocks("hello")
        assert result == [{"type": "text", "text": "hello"}]

    def test_list_returns_copy(self):
        original = [{"type": "text", "text": "block1"}, {"type": "text", "text": "block2"}]
        result = AnthropicMessagesConfig._as_system_content_blocks(original)
        assert result == original
        assert result is not original

    def test_dict_wraps_in_list(self):
        block = {"type": "text", "text": "wrapped"}
        result = AnthropicMessagesConfig._as_system_content_blocks(block)
        assert result == [block]


class TestNormalizeSystemRoleMessages:
    """Test the _normalize_system_role_messages method."""

    def _make_config(self):
        return AnthropicMessagesConfig()

    def test_no_system_role_messages_is_noop(self):
        config = self._make_config()
        messages = [{"role": "user", "content": "hello"}]
        params = {}

        result_messages, result_params = config._normalize_system_role_messages(messages, params)

        assert result_messages == messages
        assert result_params == {}

    def test_single_system_message_moved_to_top_level(self):
        config = self._make_config()
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "hello"},
        ]
        params = {}

        result_messages, result_params = config._normalize_system_role_messages(messages, params)

        assert result_messages == [{"role": "user", "content": "hello"}]
        assert result_params["system"] == [{"type": "text", "text": "You are helpful"}]

    def test_existing_system_merged_with_system_messages(self):
        config = self._make_config()
        messages = [
            {"role": "system", "content": "Extra instruction"},
            {"role": "user", "content": "hello"},
        ]
        params = {"system": "Base instruction"}

        result_messages, result_params = config._normalize_system_role_messages(messages, params)

        assert result_messages == [{"role": "user", "content": "hello"}]
        assert result_params["system"] == [
            {"type": "text", "text": "Base instruction"},
            {"type": "text", "text": "Extra instruction"},
        ]

    def test_multiple_system_messages_all_moved(self):
        config = self._make_config()
        messages = [
            {"role": "system", "content": "First"},
            {"role": "user", "content": "hi"},
            {"role": "system", "content": "Second"},
        ]
        params = {}

        result_messages, result_params = config._normalize_system_role_messages(messages, params)

        assert result_messages == [{"role": "user", "content": "hi"}]
        assert result_params["system"] == [
            {"type": "text", "text": "First"},
            {"type": "text", "text": "Second"},
        ]

    def test_system_with_list_content_preserved(self):
        config = self._make_config()
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Block content"}],
            },
            {"role": "user", "content": "hello"},
        ]
        params = {}

        result_messages, result_params = config._normalize_system_role_messages(messages, params)

        assert result_messages == [{"role": "user", "content": "hello"}]
        assert result_params["system"] == [{"type": "text", "text": "Block content"}]

    def test_existing_system_as_list_merged_correctly(self):
        config = self._make_config()
        messages = [
            {"role": "system", "content": "Extra"},
            {"role": "user", "content": "hello"},
        ]
        params = {"system": [{"type": "text", "text": "Existing block"}]}

        result_messages, result_params = config._normalize_system_role_messages(messages, params)

        assert result_messages == [{"role": "user", "content": "hello"}]
        assert result_params["system"] == [
            {"type": "text", "text": "Existing block"},
            {"type": "text", "text": "Extra"},
        ]

    def test_empty_system_content_removes_system_param(self):
        config = self._make_config()
        messages = [
            {"role": "system", "content": None},
            {"role": "user", "content": "hello"},
        ]
        params = {}

        result_messages, result_params = config._normalize_system_role_messages(messages, params)

        assert result_messages == [{"role": "user", "content": "hello"}]
        assert "system" not in result_params

    def test_non_dict_messages_preserved(self):
        config = self._make_config()
        messages = [
            {"role": "system", "content": "sys"},
            "not a dict",
            {"role": "user", "content": "hello"},
        ]
        params = {}

        result_messages, result_params = config._normalize_system_role_messages(messages, params)

        assert result_messages == [
            "not a dict",
            {"role": "user", "content": "hello"},
        ]
        assert result_params["system"] == [{"type": "text", "text": "sys"}]


class TestShouldNormalizeSystemRoleMessages:
    """Test the opt-in flag defaults and overrides."""

    def test_base_class_defaults_to_false(self):
        config = AnthropicMessagesConfig()
        assert config.should_normalize_system_role_messages() is False

    def test_vertex_ai_overrides_to_true(self):
        from litellm.llms.vertex_ai.vertex_ai_partner_models.anthropic.experimental_pass_through.transformation import (
            VertexAIPartnerModelsAnthropicMessagesConfig,
        )

        config = VertexAIPartnerModelsAnthropicMessagesConfig()
        assert config.should_normalize_system_role_messages() is True

    def test_azure_ai_overrides_to_true(self):
        from litellm.llms.azure_ai.anthropic.messages_transformation import (
            AzureAnthropicMessagesConfig,
        )

        config = AzureAnthropicMessagesConfig()
        assert config.should_normalize_system_role_messages() is True

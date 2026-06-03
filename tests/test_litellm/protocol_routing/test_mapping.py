import pytest
from litellm.protocol_routing import (
    SupportedProtocol,
    PROVIDER_DEFAULT_PROTOCOLS,
    infer_protocols,
)


class TestProviderDefaultProtocols:
    """Test PROVIDER_DEFAULT_PROTOCOLS mapping."""

    def test_openai_supports_chat_and_responses(self):
        """Verify OpenAI supports both Chat and Responses protocols."""
        protocols = PROVIDER_DEFAULT_PROTOCOLS["openai"]
        assert SupportedProtocol.OPENAI_CHAT in protocols
        assert SupportedProtocol.OPENAI_RESPONSES in protocols

    def test_anthropic_supports_messages_only(self):
        """Verify Anthropic only supports Messages protocol."""
        protocols = PROVIDER_DEFAULT_PROTOCOLS["anthropic"]
        assert protocols == [SupportedProtocol.ANTHROPIC_MESSAGES]

    def test_azure_supports_chat_and_responses(self):
        """Verify Azure supports both Chat and Responses protocols."""
        protocols = PROVIDER_DEFAULT_PROTOCOLS["azure"]
        assert SupportedProtocol.OPENAI_CHAT in protocols
        assert SupportedProtocol.OPENAI_RESPONSES in protocols

    def test_bedrock_supports_messages(self):
        """Verify Bedrock supports Anthropic Messages protocol."""
        protocols = PROVIDER_DEFAULT_PROTOCOLS["bedrock"]
        assert SupportedProtocol.ANTHROPIC_MESSAGES in protocols

    def test_gemini_supports_generate_content(self):
        """Verify Gemini supports Generate Content protocol."""
        protocols = PROVIDER_DEFAULT_PROTOCOLS["gemini"]
        assert SupportedProtocol.GOOGLE_GENERATE_CONTENT in protocols

    def test_all_major_providers_present(self):
        """Verify all major providers are in the mapping."""
        major_providers = [
            "openai",
            "anthropic",
            "azure",
            "bedrock",
            "vertex_ai",
            "gemini",
        ]
        for provider in major_providers:
            assert provider in PROVIDER_DEFAULT_PROTOCOLS, f"Missing provider: {provider}"


class TestInferProtocols:
    """Test infer_protocols function."""

    def test_infer_with_known_provider(self):
        """Verify inference with a known provider."""
        protocols = infer_protocols("openai")
        assert SupportedProtocol.OPENAI_CHAT in protocols
        assert SupportedProtocol.OPENAI_RESPONSES in protocols

    def test_infer_with_anthropic(self):
        """Verify inference with Anthropic provider."""
        protocols = infer_protocols("anthropic")
        assert protocols == [SupportedProtocol.ANTHROPIC_MESSAGES]

    def test_infer_with_unknown_provider(self):
        """Verify inference with unknown provider falls back to OpenAI Chat (safe default)."""
        protocols = infer_protocols("unknown_provider_xyz")
        assert protocols == [SupportedProtocol.OPENAI_CHAT]

    def test_infer_with_none_provider(self):
        """Verify inference with None provider falls back to OpenAI Chat (safe default)."""
        protocols = infer_protocols(None)
        assert protocols == [SupportedProtocol.OPENAI_CHAT]

    def test_infer_with_explicit_override(self):
        """Verify explicit supported_protocols in model_info overrides defaults."""
        model_info = {
            "supported_protocols": [
                SupportedProtocol.ANTHROPIC_MESSAGES,
                SupportedProtocol.OPENAI_CHAT,
            ]
        }
        protocols = infer_protocols("openai", model_info)
        assert protocols == [
            SupportedProtocol.ANTHROPIC_MESSAGES,
            SupportedProtocol.OPENAI_CHAT,
        ]

    def test_infer_with_explicit_override_string_values(self):
        """Verify explicit override works with string values."""
        model_info = {
            "supported_protocols": ["anthropic_messages", "openai_chat"]
        }
        protocols = infer_protocols("openai", model_info)
        assert SupportedProtocol.ANTHROPIC_MESSAGES in protocols
        assert SupportedProtocol.OPENAI_CHAT in protocols

    def test_infer_with_empty_model_info(self):
        """Verify inference with empty model_info falls back to provider."""
        protocols = infer_protocols("anthropic", {})
        assert protocols == [SupportedProtocol.ANTHROPIC_MESSAGES]

    def test_infer_with_model_info_no_supported_protocols(self):
        """Verify inference with model_info but no supported_protocols field."""
        model_info = {"some_other_field": "value"}
        protocols = infer_protocols("gemini", model_info)
        assert SupportedProtocol.GOOGLE_GENERATE_CONTENT in protocols

    def test_infer_returns_copy_not_reference(self):
        """Verify returned list is a copy, not a reference to the mapping."""
        protocols1 = infer_protocols("openai")
        protocols2 = infer_protocols("openai")

        # Modify the first list
        protocols1.append(SupportedProtocol.ANTHROPIC_MESSAGES)

        # Second list should not be affected
        assert SupportedProtocol.ANTHROPIC_MESSAGES not in protocols2

    def test_infer_with_multiple_providers(self):
        """Verify inference works correctly for multiple providers."""
        test_cases = [
            ("openai", [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.OPENAI_RESPONSES]),
            ("anthropic", [SupportedProtocol.ANTHROPIC_MESSAGES]),
            ("azure", [SupportedProtocol.OPENAI_CHAT, SupportedProtocol.OPENAI_RESPONSES]),
            ("bedrock", [SupportedProtocol.ANTHROPIC_MESSAGES]),
            ("gemini", [SupportedProtocol.GOOGLE_GENERATE_CONTENT]),
        ]

        for provider, expected_protocols in test_cases:
            protocols = infer_protocols(provider)
            for expected in expected_protocols:
                assert expected in protocols, f"Provider {provider} missing {expected}"

"""
Unit tests for codestral provider routing.

These tests verify that the chat and FIM endpoints for codestral
are correctly routed to different providers:
- Chat endpoint -> codestral provider
- FIM endpoint -> text-completion-codestral provider

Related issue: https://github.com/BerriAI/litellm/issues/18464
"""
import pytest

import litellm


class TestCodestralProviderRouting:
    """Tests for codestral endpoint routing in get_llm_provider"""

    def test_codestral_chat_endpoint_routes_to_codestral_provider(self):
        """
        Test that the codestral chat endpoint routes to the 'codestral' provider.

        The chat/completions endpoint should be handled by the codestral provider.
        """
        model, custom_llm_provider, _, api_base = litellm.get_llm_provider(
            model="codestral-latest",
            api_base="https://codestral.mistral.ai/v1/chat/completions",
        )

        assert custom_llm_provider == "codestral"

    def test_codestral_fim_endpoint_routes_to_text_completion_provider(self):
        """
        Test that the codestral FIM endpoint routes to 'text-completion-codestral'.

        The fim/completions endpoint should be handled by the
        text-completion-codestral provider for fill-in-the-middle completions.
        """
        model, custom_llm_provider, _, api_base = litellm.get_llm_provider(
            model="codestral-latest",
            api_base="https://codestral.mistral.ai/v1/fim/completions",
        )

        assert custom_llm_provider == "text-completion-codestral"

    def test_codestral_endpoints_are_different_providers(self):
        """
        Test that chat and FIM endpoints route to different providers.

        This is the core fix for issue #18464 - previously both endpoints
        would route to 'codestral' due to duplicate conditions.
        """
        _, chat_provider, _, _ = litellm.get_llm_provider(
            model="codestral-latest",
            api_base="https://codestral.mistral.ai/v1/chat/completions",
        )

        _, fim_provider, _, _ = litellm.get_llm_provider(
            model="codestral-latest",
            api_base="https://codestral.mistral.ai/v1/fim/completions",
        )

        assert chat_provider != fim_provider
        assert chat_provider == "codestral"
        assert fim_provider == "text-completion-codestral"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

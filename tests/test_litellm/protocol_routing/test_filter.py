import pytest
from litellm.protocol_routing import (
    SupportedProtocol,
    ProtocolMismatchError,
    filter_deployments_by_protocol,
    check_strict_protocol_for_provider,
    get_protocol_routing_mode,
    set_protocol_routing_mode,
)


@pytest.fixture
def openai_deployment():
    """OpenAI deployment fixture."""
    return {
        "model_name": "gpt-4o",
        "litellm_params": {
            "model": "openai/gpt-4o",
            "custom_llm_provider": "openai",
            "api_key": "sk-test",
        },
        "model_info": {"id": "openai-1"},
    }


@pytest.fixture
def anthropic_deployment():
    """Anthropic deployment fixture."""
    return {
        "model_name": "claude-3",
        "litellm_params": {
            "model": "anthropic/claude-3-sonnet",
            "custom_llm_provider": "anthropic",
            "api_key": "sk-ant-test",
        },
        "model_info": {"id": "anthropic-1"},
    }


@pytest.fixture
def bedrock_deployment():
    """Bedrock deployment fixture."""
    return {
        "model_name": "claude-3-bedrock",
        "litellm_params": {
            "model": "bedrock/anthropic.claude-3-sonnet",
            "custom_llm_provider": "bedrock",
            "api_key": "test",
        },
        "model_info": {"id": "bedrock-1"},
    }


@pytest.fixture
def gemini_deployment():
    """Gemini deployment fixture."""
    return {
        "model_name": "gemini-pro",
        "litellm_params": {
            "model": "gemini/gemini-pro",
            "custom_llm_provider": "gemini",
            "api_key": "test",
        },
        "model_info": {"id": "gemini-1"},
    }


class TestFilterDeploymentsByProtocol:
    """Test filter_deployments_by_protocol function."""

    def test_bridged_mode_returns_all_deployments(self, openai_deployment, anthropic_deployment):
        """Verify bridged mode returns all deployments without filtering."""
        deployments = [openai_deployment, anthropic_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type="anthropic_messages",
            model="test-model",
            mode="bridged",
        )
        assert len(result) == 2
        assert result == deployments

    def test_bridged_mode_is_default(self, openai_deployment, anthropic_deployment, monkeypatch):
        """Verify bridged mode is the default behavior when no env override is set.

        The module-level _protocol_routing_mode is initialized from env at import
        time. Patch it to "bridged" to simulate a clean-import state.
        """
        from litellm.protocol_routing import _types
        monkeypatch.setattr(_types, "_protocol_routing_mode", "bridged")
        deployments = [openai_deployment, anthropic_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type="anthropic_messages",
            model="test-model",
        )
        assert len(result) == 2

    def test_strict_mode_filters_anthropic_messages(self, openai_deployment, anthropic_deployment):
        """Verify strict mode filters to Anthropic-compatible deployments."""
        deployments = [openai_deployment, anthropic_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type="anthropic_messages",
            model="test-model",
            mode="strict",
        )
        assert len(result) == 1
        assert result[0] == anthropic_deployment

    def test_strict_mode_filters_openai_chat(self, openai_deployment, anthropic_deployment):
        """Verify strict mode filters to OpenAI Chat-compatible deployments."""
        deployments = [openai_deployment, anthropic_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type="acompletion",
            model="test-model",
            mode="strict",
        )
        assert len(result) == 1
        assert result[0] == openai_deployment

    def test_strict_mode_filters_openai_responses(self, openai_deployment, anthropic_deployment):
        """Verify strict mode filters to OpenAI Responses-compatible deployments."""
        deployments = [openai_deployment, anthropic_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type="aresponses",
            model="test-model",
            mode="strict",
        )
        assert len(result) == 1
        assert result[0] == openai_deployment

    def test_strict_mode_filters_gemini(self, openai_deployment, gemini_deployment):
        """Verify strict mode filters to Gemini-compatible deployments."""
        deployments = [openai_deployment, gemini_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type="agenerate_content",
            model="test-model",
            mode="strict",
        )
        assert len(result) == 1
        assert result[0] == gemini_deployment

    def test_strict_mode_raises_on_no_match(self, openai_deployment):
        """Verify strict mode raises ProtocolMismatchError when no deployments match."""
        deployments = [openai_deployment]
        with pytest.raises(ProtocolMismatchError) as exc_info:
            filter_deployments_by_protocol(
                deployments,
                route_type="anthropic_messages",
                model="gpt-4",
                mode="strict",
            )

        error = exc_info.value
        assert error.model == "gpt-4"
        assert error.requested_protocol == "anthropic_messages"
        assert "openai_chat" in error.available_protocols
        assert "openai_responses" in error.available_protocols

    def test_strict_mode_with_empty_deployments(self):
        """Verify strict mode handles empty deployment list."""
        result = filter_deployments_by_protocol(
            [],
            route_type="anthropic_messages",
            model="test-model",
            mode="strict",
        )
        assert result == []

    def test_strict_mode_with_explicit_supported_protocols(self, openai_deployment):
        """Verify strict mode respects explicit supported_protocols in model_info."""
        # Override openai deployment to support anthropic_messages
        openai_deployment["model_info"]["supported_protocols"] = [
            SupportedProtocol.ANTHROPIC_MESSAGES
        ]

        deployments = [openai_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type="anthropic_messages",
            model="test-model",
            mode="strict",
        )
        assert len(result) == 1
        assert result[0] == openai_deployment

    def test_strict_mode_with_multiple_compatible_deployments(
        self, openai_deployment, bedrock_deployment
    ):
        """Verify strict mode returns all compatible deployments."""
        # Both OpenAI and Bedrock support openai_chat
        deployments = [openai_deployment, bedrock_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type="acompletion",
            model="test-model",
            mode="strict",
        )
        # Both should be included (Bedrock supports openai_chat via Nova)
        assert len(result) == 2

    def test_unknown_route_type_passes_through(self, openai_deployment):
        """Verify unknown route types pass through without filtering."""
        deployments = [openai_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type="unknown_route_type",
            model="test-model",
            mode="strict",
        )
        assert len(result) == 1

    def test_none_route_type_passes_through(self, openai_deployment):
        """Verify None route_type passes through without filtering."""
        deployments = [openai_deployment]
        result = filter_deployments_by_protocol(
            deployments,
            route_type=None,
            model="test-model",
            mode="strict",
        )
        assert len(result) == 1

    def test_mode_override_takes_precedence(self, openai_deployment, anthropic_deployment):
        """Verify explicit mode parameter overrides global setting."""
        original_mode = get_protocol_routing_mode()
        try:
            # Set global to strict
            set_protocol_routing_mode("strict")

            deployments = [openai_deployment, anthropic_deployment]

            # Override to bridged
            result = filter_deployments_by_protocol(
                deployments,
                route_type="anthropic_messages",
                model="test-model",
                mode="bridged",
            )
            # Should return all deployments (bridged mode)
            assert len(result) == 2
        finally:
            set_protocol_routing_mode(original_mode)


class TestExtractProviderFallback:
    """Regression: deployments commonly leave custom_llm_provider unset and
    encode the provider in the model prefix instead."""

    def test_strict_filter_uses_model_prefix_when_custom_llm_provider_missing(self):
        from litellm.protocol_routing._filter import filter_deployments_by_protocol
        from litellm.protocol_routing._types import set_protocol_routing_mode

        deployments = [
            {
                "model_name": "minimax-m2.5",
                "litellm_params": {"model": "openai/MiniMax-M2.5"},
                "model_info": {"id": "openai-1"},
            },
            {
                "model_name": "minimax-m2.5",
                "litellm_params": {"model": "anthropic/MiniMax-M2.5"},
                "model_info": {"id": "anthropic-1"},
            },
        ]

        original = "bridged"
        try:
            set_protocol_routing_mode("strict")

            openai_only = filter_deployments_by_protocol(
                deployments, "acompletion", "minimax-m2.5"
            )
            assert len(openai_only) == 1
            assert openai_only[0]["model_info"]["id"] == "openai-1"

            anthropic_only = filter_deployments_by_protocol(
                deployments, "anthropic_messages", "minimax-m2.5"
            )
            assert len(anthropic_only) == 1
            assert anthropic_only[0]["model_info"]["id"] == "anthropic-1"
        finally:
            set_protocol_routing_mode(original)


class TestCheckStrictProtocolForProvider:
    """Test check_strict_protocol_for_provider function."""

    def test_bridged_mode_allows_any_provider(self):
        """Verify bridged mode allows any provider."""
        # Should not raise
        check_strict_protocol_for_provider(
            provider="openai",
            requested_protocol=SupportedProtocol.ANTHROPIC_MESSAGES,
            model="test-model",
            mode="bridged",
        )

    def test_strict_mode_allows_compatible_provider(self):
        """Verify strict mode allows compatible provider."""
        # Should not raise
        check_strict_protocol_for_provider(
            provider="anthropic",
            requested_protocol=SupportedProtocol.ANTHROPIC_MESSAGES,
            model="test-model",
            mode="strict",
        )

    def test_strict_mode_raises_on_incompatible_provider(self):
        """Verify strict mode raises on incompatible provider."""
        with pytest.raises(ProtocolMismatchError) as exc_info:
            check_strict_protocol_for_provider(
                provider="openai",
                requested_protocol=SupportedProtocol.ANTHROPIC_MESSAGES,
                model="gpt-4",
                mode="strict",
            )

        error = exc_info.value
        assert error.model == "gpt-4"
        assert error.requested_protocol == "anthropic_messages"
        assert "openai_chat" in error.available_protocols

    def test_strict_mode_with_none_provider(self):
        """Verify strict mode raises on None provider."""
        with pytest.raises(ProtocolMismatchError):
            check_strict_protocol_for_provider(
                provider=None,
                requested_protocol=SupportedProtocol.ANTHROPIC_MESSAGES,
                model="test-model",
                mode="strict",
            )

    def test_strict_mode_with_unknown_provider(self):
        """Verify strict mode rejects unknown provider for non-OpenAI Chat protocols.

        Unknown providers fall back to OPENAI_CHAT only, so any other protocol fails.
        """
        with pytest.raises(ProtocolMismatchError):
            check_strict_protocol_for_provider(
                provider="unknown_provider_xyz",
                requested_protocol=SupportedProtocol.ANTHROPIC_MESSAGES,
                model="test-model",
                mode="strict",
            )

    def test_strict_mode_unknown_provider_allows_openai_chat(self):
        """Verify strict mode allows OPENAI_CHAT for unknown provider (safe fallback)."""
        check_strict_protocol_for_provider(
            provider="unknown_provider_xyz",
            requested_protocol=SupportedProtocol.OPENAI_CHAT,
            model="test-model",
            mode="strict",
        )

    def test_mode_override_takes_precedence(self):
        """Verify explicit mode parameter overrides global setting."""
        original_mode = get_protocol_routing_mode()
        try:
            # Set global to strict
            set_protocol_routing_mode("strict")

            # Override to bridged - should not raise
            check_strict_protocol_for_provider(
                provider="openai",
                requested_protocol=SupportedProtocol.ANTHROPIC_MESSAGES,
                model="test-model",
                mode="bridged",
            )
        finally:
            set_protocol_routing_mode(original_mode)

    def test_strict_mode_allows_bedrock_for_anthropic(self):
        """Verify strict mode allows Bedrock for Anthropic Messages."""
        # Should not raise
        check_strict_protocol_for_provider(
            provider="bedrock",
            requested_protocol=SupportedProtocol.ANTHROPIC_MESSAGES,
            model="test-model",
            mode="strict",
        )

    def test_strict_mode_allows_gemini_for_generate_content(self):
        """Verify strict mode allows Gemini for Generate Content."""
        # Should not raise
        check_strict_protocol_for_provider(
            provider="gemini",
            requested_protocol=SupportedProtocol.GOOGLE_GENERATE_CONTENT,
            model="test-model",
            mode="strict",
        )

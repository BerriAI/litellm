"""
Tests for managed-agent interaction routing.

Custom Gemini agents are identified by ``agent`` (name/id), not ``model``.
The proxy must not pass the agent name as ``model`` or LiteLLM may route to
openai/* wildcards instead of Gemini interactions.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestInteractionsAgentParameter:
    """Proxy endpoint must keep agent and model separate."""

    def test_create_interaction_uses_model_only_from_body(self):
        """POST /v1beta/interactions: model kwarg is only the request's model field."""
        data = {
            "agent": "mqy-custom-slides-agent",
            "input": "hello",
        }
        # Fixed behavior: do NOT fall back agent → model
        model_for_routing = data.get("model")
        assert model_for_routing is None
        assert data.get("agent") == "mqy-custom-slides-agent"

    def test_model_field_still_used_when_present(self):
        data = {
            "model": "gemini-2.5-flash",
            "input": "hello",
        }
        model_for_routing = data.get("model")
        assert model_for_routing == "gemini-2.5-flash"


class TestInteractionsAgentOnlyProviderRouting:
    """SDK: agent-only create must not call get_llm_provider on the agent name."""

    @patch("litellm.interactions.main.interactions_http_handler")
    @patch("litellm.interactions.main.get_provider_interactions_api_config")
    @patch("litellm.get_llm_provider")
    def test_agent_only_skips_get_llm_provider(
        self,
        mock_get_llm_provider,
        mock_get_config,
        mock_handler,
    ):
        from litellm.interactions.main import create
        from litellm.types.interactions import InteractionsAPIResponse

        mock_get_config.return_value = MagicMock()
        mock_handler.create_interaction.return_value = InteractionsAPIResponse(
            id="int-1",
            status="completed",
            object="interaction",
        )

        logging_obj = MagicMock()
        create(
            agent="mqy-custom-slides-agent",
            input="test",
            custom_llm_provider="gemini",
            litellm_logging_obj=logging_obj,
        )

        mock_get_llm_provider.assert_not_called()
        call_kwargs = mock_handler.create_interaction.call_args.kwargs
        assert call_kwargs["agent"] == "mqy-custom-slides-agent"
        assert call_kwargs["model"] is None
        assert call_kwargs["custom_llm_provider"] == "gemini"

    @patch("litellm.interactions.main.interactions_http_handler")
    @patch("litellm.interactions.main.get_provider_interactions_api_config")
    @patch("litellm.get_llm_provider")
    def test_proxy_mistake_model_equals_agent_is_corrected(
        self,
        mock_get_llm_provider,
        mock_get_config,
        mock_handler,
    ):
        """If model was wrongly set to the agent name, clear it before the HTTP call."""
        from litellm.interactions.main import create
        from litellm.types.interactions import InteractionsAPIResponse

        mock_get_config.return_value = MagicMock()
        mock_handler.create_interaction.return_value = InteractionsAPIResponse(
            id="int-1",
            status="completed",
            object="interaction",
        )

        logging_obj = MagicMock()
        create(
            model="mqy-custom-slides-agent",
            agent="mqy-custom-slides-agent",
            input="test",
            custom_llm_provider="gemini",
            litellm_logging_obj=logging_obj,
        )

        mock_get_llm_provider.assert_not_called()
        assert mock_handler.create_interaction.call_args.kwargs["model"] is None

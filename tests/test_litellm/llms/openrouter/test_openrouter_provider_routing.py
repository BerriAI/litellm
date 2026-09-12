"""
Tests for OpenRouter model name routing in get_llm_provider.

OpenRouter-native models have IDs that start with "openrouter/" (e.g.
openrouter/auto, openrouter/free, openrouter/aurora-alpha).  When a user
configures such a model in LiteLLM they use the double-prefixed form
"openrouter/openrouter/aurora-alpha".  get_llm_provider must strip only
the outer "openrouter/" provider prefix and leave the inner one intact,
so the correct model ID is sent to the OpenRouter API.

See: https://github.com/BerriAI/litellm/issues/16353
"""


import pytest


import litellm


class TestOpenRouterNativeModelRouting:
    """get_llm_provider must not double-strip native OpenRouter model names."""

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            # Well-known native models
            ("openrouter/openrouter/auto", "openrouter/auto"),
            ("openrouter/openrouter/free", "openrouter/free"),
            ("openrouter/openrouter/bodybuilder", "openrouter/bodybuilder"),
            # Arbitrary native models — the fix must be pattern-based, not a hardcoded list
            ("openrouter/openrouter/aurora-alpha", "openrouter/aurora-alpha"),
            ("openrouter/openrouter/polaris-alpha", "openrouter/polaris-alpha"),
            ("openrouter/openrouter/some-future-model", "openrouter/some-future-model"),
        ],
    )
    def test_double_prefixed_strips_once(self, input_model, expected_model):
        """openrouter/openrouter/<model> should yield model=openrouter/<model>."""
        result_model, provider, _, _ = litellm.get_llm_provider(model=input_model)
        assert provider == "openrouter"
        assert result_model == expected_model

    @pytest.mark.parametrize(
        "input_model",
        [
            "openrouter/openrouter/aurora-alpha",
            "openrouter/openrouter/auto",
            "openrouter/openrouter/free",
            "openrouter/openrouter/some-future-model",
        ],
    )
    def test_bridge_double_call_preserves_native_model(self, input_model):
        """Simulates two consecutive get_llm_provider calls (bridge → completion).

        The first call (bridge) strips the outer prefix:
            openrouter/openrouter/<model> → openrouter/<model>

        The second call (completion) receives custom_llm_provider="openrouter"
        from the bridge, detects the native model, and preserves it:
            openrouter/<model> → openrouter/<model>  (no further stripping)
        """
        # First call: bridge resolves provider
        model_first, provider, _, _ = litellm.get_llm_provider(model=input_model)
        assert provider == "openrouter"
        expected_model = input_model.split("/", 1)[1]  # openrouter/<model>
        assert model_first == expected_model

        # Second call: completion receives model + custom_llm_provider from bridge
        model_second, provider2, _, _ = litellm.get_llm_provider(
            model=model_first,
            custom_llm_provider="openrouter",
        )
        assert provider2 == "openrouter"
        assert model_second == expected_model  # preserved, not stripped further

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            ("openrouter/anthropic/claude-3-haiku", "anthropic/claude-3-haiku"),
            (
                "openrouter/meta-llama/llama-3-70b-instruct",
                "meta-llama/llama-3-70b-instruct",
            ),
        ],
    )
    def test_regular_models_still_strip_normally(self, input_model, expected_model):
        """Non-native OpenRouter models should still have their prefix stripped."""
        result_model, provider, _, _ = litellm.get_llm_provider(model=input_model)
        assert provider == "openrouter"
        assert result_model == expected_model

    def test_wildcard_deployment_strips_routing_prefix(self):
        """openrouter/* proxy deployments pass custom_llm_provider; strip LiteLLM prefix."""
        result_model, provider, _, _ = litellm.get_llm_provider(
            model="openrouter/anthropic/claude-3.5-sonnet",
            custom_llm_provider="openrouter",
        )
        assert provider == "openrouter"
        assert result_model == "anthropic/claude-3.5-sonnet"


class TestOpenRouterLiveModelDiscovery:
    """get_valid_models(check_provider_endpoint=True) must hit the live catalog."""

    def test_get_valid_models_uses_openrouter_catalog(self):
        from unittest.mock import MagicMock, patch

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"id": "anthropic/claude-sonnet-4"},
                {"id": "openai/gpt-5"},
            ]
        }

        with patch("litellm.module_level_client.get", return_value=mock_response) as mock_get:
            models = litellm.get_valid_models(
                check_provider_endpoint=True,
                custom_llm_provider="openrouter",
                api_key="sk-or-test",
            )

        assert models == ["anthropic/claude-sonnet-4", "openai/gpt-5"]
        assert mock_get.call_args.kwargs["url"] == "https://openrouter.ai/api/v1/models"
        assert mock_get.call_args.kwargs["headers"] == {"Authorization": "Bearer sk-or-test"}

    def test_get_models_defaults_api_base_and_omits_auth_without_key(self):
        from unittest.mock import MagicMock, patch

        from litellm.llms.openrouter.chat.transformation import OpenrouterConfig

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": [{"id": "m1"}]}

        with (
            patch("litellm.module_level_client.get", return_value=mock_response) as mock_get,
            patch.dict("os.environ", {}, clear=False),
        ):
            import os

            os.environ.pop("OPENROUTER_API_KEY", None)
            os.environ.pop("OPENAI_API_KEY", None)
            models = OpenrouterConfig().get_models()

        assert models == ["m1"]
        assert mock_get.call_args.kwargs["url"] == "https://openrouter.ai/api/v1/models"
        assert mock_get.call_args.kwargs["headers"] == {}

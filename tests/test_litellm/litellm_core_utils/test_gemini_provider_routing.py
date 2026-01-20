"""Tests for gemini model routing in get_llm_provider"""

from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


class TestGeminiProviderRouting:
    """Test that gemini models route to the correct provider"""

    def test_gemini_2_0_flash_routes_to_gemini_provider(self):
        """Test that gemini-2.0-flash routes to gemini provider, not vertex_ai"""
        model, provider, _, _ = get_llm_provider("gemini-2.0-flash")
        assert provider == "gemini", (
            f"Expected 'gemini' provider for gemini-2.0-flash, got '{provider}'. "
            f"Gemini models should route to gemini provider by default, not vertex_ai."
        )
        assert model == "gemini-2.0-flash"

    def test_gemini_2_0_flash_lite_routes_to_gemini_provider(self):
        """Test that gemini-2.0-flash-lite routes to gemini provider, not vertex_ai"""
        model, provider, _, _ = get_llm_provider("gemini-2.0-flash-lite")
        assert provider == "gemini", (
            f"Expected 'gemini' provider for gemini-2.0-flash-lite, got '{provider}'. "
            f"Gemini models should route to gemini provider by default, not vertex_ai."
        )
        assert model == "gemini-2.0-flash-lite"

    def test_gemini_models_with_vertex_ai_prefix_route_to_vertex_ai(self):
        """Test that vertex_ai/gemini-* models route to vertex_ai provider when prefix is used"""
        model, provider, _, _ = get_llm_provider("vertex_ai/gemini-2.0-flash")
        assert (
            provider == "vertex_ai"
        ), f"Expected 'vertex_ai' provider when using vertex_ai/ prefix, got '{provider}'"
        assert model == "gemini-2.0-flash"

    def test_gemini_models_in_vertex_lists_route_to_gemini(self):
        """
        Test that gemini models that are in vertex_language_models or vertex_vision_models
        still route to gemini provider by default (not vertex_ai).
        This ensures intuitive behavior - gemini models go to gemini provider.
        """
        # Test models that might be in vertex lists but should route to gemini
        gemini_models_to_test = [
            "gemini-2.0-flash",
            "gemini-2.0-flash-lite",
        ]

        for model_name in gemini_models_to_test:
            model, provider, _, _ = get_llm_provider(model_name)
            assert provider == "gemini", (
                f"Model '{model_name}' routed to '{provider}' instead of 'gemini'. "
                f"Gemini models should default to gemini provider, not vertex_ai. "
                f"Use 'vertex_ai/{model_name}' prefix if vertex_ai is needed."
            )

    def test_non_gemini_vertex_models_unchanged(self):
        """Test that non-gemini vertex models still route to vertex_ai"""
        model, provider, _, _ = get_llm_provider("chat-bison")
        assert (
            provider == "vertex_ai"
        ), f"Expected 'vertex_ai' provider for chat-bison, got '{provider}'"
        assert model == "chat-bison"

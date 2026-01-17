"""
Test HeliconeLogger Gemini/Vertex AI support.
Fixes: https://github.com/BerriAI/litellm/issues/19093
"""

import pytest


def test_helicone_gemini_model_in_list():
    """
    Test that Gemini models are in the helicone_model_list.
    """
    from litellm.integrations.helicone import HeliconeLogger

    logger = HeliconeLogger()

    # Test that "gemini" is in the model list
    assert "gemini" in logger.helicone_model_list, "gemini should be in helicone_model_list"


def test_helicone_gemini_models_recognized():
    """
    Test that Gemini models are recognized and not replaced with gpt-3.5-turbo.
    """
    from litellm.integrations.helicone import HeliconeLogger

    logger = HeliconeLogger()

    test_models = ["gemini-1.5-pro", "gemini-2.0-flash", "vertex_ai/gemini-1.5-flash"]
    for model in test_models:
        is_recognized = any(
            accepted_model in model
            for accepted_model in logger.helicone_model_list
        )
        assert is_recognized, f"{model} should be recognized by helicone_model_list"

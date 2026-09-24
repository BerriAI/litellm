"""
Test HeliconeLogger Gemini/Vertex AI support.
Fixes: https://github.com/BerriAI/litellm/issues/19093
"""



def test_helicone_gemini_model_in_list():
    """
    Test that Gemini models are in the helicone_model_list.
    """
    from litellm.integrations.helicone import HeliconeLogger

    logger = HeliconeLogger()

    # Test that "gemini" is in the model list
    assert (
        "gemini" in logger.helicone_model_list
    ), "gemini should be in helicone_model_list"


def test_helicone_gemini_models_recognized():
    """
    Test that Gemini models are recognized and not replaced with gpt-3.5-turbo.
    """
    from litellm.integrations.helicone import HeliconeLogger

    logger = HeliconeLogger()

    test_models = ["gemini-1.5-pro", "gemini-2.0-flash", "vertex_ai/gemini-1.5-flash"]
    for model in test_models:
        is_recognized = any(
            accepted_model in model for accepted_model in logger.helicone_model_list
        )
        assert is_recognized, f"{model} should be recognized by helicone_model_list"


def test_helicone_vertex_gemini_gets_vertex_provider_url():
    """
    Test that vertex_ai/gemini-* models route to aiplatform.googleapis.com,
    not generativelanguage.googleapis.com.

    This verifies the branch ordering fix: is_vertex_ai must be checked
    before "gemini" in model, otherwise vertex gemini models get the wrong
    provider_url.
    """
    from unittest.mock import MagicMock, patch

    from litellm.integrations.helicone import HeliconeLogger

    logger = HeliconeLogger()

    captured = {}

    def mock_post(url, **kwargs):
        captured["url"] = url
        captured["data"] = kwargs.get("json", {})
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        return mock_resp

    test_cases = [
        # (model, custom_llm_provider, expected_provider_url)
        (
            "vertex_ai/gemini-1.5-pro",
            "",
            "https://aiplatform.googleapis.com/v1",
        ),
        (
            "gemini-2.0-flash",
            "vertex_ai",
            "https://aiplatform.googleapis.com/v1",
        ),
        (
            "gemini-1.5-flash",
            "",
            "https://generativelanguage.googleapis.com/v1beta",
        ),
    ]

    for model, custom_llm_provider, expected_url in test_cases:
        captured.clear()
        mock_client = MagicMock()
        mock_client.post = mock_post
        with patch("litellm.module_level_client", mock_client):
            logger.log_success(
                model=model,
                messages=[{"role": "user", "content": "test"}],
                response_obj={"choices": [{"message": {"content": "hi"}}]},
                start_time=MagicMock(),
                end_time=MagicMock(),
                print_verbose=lambda *args, **kwargs: None,
                kwargs={
                    "litellm_params": {
                        "custom_llm_provider": custom_llm_provider,
                        "metadata": {},
                    },
                },
            )

        assert "data" in captured, f"No request captured for {model}"
        actual_url = captured["data"]["providerRequest"]["url"]
        assert actual_url == expected_url, (
            f"Model {model} (provider={custom_llm_provider!r}): "
            f"expected provider_url={expected_url}, got {actual_url}"
        )

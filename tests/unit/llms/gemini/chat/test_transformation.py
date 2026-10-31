from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig


def test_top_logprobs_maps_to_generation_config_logprobs():
    config = GoogleAIStudioGeminiConfig()

    assert "top_logprobs" in config.get_supported_openai_params("gemini-3.5-flash")

    optional_params = config.map_openai_params(
        non_default_params={"logprobs": True, "top_logprobs": 3},
        optional_params={},
        model="gemini-3.5-flash",
        drop_params=False,
    )

    assert optional_params["responseLogprobs"] is True
    assert optional_params["logprobs"] == 3

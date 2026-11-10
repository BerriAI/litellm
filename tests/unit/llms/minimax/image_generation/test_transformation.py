from litellm.llms.minimax.image_generation.transformation import (
    MinimaxImageGenerationConfig,
)


def test_minimax_image_generation_url_and_body():
    config = MinimaxImageGenerationConfig()
    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="MiniMax-H3-Image",
            optional_params={},
            litellm_params={},
        )
        == "https://api.minimax.io/v1/images/generations"
    )
    assert "aspect_ratio" in config.get_supported_openai_params("MiniMax-H3-Image")
    assert "area" in config.get_supported_openai_params("MiniMax-H3-Image")
    assert config.transform_image_generation_request(
        model="MiniMax-H3-Image",
        prompt="a red cube",
        optional_params={"aspect_ratio": "1:1", "area": "1K"},
        litellm_params={},
        headers={},
    ) == {
        "model": "MiniMax-H3-Image",
        "prompt": "a red cube",
        "aspect_ratio": "1:1",
        "area": "1K",
    }

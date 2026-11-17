import pytest

from litellm.llms.topaz.image_variations.transformation import TopazImageVariationConfig


@pytest.mark.parametrize("api_base", ["https://gateway.example.com/topaz", "https://gateway.example.com/topaz/"])
def test_get_complete_url_ignores_a_trailing_slash_on_api_base(api_base: str):
    assert (
        TopazImageVariationConfig().get_complete_url(
            api_base=api_base, api_key=None, model="topaz-enhance", optional_params={}, litellm_params={}
        )
        == "https://gateway.example.com/topaz/image/v1/enhance"
    )

import pytest
from litellm.llms.gemini.image_generation.transformation import GoogleImageGenConfig

def test_gemini_flash_image_generation_extra_body():
    """
    Test that extra_body is properly extracted and deep merged into the
    generationConfig for Gemini 3.1 Flash Image Preview requests.
    """
    config = GoogleImageGenConfig()
    model = "gemini-3.1-flash-image-preview"
    prompt = "A realistic, high-quality close-up..."
    
    # User's optional parameters, containing the extra_body we want to inject
    optional_params = {
        "extra_body": {
            "generationConfig": {
                "imageConfig": {
                    "imageSize": "2K"
                }
            }
        }
    }
    litellm_params = {}
    headers = {}
    
    request_body = config.transform_image_generation_request(
        model=model,
        prompt=prompt,
        optional_params=optional_params,
        litellm_params=litellm_params,
        headers=headers
    )
    
    assert "generationConfig" in request_body
    gen_config = request_body["generationConfig"]
    
    # Should contain original hardcoded item
    assert "response_modalities" in gen_config
    assert gen_config["response_modalities"] == ["IMAGE", "TEXT"]
    
    # Should contain the extra_body injected item
    assert "imageConfig" in gen_config
    assert gen_config["imageConfig"] == {"imageSize": "2K"}

from litellm.llms.byteplus.image_generation.transformation import BytePlusImageGenerationConfig


class TestBytePlusImageGenerationConfig:
    def test_get_supported_openai_params(self):
        config = BytePlusImageGenerationConfig()
        params = config.get_supported_openai_params("byteplus/dola-seedream-5-0-pro-260628")
        assert "n" in params
        assert "size" in params
        assert "response_format" in params

    def test_get_complete_url(self):
        config = BytePlusImageGenerationConfig()
        url = config.get_complete_url(
            api_base="https://ark.ap-southeast.bytepluses.com/api/v3",
            api_key="key",
            model="byteplus/dola-seedream-5-0-pro-260628",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://ark.ap-southeast.bytepluses.com/api/v3/images/generations"

    def test_transform_image_generation_request(self):
        config = BytePlusImageGenerationConfig()
        req = config.transform_image_generation_request(
            model="dola-seedream-5-0-pro-260628",
            prompt="a cat",
            optional_params={"size": "2K", "output_format": "png"},
            litellm_params={},
            headers={},
        )
        assert req["model"] == "dola-seedream-5-0-pro-260628"
        assert req["prompt"] == "a cat"
        assert req["size"] == "2K"
        assert req["output_format"] == "png"

    def test_transform_image_generation_request_extra_body_reserved_fields(self):
        config = BytePlusImageGenerationConfig()
        req = config.transform_image_generation_request(
            model="dola-seedream-5-0-pro-260628",
            prompt="a cat",
            optional_params={
                "extra_body": {
                    "model": "malicious-model",
                    "prompt": "malicious prompt",
                    "custom_field": "custom_val",
                }
            },
            litellm_params={},
            headers={},
        )
        assert req["model"] == "dola-seedream-5-0-pro-260628"
        assert req["prompt"] == "a cat"
        assert req["custom_field"] == "custom_val"

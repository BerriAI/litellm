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

    def test_get_complete_url_variations(self):
        config = BytePlusImageGenerationConfig()
        url1 = config.get_complete_url(
            api_base="https://custom.com/images/generations",
            api_key="key",
            model="byteplus/dola-seedream-5-0-pro-260628",
            optional_params={},
            litellm_params={},
        )
        assert url1 == "https://custom.com/images/generations"

        url2 = config.get_complete_url(
            api_base="https://custom.com",
            api_key="key",
            model="byteplus/dola-seedream-5-0-pro-260628",
            optional_params={},
            litellm_params={},
        )
        assert url2 == "https://custom.com/api/v3/images/generations"

    def test_map_openai_params(self):
        config = BytePlusImageGenerationConfig()
        res = config.map_openai_params(
            non_default_params={"size": "1024x1024", "unsupported": 123},
            optional_params={},
            model="byteplus/dola-seedream-5-0-pro-260628",
            drop_params=False,
        )
        assert res.get("size") == "1024x1024"
        assert "unsupported" not in res

    def test_validate_environment(self, monkeypatch):
        import pytest

        monkeypatch.delenv("BYTEPLUS_API_KEY", raising=False)
        monkeypatch.delenv("ARK_API_KEY", raising=False)
        config = BytePlusImageGenerationConfig()
        with pytest.raises(ValueError, match="BytePlus API key is required"):
            config.validate_environment(
                headers={},
                model="byteplus/dola-seedream-5-0-pro-260628",
                messages=[],
                optional_params={},
                litellm_params={},
            )

        headers = config.validate_environment(
            headers={},
            model="byteplus/dola-seedream-5-0-pro-260628",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )
        assert headers.get("Authorization") == "Bearer test-key"

    def test_get_error_class(self):
        config = BytePlusImageGenerationConfig()
        err = config.get_error_class("custom error", 400, headers={"x-request-id": "123"})
        assert err.status_code == 400
        assert "custom error" in err.message

    def test_transform_image_generation_response(self):
        import httpx
        from unittest.mock import MagicMock
        from litellm.types.utils import ImageResponse

        config = BytePlusImageGenerationConfig()
        raw_json = {
            "created": 123456789,
            "data": [{"url": "https://example.com/image.png"}],
        }
        raw_resp = httpx.Response(status_code=200, json=raw_json)
        mock_logging = MagicMock()
        res = config.transform_image_generation_response(
            model="byteplus/dola-seedream-5-0-pro-260628",
            raw_response=raw_resp,
            model_response=ImageResponse(),
            logging_obj=mock_logging,
            request_data={"prompt": "a dog"},
            optional_params={},
            litellm_params={},
        )
        assert len(res.data) == 1
        assert res.data[0].url == "https://example.com/image.png"
        assert mock_logging.post_call.called

    def test_get_byteplus_image_generation_config_and_manager(self):
        from litellm.llms.byteplus.image_generation import get_byteplus_image_generation_config
        from litellm.types.utils import LlmProviders
        from litellm.utils import ProviderConfigManager

        cfg = get_byteplus_image_generation_config("byteplus/dola-seedream-5-0-pro-260628")
        assert isinstance(cfg, BytePlusImageGenerationConfig)

        manager_cfg = ProviderConfigManager.get_provider_image_generation_config(
            model="byteplus/dola-seedream-5-0-pro-260628",
            provider=LlmProviders.BYTEPLUS,
        )
        assert isinstance(manager_cfg, BytePlusImageGenerationConfig)

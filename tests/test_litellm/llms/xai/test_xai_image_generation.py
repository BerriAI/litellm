from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.xai.image_generation.transformation import XAIImageGenerationConfig
from litellm.types.utils import ImageResponse
from litellm.utils import ProviderConfigManager
from litellm.types.utils import LlmProviders


def test_provider_config_manager_returns_xai_image_config():
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="grok-imagine-image",
        provider=LlmProviders.XAI,
    )
    assert isinstance(config, XAIImageGenerationConfig)


def test_map_size_to_aspect_ratio():
    config = XAIImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={"size": "1024x1792"},
        optional_params={},
        model="grok-imagine-image",
        drop_params=True,
    )
    assert mapped["aspect_ratio"] == "9:16"
    assert "size" not in mapped


def test_get_complete_url_default():
    config = XAIImageGenerationConfig()
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="grok-imagine-image",
        optional_params={},
        litellm_params={},
    )
    assert url.endswith("/v1/images/generations")


def test_validate_environment_requires_credentials():
    config = XAIImageGenerationConfig()
    with pytest.raises(Exception, match="Missing xAI credentials"):
        config.validate_environment(
            headers={},
            model="grok-imagine-image",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )


def test_validate_environment_oauth_injects_bearer():
    config = XAIImageGenerationConfig()
    with patch(
        "litellm.llms.xai.oauth.XAIOAuthAuthenticator.get_access_token",
        return_value="oauth-token",
    ):
        headers = config.validate_environment(
            headers={},
            model="grok-imagine-image",
            messages=[],
            optional_params={},
            litellm_params={"use_xai_oauth": True},
            api_key=None,
        )
    assert headers["Authorization"] == "Bearer oauth-token"


def test_map_string_n_to_int():
    config = XAIImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={"n": "1"},
        optional_params={},
        model="grok-imagine-image",
        drop_params=True,
    )
    assert mapped["n"] == 1
    assert isinstance(mapped["n"], int)


def test_transform_string_n_to_int():
    config = XAIImageGenerationConfig()
    request = config.transform_image_generation_request(
        model="xai/grok-imagine-image",
        prompt="a red apple",
        optional_params={"n": "1"},
        litellm_params={},
        headers={},
    )
    assert request["n"] == 1
    assert isinstance(request["n"], int)


def test_transform_request_and_response():
    config = XAIImageGenerationConfig()
    request = config.transform_image_generation_request(
        model="xai/grok-imagine-image",
        prompt="a red apple",
        optional_params={"aspect_ratio": "1:1"},
        litellm_params={},
        headers={},
    )
    assert request == {
        "model": "grok-imagine-image",
        "prompt": "a red apple",
        "aspect_ratio": "1:1",
    }

    raw = httpx.Response(
        200,
        json={
            "data": [
                {
                    "url": "https://imgen.x.ai/example.jpeg",
                    "mime_type": "image/jpeg",
                }
            ]
        },
    )
    logging_obj = MagicMock()
    response = config.transform_image_generation_response(
        model="grok-imagine-image",
        raw_response=raw,
        model_response=ImageResponse(),
        logging_obj=logging_obj,
        request_data=request,
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.data is not None
    assert response.data[0].url == "https://imgen.x.ai/example.jpeg"

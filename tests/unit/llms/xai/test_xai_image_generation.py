from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.llms.xai.image_generation.transformation import XAIImageGenerationConfig
from litellm.types.utils import ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_xai_image_config():
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="grok-imagine-image",
        provider=LlmProviders.XAI,
    )
    assert isinstance(config, XAIImageGenerationConfig)


def test_map_size_to_aspect_ratio():
    mapped = XAIImageGenerationConfig().map_openai_params(
        non_default_params={"size": "1024x1792"},
        optional_params={},
        model="grok-imagine-image",
        drop_params=True,
    )
    assert mapped["aspect_ratio"] == "9:16"
    assert "size" not in mapped


def test_get_complete_url_default():
    url = XAIImageGenerationConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model="grok-imagine-image",
        optional_params={},
        litellm_params={},
    )
    assert url.endswith("/v1/images/generations")


def test_validate_environment_requires_credentials():
    with pytest.raises(Exception, match="Missing xAI credentials"):
        XAIImageGenerationConfig().validate_environment(
            headers={},
            model="grok-imagine-image",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=None,
        )


def test_validate_environment_oauth_injects_bearer():
    with patch(
        "litellm.llms.xai.oauth.XAIOAuthAuthenticator.get_access_token",
        return_value="oauth-token",
    ):
        headers = XAIImageGenerationConfig().validate_environment(
            headers={},
            model="grok-imagine-image",
            messages=[],
            optional_params={},
            litellm_params={"use_xai_oauth": True},
            api_key=None,
        )
    assert headers["Authorization"] == "Bearer oauth-token"


def test_map_string_n_to_int():
    mapped = XAIImageGenerationConfig().map_openai_params(
        non_default_params={"n": "1"},
        optional_params={},
        model="grok-imagine-image",
        drop_params=True,
    )
    assert mapped["n"] == 1
    assert isinstance(mapped["n"], int)


def test_transform_string_n_to_int():
    request = XAIImageGenerationConfig().transform_image_generation_request(
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
    response = config.transform_image_generation_response(
        model="grok-imagine-image",
        raw_response=raw,
        model_response=ImageResponse(),
        logging_obj=MagicMock(),
        request_data=request,
        optional_params={},
        litellm_params={},
        encoding=None,
    )
    assert response.data is not None
    assert response.data[0].url == "https://imgen.x.ai/example.jpeg"


def test_validate_environment_api_key_sets_bearer_and_json_content_type():
    headers = XAIImageGenerationConfig().validate_environment(
        headers={},
        model="grok-imagine-image",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="fake-key",
    )

    assert headers == {"Authorization": "Bearer fake-key", "Content-Type": "application/json"}


def test_map_rejects_unknown_param_unless_dropped():
    config = XAIImageGenerationConfig()
    with pytest.raises(ValueError, match="Parameter quality is not supported"):
        config.map_openai_params(
            non_default_params={"quality": "hd", "n": 2},
            optional_params={},
            model="grok-imagine-image",
            drop_params=False,
        )

    assert config.map_openai_params(
        non_default_params={"quality": "hd", "n": 2},
        optional_params={},
        model="grok-imagine-image",
        drop_params=True,
    ) == {"n": 2}


def test_get_complete_url_appends_v1_to_a_bare_custom_base():
    url = XAIImageGenerationConfig().get_complete_url(
        api_base="https://xai-gateway.example.com/",
        api_key="fake-key",
        model="grok-imagine-image",
        optional_params={},
        litellm_params={},
    )

    assert url == "https://xai-gateway.example.com/v1/images/generations"


@pytest.mark.parametrize(
    "raw",
    [
        httpx.Response(502, text="<html>bad gateway</html>"),
        httpx.Response(200, json={"data": ["junk"]}),
    ],
)
def test_transform_response_raises_provider_error_without_images(raw):
    with pytest.raises(Exception) as exc_info:
        XAIImageGenerationConfig().transform_image_generation_response(
            model="grok-imagine-image",
            raw_response=raw,
            model_response=ImageResponse(),
            logging_obj=MagicMock(),
            request_data={"prompt": "a red apple"},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    assert getattr(exc_info.value, "status_code", None) == raw.status_code

from io import BytesIO
from unittest.mock import MagicMock, patch

import httpx
import pytest

from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def test_provider_config_manager_returns_xai_image_edit_config():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    config = ProviderConfigManager.get_provider_image_edit_config(
        model="grok-imagine-image",
        provider=LlmProviders.XAI,
    )
    assert isinstance(config, XAIImageEditConfig)


def test_get_complete_url_default():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    url = XAIImageEditConfig().get_complete_url(
        model="grok-imagine-image",
        api_base=None,
        litellm_params={},
    )
    assert url.endswith("/v1/images/edits")


def test_uses_json_not_multipart():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    assert XAIImageEditConfig().use_multipart_form_data() is False


def test_map_size_to_aspect_ratio():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    mapped = XAIImageEditConfig().map_openai_params(
        image_edit_optional_params={"size": "1024x1792", "n": 1},
        model="grok-imagine-image",
        drop_params=True,
    )
    assert mapped["aspect_ratio"] == "9:16"
    assert mapped["n"] == 1
    assert "size" not in mapped


def test_validate_environment_requires_credentials():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    with pytest.raises(Exception, match="Missing xAI credentials"):
        XAIImageEditConfig().validate_environment(
            headers={},
            model="grok-imagine-image",
            api_key=None,
            litellm_params={},
        )


def test_validate_environment_oauth_injects_bearer():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    with patch(
        "litellm.llms.xai.oauth.XAIOAuthAuthenticator.get_access_token",
        return_value="oauth-token",
    ):
        headers = XAIImageEditConfig().validate_environment(
            headers={},
            model="grok-imagine-image",
            api_key=None,
            litellm_params={"use_xai_oauth": True},
        )
    assert headers["Authorization"] == "Bearer oauth-token"


def test_map_string_n_to_int():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    mapped = XAIImageEditConfig().map_openai_params(
        image_edit_optional_params={"n": "1"},
        model="grok-imagine-image",
        drop_params=True,
    )
    assert mapped["n"] == 1
    assert isinstance(mapped["n"], int)


def test_transform_string_n_to_int():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    data, _ = XAIImageEditConfig().transform_image_edit_request(
        model="xai/grok-imagine-image",
        prompt="make the cube red",
        image="https://imgen.x.ai/source.jpeg",
        image_edit_optional_request_params={"n": "1"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert data["n"] == 1
    assert isinstance(data["n"], int)


def test_transform_bytes_to_data_uri_and_response():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    config = XAIImageEditConfig()
    data, files = config.transform_image_edit_request(
        model="xai/grok-imagine-image",
        prompt="make the cube red",
        image=b"\xff\xd8\xfffakejpeg",
        image_edit_optional_request_params={"aspect_ratio": "1:1"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files == []
    assert data["model"] == "grok-imagine-image"
    assert data["prompt"] == "make the cube red"
    assert data["aspect_ratio"] == "1:1"
    assert data["image"]["url"].startswith("data:image/jpeg;base64,")

    raw = httpx.Response(
        200,
        json={"data": [{"url": "https://imgen.x.ai/edited.jpeg", "mime_type": "image/jpeg"}]},
    )
    response = config.transform_image_edit_response(
        model="grok-imagine-image",
        raw_response=raw,
        logging_obj=MagicMock(),
    )
    assert response.data is not None
    assert response.data[0].url == "https://imgen.x.ai/edited.jpeg"


def test_transform_http_url_passthrough():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    data, files = XAIImageEditConfig().transform_image_edit_request(
        model="grok-imagine-image",
        prompt="make it night",
        image="https://imgen.x.ai/source.jpeg",
        image_edit_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert files == []
    assert data["image"] == {"url": "https://imgen.x.ai/source.jpeg"}


def test_transform_multiple_images_uses_images_array():
    from litellm.llms.xai.image_edit.transformation import XAIImageEditConfig

    data, _ = XAIImageEditConfig().transform_image_edit_request(
        model="grok-imagine-image",
        prompt="combine styles",
        image=["https://imgen.x.ai/a.jpeg", "https://imgen.x.ai/b.jpeg"],
        image_edit_optional_request_params={},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )
    assert "image" not in data
    assert data["images"] == [
        {"url": "https://imgen.x.ai/a.jpeg"},
        {"url": "https://imgen.x.ai/b.jpeg"},
    ]

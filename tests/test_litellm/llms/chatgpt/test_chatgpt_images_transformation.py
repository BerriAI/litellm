from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import litellm
from litellm.llms.chatgpt.image_edit.transformation import ChatGPTImageEditConfig
from litellm.llms.chatgpt.image_generation.transformation import (
    ChatGPTImageGenerationConfig,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.mark.parametrize(
    ("factory", "expected_type"),
    [
        (
            ProviderConfigManager.get_provider_image_generation_config,
            ChatGPTImageGenerationConfig,
        ),
        (
            ProviderConfigManager.get_provider_image_edit_config,
            ChatGPTImageEditConfig,
        ),
    ],
)
def test_chatgpt_image_provider_configs_are_registered(factory, expected_type):
    config = factory(model="gpt-image-2", provider=LlmProviders.CHATGPT)
    assert isinstance(config, expected_type)


@patch("litellm.llms.chatgpt.image_generation.transformation.Authenticator")
def test_chatgpt_image_generation_request(mock_authenticator_class):
    authenticator = MagicMock()
    authenticator.get_access_token.return_value = "oauth-token"
    authenticator.get_account_id.return_value = "account-id"
    authenticator.get_api_base.return_value = "https://chatgpt.example/backend-api/codex"
    mock_authenticator_class.return_value = authenticator
    config = ChatGPTImageGenerationConfig()

    headers = config.validate_environment(
        headers={},
        model="gpt-image-2",
        messages=[],
        optional_params={},
        litellm_params={"litellm_session_id": "turn-id"},
    )
    request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a fox",
        optional_params={"quality": "high", "size": "1024x1024"},
        litellm_params={},
        headers=headers,
    )

    assert headers["Authorization"] == "Bearer oauth-token"
    assert headers["ChatGPT-Account-Id"] == "account-id"
    assert headers["x-codex-image-turn-id"] == "turn-id"
    assert config.get_complete_url(None, None, "gpt-image-2", {}, {}) == (
        "https://chatgpt.example/backend-api/codex/images/generations"
    )
    assert request == {
        "prompt": "draw a fox",
        "model": "gpt-image-2",
        "quality": "high",
        "size": "1024x1024",
    }


@patch("litellm.llms.chatgpt.image_edit.transformation.Authenticator")
def test_chatgpt_image_edit_uses_json_data_urls(mock_authenticator_class):
    authenticator = MagicMock()
    authenticator.get_api_base.return_value = "https://chatgpt.example/backend-api/codex/"
    mock_authenticator_class.return_value = authenticator
    config = ChatGPTImageEditConfig()
    image = BytesIO(b"\x89PNG\r\n\x1a\nimage-data")

    request, files = config.transform_image_edit_request(
        model="gpt-image-2",
        prompt="add a hat",
        image=[image],
        image_edit_optional_request_params={"quality": "medium"},
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert config.use_multipart_form_data() is False
    assert files == {}
    assert request["images"] == [
        {"image_url": "data:image/png;base64,iVBORw0KGgppbWFnZS1kYXRh"}
    ]
    assert request["prompt"] == "add a hat"
    assert request["model"] == "gpt-image-2"
    assert request["quality"] == "medium"
    assert image.tell() == 0
    assert config.get_complete_url("gpt-image-2", None, {}) == (
        "https://chatgpt.example/backend-api/codex/images/edits"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("route_type", ["aimage_generation", "aimage_edit"])
async def test_unconfigured_gpt_image_model_auto_routes_to_chatgpt(route_type):
    from litellm.proxy.route_llm_request import route_request

    async def fake_image_call(**kwargs):
        return kwargs

    with patch.object(litellm, route_type, side_effect=fake_image_call) as image_call:
        pending = await route_request(
            data={"model": "gpt-image-2", "prompt": "draw a fox"},
            llm_router=None,
            user_model=None,
            route_type=route_type,
        )
        result = await pending

    assert result["model"] == "chatgpt/gpt-image-2"
    assert result["_litellm_client_requested_model"] == "gpt-image-2"
    image_call.assert_called_once()

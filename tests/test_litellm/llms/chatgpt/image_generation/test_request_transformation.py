from typing import Any, cast

import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.image_generation import ChatGPTImageGenerationConfig
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.fixture(autouse=True)
def _chatgpt_token_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))


def test_chatgpt_image_generation_transforms_request():
    config = ChatGPTImageGenerationConfig()

    request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a quiet harbor at sunrise",
        optional_params={"size": "1024x1024", "output_format": "png"},
        litellm_params={"chatgpt_responses_model": "gpt-5.5"},
        headers={},
    )

    assert request["model"] == "gpt-5.5"
    assert request["input"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": "draw a quiet harbor at sunrise",
                }
            ],
        }
    ]
    assert request["stream"] is True
    assert request["store"] is False
    assert request["tools"] == [
        {
            "type": "image_generation",
            "model": "gpt-image-2",
            "size": "1024x1024",
            "output_format": "png",
        }
    ]
    assert request["tool_choice"] == {"type": "image_generation"}


def test_chatgpt_image_generation_does_not_add_openai_defaults():
    config = ChatGPTImageGenerationConfig()

    request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a quiet harbor at sunrise",
        optional_params={},
        litellm_params={"chatgpt_responses_model": "gpt-5.5"},
        headers={},
    )

    assert request["tools"] == [{"type": "image_generation", "model": "gpt-image-2"}]


def test_chatgpt_image_generation_forwards_supported_generate_params():
    config = ChatGPTImageGenerationConfig()

    request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a quiet harbor at sunrise",
        optional_params={
            "output_format": "webp",
            "size": "1536x1024",
        },
        litellm_params={"chatgpt_responses_model": "gpt-5.5"},
        headers={},
    )

    assert request["tools"] == [
        {
            "type": "image_generation",
            "model": "gpt-image-2",
            "output_format": "webp",
            "size": "1536x1024",
        }
    ]


@pytest.mark.parametrize(
    "optional_params, error",
    [
        ({"output_format": "jpg"}, "output_format must be one of png, jpeg, or webp"),
    ],
)
def test_chatgpt_image_generation_validates_params(optional_params, error):
    config = ChatGPTImageGenerationConfig()

    with pytest.raises(ValueError, match=error):
        config.transform_image_generation_request(
            model="gpt-image-2",
            prompt="draw a quiet harbor at sunrise",
            optional_params=optional_params,
            litellm_params={"chatgpt_responses_model": "gpt-5.5"},
            headers={},
        )


def test_chatgpt_image_generation_config_registered():
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="gpt-image-2",
        provider=LlmProviders.CHATGPT,
    )

    assert isinstance(config, ChatGPTImageGenerationConfig)


def test_chatgpt_image_generation_only_supports_prompt_output_format_and_size():
    config = ChatGPTImageGenerationConfig()

    assert config.get_supported_openai_params("gpt-image-2") == [
        "output_format",
        "size",
    ]


def test_chatgpt_image_generation_rejects_unsupported_optional_params():
    config = ChatGPTImageGenerationConfig()

    with pytest.raises(
        ValueError, match="Parameters \\['quality'\\] are not supported"
    ):
        config.transform_image_generation_request(
            model="gpt-image-2",
            prompt="draw a cat",
            optional_params={"quality": "high"},
            litellm_params={},
            headers={},
        )


def test_chatgpt_image_generation_maps_supported_openai_params():
    config = ChatGPTImageGenerationConfig()

    optional_params = {"output_format": "png"}
    result = config.map_openai_params(
        non_default_params={
            "quality": "low",
            "size": "1024x1024",
            "unsupported": "drop-me",
        },
        optional_params=optional_params,
        model="gpt-image-2",
        drop_params=True,
    )

    assert result is optional_params
    assert result == {"output_format": "png", "size": "1024x1024"}


def test_chatgpt_image_generation_rejects_unsupported_openai_param():
    config = ChatGPTImageGenerationConfig()

    with pytest.raises(ValueError, match="Parameter unsupported is not supported"):
        config.map_openai_params(
            non_default_params={"unsupported": "keep-me"},
            optional_params={},
            model="gpt-image-2",
            drop_params=False,
        )


def test_chatgpt_image_generation_validates_environment():
    config = ChatGPTImageGenerationConfig()

    class FakeAuthenticator:
        def get_access_token(self):
            return "access-token"

        def get_account_id(self):
            return "account-id"

    config.authenticator = cast(Any, FakeAuthenticator())

    headers = config.validate_environment(
        headers={"content-type": "application/custom", "x-extra": "1"},
        model="gpt-image-2",
        messages=[],
        optional_params={},
        litellm_params={"session_id": "session-123"},
    )

    assert headers["Authorization"] == "Bearer access-token"
    assert headers["ChatGPT-Account-Id"] == "account-id"
    assert headers["session_id"] == "session-123"
    assert headers["content-type"] == "application/custom"
    assert headers["x-extra"] == "1"


def test_chatgpt_image_generation_validate_environment_auth_error():
    config = ChatGPTImageGenerationConfig()

    class FakeAuthenticator:
        def get_access_token(self):
            raise GetAccessTokenError(status_code=401, message="token expired")

    config.authenticator = cast(Any, FakeAuthenticator())

    with pytest.raises(AuthenticationError, match="token expired"):
        config.validate_environment(
            headers={},
            model="gpt-image-2",
            messages=[],
            optional_params={},
            litellm_params={},
        )


@pytest.mark.parametrize(
    "server_api_base, expected",
    [
        (
            "https://chatgpt.com/backend-api",
            "https://chatgpt.com/backend-api/codex/responses",
        ),
        (
            "https://chatgpt.com/backend-api/responses",
            "https://chatgpt.com/backend-api/codex/responses",
        ),
        (
            "https://example.test/custom/",
            "https://example.test/custom/responses",
        ),
    ],
)
def test_chatgpt_image_generation_get_complete_url_canonicalizes_server_api_base(
    server_api_base, expected
):
    config = ChatGPTImageGenerationConfig()

    class FakeAuthenticator:
        def get_api_base(self):
            return server_api_base

    config.authenticator = cast(Any, FakeAuthenticator())

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gpt-image-2",
            optional_params={},
            litellm_params={},
        )
        == expected
    )


def test_chatgpt_image_generation_get_complete_url_ignores_request_api_base():
    config = ChatGPTImageGenerationConfig()

    class FakeAuthenticator:
        def get_api_base(self):
            return "https://chatgpt.com/backend-api"

    config.authenticator = cast(Any, FakeAuthenticator())

    assert (
        config.get_complete_url(
            api_base="https://attacker.test/collect",
            api_key=None,
            model="gpt-image-2",
            optional_params={},
            litellm_params={},
        )
        == "https://chatgpt.com/backend-api/codex/responses"
    )


def test_chatgpt_image_generation_get_complete_url_uses_authenticator_api_base():
    config = ChatGPTImageGenerationConfig()

    class FakeAuthenticator:
        def get_api_base(self):
            return "https://example.test/backend-api"

    config.authenticator = cast(Any, FakeAuthenticator())

    assert (
        config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gpt-image-2",
            optional_params={},
            litellm_params={},
        )
        == "https://example.test/backend-api/codex/responses"
    )


def test_chatgpt_image_generation_uses_optional_responses_model():
    config = ChatGPTImageGenerationConfig()
    optional_params = {"chatgpt_responses_model": "gpt-override"}

    request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a cat",
        optional_params=optional_params,
        litellm_params={"chatgpt_responses_model": "gpt-litellm-param"},
        headers={},
    )

    assert request["model"] == "gpt-override"
    assert "chatgpt_responses_model" not in optional_params

    litellm_params_request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a cat",
        optional_params={},
        litellm_params={"chatgpt_responses_model": "gpt-litellm-param"},
        headers={},
    )
    assert litellm_params_request["model"] == "gpt-litellm-param"

    default_request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a cat",
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert default_request["model"] == "gpt-5.5"


@pytest.mark.parametrize(
    "model, optional_params, error",
    [
        ("dall-e-3", {}, "requires a GPT Image model"),
        ("gpt-image-1.5", {"size": "auto"}, None),
        ("gpt-image-2", {"size": "auto"}, None),
        ("gpt-image-2", {"size": "bad-size"}, None),
    ],
)
def test_chatgpt_image_generation_validates_additional_param_paths(
    model, optional_params, error
):
    config = ChatGPTImageGenerationConfig()

    if error is None:
        config.transform_image_generation_request(
            model=model,
            prompt="draw a cat",
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )
        return

    with pytest.raises(ValueError, match=error):
        config.transform_image_generation_request(
            model=model,
            prompt="draw a cat",
            optional_params=optional_params,
            litellm_params={},
            headers={},
        )


def test_chatgpt_image_generation_forwards_size_without_local_constraints():
    config = ChatGPTImageGenerationConfig()

    request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a cat",
        optional_params={"size": "bad-size"},
        litellm_params={},
        headers={},
    )

    assert request["tools"][0]["size"] == "bad-size"


def test_chatgpt_image_generation_map_openai_params_keeps_existing_value():
    config = ChatGPTImageGenerationConfig()

    optional_params = {"size": "1024x1024"}
    result = config.map_openai_params(
        non_default_params={"size": "1536x1024"},
        optional_params=optional_params,
        model="gpt-image-2",
        drop_params=False,
    )

    assert result == {"size": "1024x1024"}

from typing import Any, cast

import httpx
import pytest

from litellm.exceptions import AuthenticationError
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.image_generation.transformation import (
    ChatGPTImageGenerationConfig,
)
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.utils import LlmProviders
from litellm.types.utils import ImageResponse
from litellm.utils import ProviderConfigManager


class MockLogging:
    def post_call(self, *args, **kwargs):
        pass


def mock_logging() -> Any:
    return MockLogging()


def test_chatgpt_image_generation_transforms_request(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a quiet harbor at sunrise",
        optional_params={"size": "1024x1024", "quality": "high"},
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
            "quality": "high",
        }
    ]
    assert request["tool_choice"] == {"type": "image_generation"}


def test_chatgpt_image_generation_does_not_add_openai_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a quiet harbor at sunrise",
        optional_params={},
        litellm_params={"chatgpt_responses_model": "gpt-5.5"},
        headers={},
    )

    assert request["tools"] == [{"type": "image_generation", "model": "gpt-image-2"}]


def test_chatgpt_image_generation_forwards_official_generate_params(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a quiet harbor at sunrise",
        optional_params={
            "background": "opaque",
            "moderation": "low",
            "n": 1,
            "output_compression": 75,
            "output_format": "webp",
            "partial_images": 2,
            "quality": "medium",
            "response_format": "b64_json",
            "size": "1536x1024",
            "stream": True,
            "user": "user-123",
        },
        litellm_params={"chatgpt_responses_model": "gpt-5.5"},
        headers={},
    )

    assert request["tools"] == [
        {
            "type": "image_generation",
            "model": "gpt-image-2",
            "background": "opaque",
            "output_format": "webp",
            "quality": "medium",
            "size": "1536x1024",
        }
    ]
    assert request["partial_images"] == 2
    assert request["user"] == "user-123"
    assert "response_format" not in request["tools"][0]
    assert "stream" not in request["tools"][0]
    assert "user" not in request["tools"][0]


@pytest.mark.parametrize(
    "optional_params, error",
    [
        ({"n": 0}, "n must be between 1 and 10"),
        ({"n": 2}, "n > 1 is not supported for ChatGPT image generation"),
        ({"quality": "hd"}, "quality must be one of low, medium, high, or auto"),
        ({"output_format": "jpg"}, "output_format must be one of png, jpeg, or webp"),
        ({"output_compression": 101}, "output_compression must be between 0 and 100"),
        ({"background": "transparent"}, "transparent backgrounds are not supported"),
        (
            {"background": "transparent", "output_format": "jpeg"},
            "transparent backgrounds are not supported",
        ),
        ({"moderation": "strict"}, "moderation must be one of low or auto"),
        ({"partial_images": 4}, "partial_images must be between 0 and 3"),
        ({"response_format": "url"}, "response_format='url' is not supported"),
        ({"size": "1535x1024"}, "multiples of 16px"),
        ({"size": "4096x1024"}, "maximum edge length"),
        ({"size": "1024x256"}, "ratio must not exceed 3:1"),
        ({"size": "512x512"}, "total pixels must be between"),
    ],
)
def test_chatgpt_image_generation_validates_params(
    monkeypatch, tmp_path, optional_params, error
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    with pytest.raises(ValueError, match=error):
        config.transform_image_generation_request(
            model="gpt-image-2",
            prompt="draw a quiet harbor at sunrise",
            optional_params=optional_params,
            litellm_params={"chatgpt_responses_model": "gpt-5.5"},
            headers={},
        )


def test_chatgpt_image_generation_config_registered(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ProviderConfigManager.get_provider_image_generation_config(
        model="gpt-image-2",
        provider=LlmProviders.CHATGPT,
    )

    assert isinstance(config, ChatGPTImageGenerationConfig)


def test_chatgpt_image_generation_maps_supported_openai_params(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    optional_params = {"quality": "high"}
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
    assert result == {"quality": "high", "size": "1024x1024"}


def test_chatgpt_image_generation_rejects_unsupported_openai_param(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    with pytest.raises(ValueError, match="Parameter unsupported is not supported"):
        config.map_openai_params(
            non_default_params={"unsupported": "keep-me"},
            optional_params={},
            model="gpt-image-2",
            drop_params=False,
        )


def test_chatgpt_image_generation_validates_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
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


def test_chatgpt_image_generation_validate_environment_auth_error(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
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
    "api_base, expected",
    [
        (
            "https://chatgpt.com/backend-api",
            "https://chatgpt.com/backend-api/codex/responses",
        ),
        (
            "https://chatgpt.com/backend-api/responses",
            "https://chatgpt.com/backend-api/codex/responses",
        ),
        ("https://example.test/custom/", "https://example.test/custom/responses"),
    ],
)
def test_chatgpt_image_generation_get_complete_url_canonicalizes_api_base(
    monkeypatch, tmp_path, api_base, expected
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    assert (
        config.get_complete_url(
            api_base=api_base,
            api_key=None,
            model="gpt-image-2",
            optional_params={},
            litellm_params={},
        )
        == expected
    )


def test_chatgpt_image_generation_get_complete_url_uses_authenticator_api_base(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
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


def test_chatgpt_image_generation_extracts_b64_from_sse_completed_response(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        text=(
            'data: {"type":"response.completed","response":{"output":['
            '{"type":"image_generation_call","result":"b64-image-data"}]}}\n\n'
            "data: [DONE]\n\n"
        ),
    )

    response = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=mock_logging(),
        request_data={"input": "draw a cat"},
        optional_params={"size": "1024x1024", "quality": "high"},
        litellm_params={},
        encoding=None,
    )

    assert response.data is not None
    assert response.data[0].b64_json == "b64-image-data"
    assert response.size == "1024x1024"
    assert response.quality == "high"
    assert response.output_format is None
    assert response.usage is None
    assert response._hidden_params is not None
    assert response._hidden_params["model"] == "gpt-image-2"


def test_chatgpt_image_generation_uses_optional_responses_model_and_env(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("CHATGPT_IMAGE_RESPONSES_MODEL", "gpt-env")
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

    env_request = config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="draw a cat",
        optional_params={},
        litellm_params={},
        headers={},
    )
    assert env_request["model"] == "gpt-env"


@pytest.mark.parametrize(
    "model, optional_params, error",
    [
        ("dall-e-3", {}, "requires a GPT Image model"),
        ("gpt-image-1.5", {"size": "auto"}, None),
        ("gpt-image-2", {"size": "auto"}, None),
        ("gpt-image-2", {"size": "bad-size"}, "size must be auto or WIDTHxHEIGHT"),
        (
            "gpt-image-1.5",
            {"background": "transparent", "output_format": "jpeg"},
            "transparent background requires output_format png or webp",
        ),
        (
            "gpt-image-1.5",
            {"background": "not-real"},
            "background must be one of transparent, opaque, or auto",
        ),
    ],
)
def test_chatgpt_image_generation_validates_additional_param_paths(
    monkeypatch, tmp_path, model, optional_params, error
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
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


def test_chatgpt_image_generation_extracts_b64_from_deep_nested_payload(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    nested_payload = {"type": "image_generation_call", "result": "b64-image-data"}
    for _ in range(1200):
        nested_payload = {"nested": [nested_payload]}

    images, partial_images = config._extract_images_from_payload(nested_payload)

    assert images == ["b64-image-data"]
    assert partial_images == []


def test_chatgpt_image_generation_extracts_json_response(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        json={
            "output": [
                {
                    "type": "image_generation",
                    "image": ["b64-image-data", "b64-image-data", 123],
                }
            ],
            "tool_usage": {
                "image_gen": {
                    "input_tokens": 11,
                    "input_tokens_details": {"image_tokens": 1, "text_tokens": 10},
                    "output_tokens": 22,
                }
            },
        },
    )

    response = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=mock_logging(),
        request_data={},
        optional_params={"response_format": "b64_json"},
        litellm_params={},
        encoding=None,
    )

    assert response.data is not None
    assert [item.b64_json for item in response.data] == ["b64-image-data"]
    assert response.output_format == "b64_json"
    assert response.usage is not None
    assert response.usage.input_tokens == 11
    assert response.usage.input_tokens_details.image_tokens == 1
    assert response.usage.input_tokens_details.text_tokens == 10
    assert response.usage.output_tokens == 22
    assert response.usage.total_tokens == 33


def test_chatgpt_image_generation_raises_when_no_image_data(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(status_code=200, json={"output": []})

    with pytest.raises(OpenAIError, match="No image data found"):
        config.transform_image_generation_response(
            model="gpt-image-2",
            raw_response=raw_response,
            model_response=ImageResponse(),
            logging_obj=mock_logging(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )


def test_chatgpt_image_generation_raises_provider_error_event(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    with pytest.raises(OpenAIError, match="image blocked"):
        config._extract_images_from_payload(
            {
                "type": "response.failed",
                "response": {"error": {"message": "image blocked"}},
            }
        )


def test_chatgpt_image_generation_handles_invalid_json_payloads(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        text="{not-json",
    )

    assert config._extract_image_payloads(raw_response) == []
    assert config._get_parsed_payloads(raw_response) == [{}]


def test_chatgpt_image_generation_ignores_non_dict_json_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(status_code=200, json=[])

    assert config._extract_image_payloads(raw_response) == []
    assert config._get_parsed_payloads(raw_response) == []


def test_chatgpt_image_generation_extracts_from_cyclic_nested_payload(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    payload = {
        "type": "image_generation_call",
        "result": "b64-result",
        "b64_json": "b64-json",
        "image": ["b64-image", 123],
    }
    payload["self"] = payload

    assert config._extract_images_from_nested_value(payload) == [
        "b64-result",
        "b64-json",
        "b64-image",
    ]

    cyclic_list = []
    cyclic_list.append(cyclic_list)
    assert config._extract_images_from_nested_value(cyclic_list) == []


def test_chatgpt_image_generation_usage_helpers_ignore_invalid_payloads(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    assert config._get_image_generation_usage("not-a-dict") is None
    assert config._get_image_generation_usage({"tool_usage": []}) is None
    assert config._get_image_generation_usage({"tool_usage": {"image_gen": []}}) is None
    assert (
        config._get_image_generation_usage(
            {"tool_usage": {"image_gen": {"input_tokens": 1}}}
        )
        is None
    )
    assert config._is_zero_image_usage(
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    )


def test_chatgpt_image_generation_extracts_tool_usage_from_completed_response(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        text=(
            'data: {"type":"response.completed","response":{"output":['
            '{"type":"image_generation_call","result":"b64-image-data"}],'
            '"usage":{"input_tokens":1732,"output_tokens":121,"total_tokens":1853},'
            '"tool_usage":{"image_gen":{"input_tokens":108,'
            '"input_tokens_details":{"image_tokens":0,"text_tokens":108},'
            '"output_tokens":1756,'
            '"output_tokens_details":{"image_tokens":1756,"text_tokens":0},'
            '"total_tokens":1864}}}}\n\n'
            "data: [DONE]\n\n"
        ),
    )

    response = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=mock_logging(),
        request_data={"input": "draw a cat"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.usage is not None
    assert response.usage.input_tokens == 108
    assert response.usage.input_tokens_details.text_tokens == 108
    assert response.usage.input_tokens_details.image_tokens == 0
    assert response.usage.output_tokens == 1756
    assert response.usage.total_tokens == 1864


def test_chatgpt_image_generation_prefers_completed_tool_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    zero_usage = (
        '"tool_usage":{"image_gen":{"input_tokens":0,'
        '"input_tokens_details":{"image_tokens":0,"text_tokens":0},'
        '"output_tokens":0,'
        '"output_tokens_details":{"image_tokens":0,"text_tokens":0},'
        '"total_tokens":0}}'
    )
    completed_usage = (
        '"tool_usage":{"image_gen":{"input_tokens":105,'
        '"input_tokens_details":{"image_tokens":0,"text_tokens":105},'
        '"output_tokens":1372,'
        '"output_tokens_details":{"image_tokens":1372,"text_tokens":0},'
        '"total_tokens":1477}}'
    )
    raw_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        text=(
            'data: {"type":"response.created","response":{'
            f"{zero_usage}"
            "}}\n\n"
            'data: {"type":"response.in_progress","response":{'
            f"{zero_usage}"
            "}}\n\n"
            'data: {"type":"response.image_generation_call.partial_image",'
            '"partial_image_b64":"partial-image"}\n\n'
            'data: {"type":"response.completed","response":{"output":['
            '{"type":"image_generation_call","result":"b64-image-data"}],'
            f"{completed_usage}"
            ',"usage":{"input_tokens":2344,"output_tokens":118,"total_tokens":2462}}}\n\n'
            "data: [DONE]\n\n"
        ),
    )

    response = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=mock_logging(),
        request_data={"input": "draw a cat"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.usage is not None
    assert response.usage.input_tokens == 105
    assert response.usage.input_tokens_details.text_tokens == 105
    assert response.usage.input_tokens_details.image_tokens == 0
    assert response.usage.output_tokens == 1372
    assert response.usage.total_tokens == 1477


def test_chatgpt_image_generation_extracts_usage_with_partial_image_payload(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        text=(
            "event: response.created\n"
            'data: {"type":"response.created","response":{"tool_usage":{"image_gen":{'
            '"input_tokens":0,"input_tokens_details":{"image_tokens":0,"text_tokens":0},'
            '"output_tokens":0,"total_tokens":0}}}}\n\n'
            "event: response.image_generation_call.partial_image\n"
            'data: {"type":"response.image_generation_call.partial_image",'
            '"partial_image_b64":"partial-image-data","size":"1536x1024"}\n\n'
            "event: response.completed\n"
            'data: {"type":"response.completed","response":{"output":[],'
            '"tool_usage":{"image_gen":{"input_tokens":105,'
            '"input_tokens_details":{"image_tokens":0,"text_tokens":105},'
            '"output_tokens":1372,'
            '"output_tokens_details":{"image_tokens":1372,"text_tokens":0},'
            '"total_tokens":1477}},'
            '"usage":{"input_tokens":2344,"output_tokens":118,"total_tokens":2462}}}\n\n'
        ),
    )

    response = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=mock_logging(),
        request_data={"input": "draw a cat"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.data is not None
    assert response.data[0].b64_json == "partial-image-data"
    assert response.usage is not None
    assert response.usage.input_tokens == 105
    assert response.usage.input_tokens_details.text_tokens == 105
    assert response.usage.input_tokens_details.image_tokens == 0
    assert response.usage.output_tokens == 1372
    assert response.usage.total_tokens == 1477


def test_chatgpt_image_generation_extracts_top_level_tool_usage(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        text=(
            'data: {"type":"response.completed","response":{"output":['
            '{"type":"image_generation_call","result":"b64-image-data"}]},'
            '"tool_usage":{"image_gen":{"input_tokens":12,'
            '"input_tokens_details":{"image_tokens":2,"text_tokens":10},'
            '"output_tokens":34}}}\n\n'
            "data: [DONE]\n\n"
        ),
    )

    response = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=mock_logging(),
        request_data={"input": "draw a cat"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.usage is not None
    assert response.usage.input_tokens == 12
    assert response.usage.input_tokens_details.text_tokens == 10
    assert response.usage.input_tokens_details.image_tokens == 2
    assert response.usage.output_tokens == 34
    assert response.usage.total_tokens == 46


def test_chatgpt_image_generation_extracts_b64_from_streaming_completed_event(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        text=(
            'data: {"type":"image_generation.partial_image","b64_json":"partial-image"}\n\n'
            'data: {"type":"image_generation.completed","b64_json":"final-image"}\n\n'
            "data: [DONE]\n\n"
        ),
    )

    response = config.transform_image_generation_response(
        model="gpt-image-2",
        raw_response=raw_response,
        model_response=ImageResponse(),
        logging_obj=mock_logging(),
        request_data={"input": "draw a cat"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert response.data is not None
    assert [item.b64_json for item in response.data] == ["final-image"]


def test_chatgpt_image_generation_get_error_class(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))
    config = ChatGPTImageGenerationConfig()

    error = config.get_error_class(
        error_message="bad request",
        status_code=400,
        headers={"x-request-id": "req-123"},
    )

    assert isinstance(error, OpenAIError)
    assert error.status_code == 400
    assert error.message == "bad request"

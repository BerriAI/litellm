import httpx
import pytest

from litellm.llms.chatgpt.image_generation.transformation import (
    ChatGPTImageGenerationConfig,
)
from litellm.types.utils import LlmProviders
from litellm.types.utils import ImageResponse
from litellm.utils import ProviderConfigManager


class MockLogging:
    def post_call(self, *args, **kwargs):
        pass


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
        logging_obj=MockLogging(),
        request_data={"input": "draw a cat"},
        optional_params={"size": "1024x1024", "quality": "high"},
        litellm_params={},
        encoding=None,
    )

    assert response.data[0].b64_json == "b64-image-data"
    assert response.size == "1024x1024"
    assert response.quality == "high"
    assert response.usage is None
    assert response._hidden_params["model"] == "gpt-image-2"


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
        logging_obj=MockLogging(),
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
        logging_obj=MockLogging(),
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
        logging_obj=MockLogging(),
        request_data={"input": "draw a cat"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

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
        logging_obj=MockLogging(),
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
        logging_obj=MockLogging(),
        request_data={"input": "draw a cat"},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert [item.b64_json for item in response.data] == ["final-image"]

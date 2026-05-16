import httpx
import pytest

from litellm.llms.chatgpt.image_generation import ChatGPTImageGenerationConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.utils import ImageResponse
from tests.test_litellm.llms.chatgpt.chatgpt_image_test_utils import mock_logging


@pytest.fixture(autouse=True)
def _chatgpt_token_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))


def test_chatgpt_image_generation_extracts_b64_from_sse_completed_response():
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
        optional_params={"size": "1024x1024"},
        litellm_params={},
        encoding=None,
    )

    assert response.data is not None
    assert response.data[0].b64_json == "b64-image-data"
    assert response.size == "1024x1024"
    assert response.quality is None
    assert response.output_format is None
    assert response.usage is None
    assert response._hidden_params is not None
    assert response._hidden_params["model"] == "gpt-image-2"


def test_chatgpt_image_generation_extracts_b64_from_deep_nested_payload():
    config = ChatGPTImageGenerationConfig()
    nested_payload = {"type": "image_generation_call", "result": "b64-image-data"}
    for _ in range(1200):
        nested_payload = {"nested": [nested_payload]}

    images, partial_images = config._extract_images_from_payload(nested_payload)

    assert images == ["b64-image-data"]
    assert partial_images == []


def test_chatgpt_image_generation_extracts_json_response():
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
        optional_params={"output_format": "png"},
        litellm_params={},
        encoding=None,
    )

    assert response.data is not None
    assert [item.b64_json for item in response.data] == ["b64-image-data"]
    assert response.output_format == "png"
    assert response.usage is not None
    assert response.usage.input_tokens == 11
    assert response.usage.input_tokens_details.image_tokens == 1
    assert response.usage.input_tokens_details.text_tokens == 10
    assert response.usage.output_tokens == 22
    assert response.usage.total_tokens == 33


def test_chatgpt_image_generation_raises_when_no_image_data():
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


def test_chatgpt_image_generation_raises_provider_error_event():
    config = ChatGPTImageGenerationConfig()

    with pytest.raises(OpenAIError, match="image blocked"):
        config._extract_images_from_payload(
            {
                "type": "response.failed",
                "response": {"error": {"message": "image blocked"}},
            }
        )


def test_chatgpt_image_generation_handles_invalid_json_payloads():
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(
        status_code=200,
        headers={"content-type": "application/json"},
        text="{not-json",
    )

    assert config._extract_image_payloads(raw_response) == []
    assert config._get_parsed_payloads(raw_response) == [{}]


def test_chatgpt_image_generation_ignores_non_dict_json_payload():
    config = ChatGPTImageGenerationConfig()
    raw_response = httpx.Response(status_code=200, json=[])

    assert config._extract_image_payloads(raw_response) == []
    assert config._get_parsed_payloads(raw_response) == []


def test_chatgpt_image_generation_extracts_from_cyclic_nested_payload():
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


def test_chatgpt_image_generation_extracts_b64_from_streaming_completed_event():
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


def test_chatgpt_image_generation_get_error_class():
    config = ChatGPTImageGenerationConfig()

    error = config.get_error_class(
        error_message="bad request",
        status_code=400,
        headers={"x-request-id": "req-123"},
    )

    assert isinstance(error, OpenAIError)
    assert error.status_code == 400
    assert error.message == "bad request"

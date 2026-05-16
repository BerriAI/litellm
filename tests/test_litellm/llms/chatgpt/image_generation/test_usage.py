import httpx
import pytest

from litellm.llms.chatgpt.image_generation import ChatGPTImageGenerationConfig
from litellm.types.utils import ImageResponse
from tests.test_litellm.llms.chatgpt.chatgpt_image_test_utils import mock_logging


@pytest.fixture(autouse=True)
def _chatgpt_token_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path))


def test_chatgpt_image_generation_usage_helpers_ignore_invalid_payloads():
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


def test_chatgpt_image_generation_extracts_tool_usage_from_completed_response():
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


def test_chatgpt_image_generation_prefers_completed_tool_usage():
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


def test_chatgpt_image_generation_extracts_usage_with_partial_image_payload():
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


def test_chatgpt_image_generation_extracts_top_level_tool_usage():
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

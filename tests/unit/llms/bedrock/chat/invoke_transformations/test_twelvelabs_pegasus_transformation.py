from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.bedrock.chat.invoke_transformations.amazon_twelvelabs_pegasus_transformation import (
    AmazonTwelveLabsPegasusConfig,
)
from litellm.types.utils import ModelResponse


def _make_messages() -> list[dict]:
    return [
        {"role": "system", "content": "You are an assistant"},
        {"role": "user", "content": "Summarize the attached video."},
    ]


def test_supported_openai_params():
    config = AmazonTwelveLabsPegasusConfig()
    supported = config.get_supported_openai_params("twelvelabs.pegasus-1-2-v1:0")
    assert "max_tokens" in supported
    assert "temperature" in supported
    assert "response_format" in supported


def test_map_openai_params_translates_fields():
    config = AmazonTwelveLabsPegasusConfig()
    optional_params: dict = {}
    config.map_openai_params(
        non_default_params={
            "max_tokens": 20,
            "temperature": 0.6,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "video_schema", "schema": {"type": "object"}},
            },
        },
        optional_params=optional_params,
        model="twelvelabs.pegasus-1-2-v1:0",
        drop_params=False,
    )

    assert optional_params["maxOutputTokens"] == 20
    assert optional_params["temperature"] == 0.6
    assert "responseFormat" in optional_params
    # TwelveLabs format: responseFormat contains jsonSchema directly (not json_schema)
    assert "jsonSchema" in optional_params["responseFormat"]
    assert optional_params["responseFormat"]["jsonSchema"]["type"] == "object"


def test_transform_request_includes_base64_media():
    config = AmazonTwelveLabsPegasusConfig()
    optional_params = config.map_openai_params(
        non_default_params={"max_tokens": 10},
        optional_params={},
        model="twelvelabs.pegasus-1-2-v1:0",
        drop_params=False,
    )
    optional_params["video_base64"] = "data:video/mp4;base64,AAA"

    request = config.transform_request(
        model="twelvelabs.pegasus-1-2-v1:0",
        messages=_make_messages(),
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert request["inputPrompt"].startswith("system:")
    assert request["mediaSource"]["base64String"] == "AAA"
    assert request["maxOutputTokens"] == 10


def test_transform_request_includes_s3_media():
    config = AmazonTwelveLabsPegasusConfig()
    optional_params = {
        "video_s3_uri": "s3://test-bucket/video.mp4",
        "video_s3_bucket_owner": "123456789012",
    }

    request = config.transform_request(
        model="twelvelabs.pegasus-1-2-v1:0",
        messages=_make_messages(),
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    s3_location = request["mediaSource"]["s3Location"]
    assert s3_location["uri"] == "s3://test-bucket/video.mp4"
    assert s3_location["bucketOwner"] == "123456789012"


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [("length", "length"), (None, "stop")],
)
def test_transform_response_accepts_null_finish_reason(finish_reason, expected):
    result = AmazonTwelveLabsPegasusConfig().transform_response(
        model="twelvelabs.pegasus-1-2-v1:0",
        raw_response=httpx.Response(200, json={"message": "A cat jumps.", "finishReason": finish_reason}),
        model_response=ModelResponse(),
        logging_obj=Mock(),
        request_data={},
        messages=_make_messages(),
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert result.choices[0].message.content == "A cat jumps."
    assert result.choices[0].finish_reason == expected

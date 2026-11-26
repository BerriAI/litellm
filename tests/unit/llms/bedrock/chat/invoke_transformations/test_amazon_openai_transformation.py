from typing import Final

import pytest

from litellm.llms.bedrock.chat.invoke_transformations.amazon_openai_transformation import (
    AmazonBedrockOpenAIConfig,
)


def test_openai_imported_model_preserves_message_types_and_request_parameters() -> None:
    arn: Final = "arn:aws:bedrock:us-east-1:117159858402:imported-model/m4gc1mrfuddy"
    model: Final = f"bedrock/openai/{arn}"
    messages: Final = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Simple text message"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Spot the difference"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,YWJj"}},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,ZA=="}},
            ],
        },
    ]
    config: Final = AmazonBedrockOpenAIConfig()
    request: Final = config.transform_request(
        model=model,
        messages=messages,
        optional_params={"max_tokens": 300, "temperature": 0.5},
        litellm_params={},
        headers={},
    )
    url: Final = config.get_complete_url(
        api_base=None,
        api_key=None,
        model=model,
        optional_params={
            "aws_access_key_id": "AKIAUNITTESTKEY",
            "aws_secret_access_key": "unit-test-secret",
            "aws_region_name": "us-east-1",
        },
        litellm_params={},
        stream=False,
    )

    assert request == {
        "model": arn,
        "messages": messages,
        "max_tokens": 300,
        "temperature": 0.5,
    }
    assert url == (
        "https://bedrock-runtime.us-east-1.amazonaws.com/model/"
        "arn:aws:bedrock:us-east-1:117159858402:imported-model%2Fm4gc1mrfuddy/invoke"
    )


@pytest.mark.parametrize(
    ("arn", "encoded_arn"),
    [
        (
            "arn:aws:bedrock:us-east-1:117159858402:custom-model-deployment/k3n7q2xz8w1v",
            "arn:aws:bedrock:us-east-1:117159858402:custom-model-deployment%2Fk3n7q2xz8w1v",
        ),
        (
            "arn:aws:bedrock:us-east-1:117159858402:custom-model/openai.gpt-oss-20b-1:0/k3n7q2xz8w1v",
            "arn:aws:bedrock:us-east-1:117159858402:custom-model%2Fopenai.gpt-oss-20b-1:0%2Fk3n7q2xz8w1v",
        ),
    ],
)
def test_openai_custom_model_arn_reaches_aws_as_one_path_segment(arn: str, encoded_arn: str) -> None:
    url: Final = AmazonBedrockOpenAIConfig().get_complete_url(
        api_base=None,
        api_key=None,
        model=f"bedrock/openai/{arn}",
        optional_params={
            "aws_access_key_id": "AKIAUNITTESTKEY",
            "aws_secret_access_key": "unit-test-secret",
            "aws_region_name": "us-east-1",
        },
        litellm_params={},
        stream=True,
    )

    assert url == f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{encoded_arn}/invoke-with-response-stream"

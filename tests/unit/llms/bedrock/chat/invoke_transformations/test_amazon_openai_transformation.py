from typing import Final

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


def test_model_id_overrides_model_name_in_url_and_body() -> None:
    profile_arn: Final = "arn:aws:bedrock:ap-northeast-2:123456789012:application-inference-profile/abc123def456"
    model: Final = "bedrock/openai/global.openai.gpt-6-luna"
    messages: Final = [{"role": "user", "content": "Reply with the single word pong"}]
    optional_params: Final = {
        "model_id": profile_arn,
        "max_completion_tokens": 200,
        "aws_access_key_id": "AKIAUNITTESTKEY",
        "aws_secret_access_key": "unit-test-secret",
        "aws_region_name": "ap-northeast-2",
    }
    config: Final = AmazonBedrockOpenAIConfig()
    url: Final = config.get_complete_url(
        api_base=None,
        api_key=None,
        model=model,
        optional_params=dict(optional_params),
        litellm_params={},
        stream=False,
    )
    request: Final = config.transform_request(
        model=model,
        messages=messages,
        optional_params=dict(optional_params),
        litellm_params={},
        headers={},
    )

    assert url == (
        "https://bedrock-runtime.ap-northeast-2.amazonaws.com/model/"
        "arn:aws:bedrock:ap-northeast-2:123456789012:application-inference-profile%2Fabc123def456/invoke"
    )
    assert request == {
        "model": profile_arn,
        "messages": messages,
        "max_completion_tokens": 200,
    }

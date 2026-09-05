from unittest.mock import MagicMock, patch

import pytest

from litellm.llms.bedrock.image_generation.image_handler import BedrockImageGeneration


def test_bedrock_image_prepare_request_with_arn() -> None:
    """Test that ARN model identifiers are correctly URL-encoded in the request endpoint."""
    dummy_arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abcdefghi123"

    image_generation = BedrockImageGeneration()

    with (
        patch(
            "litellm.llms.bedrock.image_generation.image_handler.BedrockImageGeneration._get_boto_credentials_from_optional_params"
        ),
        patch(
            "litellm.llms.bedrock.image_generation.image_handler.BedrockImageGeneration.get_request_headers"
        ),
    ):
        request = image_generation._prepare_request(
            model="amazon.nova-canvas-v1:0",
            prompt="A cute baby sea otter",
            optional_params={
                "model_id": dummy_arn,
            },
            api_base="https://bedrock-runtime.test.com",
            extra_headers=None,
            api_key=None,
            logging_obj=MagicMock(),
        )

    assert (
        request.endpoint_url
        == "https://bedrock-runtime.test.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Aapplication-inference-profile%2Fabcdefghi123/invoke"
    )


def test_bedrock_image_prepare_request_without_arn() -> None:
    """Test that regular model identifiers are used directly in the request endpoint."""
    image_generation = BedrockImageGeneration()

    with (
        patch(
            "litellm.llms.bedrock.image_generation.image_handler.BedrockImageGeneration._get_boto_credentials_from_optional_params"
        ),
        patch(
            "litellm.llms.bedrock.image_generation.image_handler.BedrockImageGeneration.get_request_headers"
        ),
    ):
        request = image_generation._prepare_request(
            model="amazon.nova-canvas-v1:0",
            prompt="A cute baby sea otter",
            optional_params={},
            api_base="https://bedrock-runtime.test.com",
            extra_headers=None,
            api_key=None,
            logging_obj=MagicMock(),
        )

    assert (
        request.endpoint_url
        == "https://bedrock-runtime.test.com/model/amazon.nova-canvas-v1:0/invoke"
    )


@pytest.mark.parametrize(
    "model,aws_region_name,expected_url",
    [
        (
            "us-east-1/amazon.nova-canvas-v1:0",
            None,
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/amazon.nova-canvas-v1:0/invoke",
        ),
        (
            "us-east-1/amazon.nova-canvas-v1:0",
            "eu-west-1",
            "https://bedrock-runtime.eu-west-1.amazonaws.com/model/amazon.nova-canvas-v1:0/invoke",
        ),
        (
            "amazon.nova-canvas-v1:0",
            None,
            "https://bedrock-runtime.us-west-2.amazonaws.com/model/amazon.nova-canvas-v1:0/invoke",
        ),
    ],
)
def test_region_prefixed_image_model_calls_that_region_with_the_bare_model_id(
    monkeypatch: pytest.MonkeyPatch, model: str, aws_region_name: str | None, expected_url: str
) -> None:
    monkeypatch.setenv("AWS_REGION_NAME", "us-west-2")
    monkeypatch.delenv("AWS_BEDROCK_RUNTIME_ENDPOINT", raising=False)
    optional_params = {} if aws_region_name is None else {"aws_region_name": aws_region_name}

    request = BedrockImageGeneration()._prepare_request(
        model=model,
        prompt="A cute baby sea otter",
        optional_params=optional_params,
        api_base=None,
        extra_headers=None,
        api_key="test-bearer-token",
        logging_obj=MagicMock(),
    )

    assert request.endpoint_url == expected_url

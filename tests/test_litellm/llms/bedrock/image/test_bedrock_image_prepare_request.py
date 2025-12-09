from unittest.mock import patch, MagicMock

from litellm.llms.bedrock.image.image_handler import BedrockImageGeneration

def test_bedrock_image_prepare_request_with_arn() -> None:
    dummy_arn = "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abcdefghi123"

    image_generation = BedrockImageGeneration()

    with (
        patch("litellm.llms.bedrock.image.image_handler.BedrockImageGeneration._get_boto_credentials_from_optional_params"),
        patch("litellm.llms.bedrock.image.image_handler.BedrockImageGeneration.get_request_headers"),
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

    assert request.endpoint_url == "https://bedrock-runtime.test.com/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Aapplication-inference-profile%2Fabcdefghi123/invoke"


def test_bedrock_image_prepare_request_without_arn() -> None:
    image_generation = BedrockImageGeneration()

    with (
        patch("litellm.llms.bedrock.image.image_handler.BedrockImageGeneration._get_boto_credentials_from_optional_params"),
        patch("litellm.llms.bedrock.image.image_handler.BedrockImageGeneration.get_request_headers"),
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

    assert request.endpoint_url == "https://bedrock-runtime.test.com/model/amazon.nova-canvas-v1:0/invoke"

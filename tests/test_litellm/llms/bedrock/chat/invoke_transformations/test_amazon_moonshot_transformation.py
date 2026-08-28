from litellm.llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation import (
    AmazonMoonshotConfig,
)


def test_transform_request_bearer_token_skips_aws_credentials():
    config = AmazonMoonshotConfig()

    transformed = config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "aws_profile_name": "litellm-profile-that-does-not-exist",
            "aws_region_name": "us-east-1",
        },
        litellm_params={"api_key": "bedrock-bearer-token"},
        headers={},
    )

    assert transformed["model"] == "moonshot.kimi-k2-thinking"

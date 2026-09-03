import pytest

from litellm.llms.bedrock.chat.invoke_transformations.amazon_moonshot_transformation import (
    AmazonMoonshotConfig,
)

AWS_AUTH_PARAMS = {
    "aws_access_key_id": "AKIAEXAMPLE",
    "aws_secret_access_key": "secret",
    "aws_session_token": "token",
    "aws_region_name": "us-west-2",
    "aws_session_name": "session",
    "aws_role_name": "arn:aws:iam::000000000000:role/example",
    "aws_web_identity_token": "web-identity",
    "aws_sts_endpoint": "https://sts.us-west-2.amazonaws.com",
    "aws_bedrock_runtime_endpoint": "https://bedrock-runtime.us-west-2.amazonaws.com",
    "aws_external_id": "external",
}


def test_transform_request_never_resolves_aws_credentials():
    """A broken credential chain must not stop the request body from being built."""
    config = AmazonMoonshotConfig()

    transformed = config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"aws_profile_name": "litellm-profile-that-does-not-exist", "max_tokens": 16},
        litellm_params={},
        headers={},
    )

    assert transformed["model"] == "moonshot.kimi-k2-thinking"
    assert transformed["max_tokens"] == 16
    assert "aws_profile_name" not in transformed


@pytest.mark.parametrize("aws_param", sorted(AWS_AUTH_PARAMS))
def test_transform_request_keeps_aws_params_out_of_the_body(aws_param: str):
    config = AmazonMoonshotConfig()

    transformed = config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={aws_param: AWS_AUTH_PARAMS[aws_param]},
        litellm_params={},
        headers={},
    )

    assert aws_param not in transformed


def test_transform_request_leaves_the_caller_aws_params_in_place_for_signing():
    """sign_request reads the aws_* keys off optional_params after transform_request runs."""
    config = AmazonMoonshotConfig()
    optional_params = dict(AWS_AUTH_PARAMS)

    config.transform_request(
        model="bedrock/invoke/moonshot.kimi-k2-thinking",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    assert optional_params == AWS_AUTH_PARAMS

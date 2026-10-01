from unittest.mock import Mock

def test_image_generation_bearer_token_never_runs_the_sigv4_credential_chain(monkeypatch):
    """The deployment's AWS profile does not exist, so resolving SigV4 credentials
    raises; a bearer-token deployment must still sign the request with the
    bearer token alone."""
    from litellm.llms.bedrock.image_generation.image_handler import BedrockImageGeneration

    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-bearer-token-12345")

    request = BedrockImageGeneration()._prepare_request(
        model="amazon.nova-canvas-v1:0",
        prompt="A cute baby sea otter",
        optional_params={"aws_region_name": "us-west-2", "aws_profile_name": "litellm-no-such-aws-profile"},
        api_base=None,
        extra_headers=None,
        api_key=None,
        logging_obj=Mock(),
    )

    assert request.prepped.headers["Authorization"] == "Bearer env-bearer-token-12345"

import pytest
from pydantic import ValidationError

from litellm.types.llms.bedrock import AWS_AUTH_PARAM_KEYS, AwsAuthParams


def test_model_validate_keeps_auth_params_and_ignores_request_params():
    auth_params = AwsAuthParams.model_validate(
        {
            "aws_role_name": "arn:aws:iam::999999999999:role/litellm-role",
            "aws_session_name": "litellm-session",
            "aws_external_id": "litellm-external-id",
            "aws_region_name": "us-west-2",
            "aws_bedrock_runtime_endpoint": "https://bedrock.example.com",
            "model": "anthropic.claude-haiku-4-5-20251001-v1:0",
            "temperature": 0.1,
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    assert auth_params.aws_role_name == "arn:aws:iam::999999999999:role/litellm-role"
    assert auth_params.aws_session_name == "litellm-session"
    assert auth_params.aws_external_id == "litellm-external-id"
    assert auth_params.aws_access_key_id is None
    assert set(auth_params.model_dump()) == set(AWS_AUTH_PARAM_KEYS)
    assert not set(AWS_AUTH_PARAM_KEYS) & {"aws_region_name", "aws_bedrock_runtime_endpoint", "model", "temperature"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("aws_role_name", 1234),
        ("aws_session_name", ["litellm-session"]),
        ("aws_external_id", {"id": "x"}),
    ],
)
def test_model_validate_rejects_non_string_credentials(field, value):
    with pytest.raises(ValidationError):
        AwsAuthParams.model_validate({field: value})


def test_frozen_struct_rejects_field_assignment():
    auth_params = AwsAuthParams(aws_role_name="arn:aws:iam::999999999999:role/litellm-role")

    with pytest.raises(ValidationError):
        auth_params.aws_role_name = "arn:aws:iam::999999999999:role/other-role"

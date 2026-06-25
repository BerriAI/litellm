"""
Regression: Bedrock credentials dropped by ``CredentialLiteLLMParams``.

``Router.get_deployment_credentials_with_provider`` resolves a
deployment's upstream credentials by re-validating ``litellm_params``
through ``CredentialLiteLLMParams``::

    return CredentialLiteLLMParams(
        **deployment.litellm_params.model_dump(exclude_none=True)
    ).model_dump(exclude_none=True)

``CredentialLiteLLMParams`` has no ``extra="allow"``, so any field it
does not declare is silently dropped. ``BaseAWSLLM.get_credentials``
consumes ten AWS parameters, but the class declared only three
(``aws_access_key_id``, ``aws_secret_access_key``, ``aws_region_name``).
The Bedrock batch transform additionally needs ``aws_batch_role_arn``
and the S3 bucket/region config.

Effect: a Bedrock deployment that authenticates by assumed role,
profile, STS session token, or web identity, or that configures its S3
buckets in proxy config, lost those values on every ``/v1/files``,
``/v1/batches`` and passthrough call. ``POST /v1/batches`` failed with
"AWS IAM role ARN is required for Bedrock batch jobs" because
``aws_batch_role_arn`` never survived the resolver.

Same failure mode as the Azure ``azure_ad_token`` drop (#30235).
"""

import pytest

AWS_CREDENTIAL_FIELDS = {
    "aws_session_token": "FwoGZXIvYXdzEXAMPLESESSIONTOKEN",
    "aws_session_name": "litellm-batch-session",
    "aws_profile_name": "bedrock-batch",
    "aws_role_name": "arn:aws:iam::123456789012:role/bedrock-caller",
    "aws_web_identity_token": "eyJhbGciOiJEXAMPLE",
    "aws_sts_endpoint": "https://sts.us-east-1.amazonaws.com",
    "aws_external_id": "external-id-xyz",
}

S3_CONFIG_FIELDS = {
    "s3_bucket_name": "litellm-batch-input",
    "s3_output_bucket_name": "litellm-batch-output",
    "s3_region_name": "us-east-1",
}

BATCH_ROLE_FIELD = {
    "aws_batch_role_arn": "arn:aws:iam::123456789012:role/bedrock-batch"
}


class TestCredentialLiteLLMParamsBedrockFields:
    def test_aws_batch_role_arn_round_trips(self):
        from litellm.types.router import CredentialLiteLLMParams

        dumped = CredentialLiteLLMParams(**BATCH_ROLE_FIELD).model_dump(
            exclude_none=True
        )
        assert dumped["aws_batch_role_arn"] == BATCH_ROLE_FIELD["aws_batch_role_arn"], (
            "aws_batch_role_arn dropped by CredentialLiteLLMParams — POST /v1/batches "
            "fails 'AWS IAM role ARN is required'"
        )

    @pytest.mark.parametrize("field,value", list(AWS_CREDENTIAL_FIELDS.items()))
    def test_aws_auth_field_round_trips(self, field, value):
        from litellm.types.router import CredentialLiteLLMParams

        dumped = CredentialLiteLLMParams(**{field: value}).model_dump(exclude_none=True)
        assert dumped.get(field) == value, (
            f"{field} dropped by CredentialLiteLLMParams; get_credentials consumes it, "
            "so deployments using this AWS auth method lose it through the resolver"
        )

    @pytest.mark.parametrize("field,value", list(S3_CONFIG_FIELDS.items()))
    def test_s3_config_field_round_trips(self, field, value):
        from litellm.types.router import CredentialLiteLLMParams

        dumped = CredentialLiteLLMParams(**{field: value}).model_dump(exclude_none=True)
        assert dumped.get(field) == value, (
            f"{field} dropped by CredentialLiteLLMParams; Bedrock batch/file transforms "
            "read it from litellm_params for bucket/region resolution"
        )

    def test_new_fields_are_optional(self):
        """Deployments that don't set these must be unaffected: defaults are
        None and excluded by ``exclude_none``."""
        from litellm.types.router import CredentialLiteLLMParams

        dumped = CredentialLiteLLMParams(api_key="sk-static").model_dump(
            exclude_none=True
        )
        for field in {
            *AWS_CREDENTIAL_FIELDS,
            *S3_CONFIG_FIELDS,
            *BATCH_ROLE_FIELD,
        }:
            assert field not in dumped
        assert dumped["api_key"] == "sk-static"


class TestRouterBedrockCredentialResolution:
    def test_resolver_preserves_bedrock_batch_credentials(self):
        from litellm import Router

        deployment_id = "bedrock-batch-deployment-fixed-uuid"
        litellm_params = {
            "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "aws_region_name": "us-east-1",
            **BATCH_ROLE_FIELD,
            **S3_CONFIG_FIELDS,
            **AWS_CREDENTIAL_FIELDS,
        }
        router = Router(
            model_list=[
                {
                    "model_name": "bedrock-batch-haiku",
                    "litellm_params": litellm_params,
                    "model_info": {"id": deployment_id},
                }
            ]
        )

        credentials = router.get_deployment_credentials_with_provider(
            model_id=deployment_id
        )
        assert credentials is not None
        assert credentials["custom_llm_provider"] == "bedrock"
        for field, value in {
            **BATCH_ROLE_FIELD,
            **S3_CONFIG_FIELDS,
            **AWS_CREDENTIAL_FIELDS,
        }.items():
            assert credentials.get(field) == value, (
                f"Router credential resolution dropped {field}; the batch/file callers "
                "cannot authenticate or resolve the S3 bucket"
            )

    def test_resolver_static_key_deployment_unaffected(self):
        """A static-key deployment with none of the new fields keeps its key
        and gains no spurious AWS fields."""
        from litellm import Router

        deployment_id = "bedrock-static-key-deployment-fixed-uuid"
        router = Router(
            model_list=[
                {
                    "model_name": "bedrock-static",
                    "litellm_params": {
                        "model": "bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
                        "aws_access_key_id": "AKIAEXAMPLE",
                        "aws_secret_access_key": "secret",
                        "aws_region_name": "us-east-1",
                    },
                    "model_info": {"id": deployment_id},
                }
            ]
        )

        credentials = router.get_deployment_credentials_with_provider(
            model_id=deployment_id
        )
        assert credentials is not None
        assert credentials["aws_access_key_id"] == "AKIAEXAMPLE"
        for field in {*AWS_CREDENTIAL_FIELDS, *S3_CONFIG_FIELDS, *BATCH_ROLE_FIELD}:
            assert field not in credentials

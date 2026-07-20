"""
Regression for Bedrock batch creation through the proxy.

``Router.get_deployment_credentials_with_provider`` resolves the upstream
credentials for a deployment by model id for the proxy's ``/v1/files`` and
``/v1/batches`` routing. It builds the credentials by round-tripping
``litellm_params`` through ``CredentialLiteLLMParams``::

    CredentialLiteLLMParams(
        **deployment.litellm_params.model_dump(exclude_none=True)
    ).model_dump(exclude_none=True)

That re-validation is strict: any field not declared on
``CredentialLiteLLMParams`` is silently dropped. Two Bedrock-batch bugs came
from that plus a missing field on the resolved dict:

1. ``aws_batch_role_arn`` (and the other AWS/S3 job fields) were undeclared, so
   a Bedrock deployment lost its IAM role on the way to ``create_batch`` and the
   transform raised "AWS IAM role ARN is required for Bedrock batch jobs".
2. The resolved credentials never carried the deployment's real ``model``, so
   the batch endpoints forwarded the proxy model-group name (alias) as the
   Bedrock ``modelId``. Bedrock create needs the real model id (e.g.
   ``bedrock/us.anthropic...``); the alias is rejected by AWS.

The tests below pin both: the strict credential model keeps the AWS/S3 batch
fields, and the resolver returns the deployment's underlying model.
"""


class TestCredentialLiteLLMParamsBedrockFields:
    def test_aws_batch_role_arn_round_trips_through_model_dump(self):
        from litellm.types.router import CredentialLiteLLMParams

        params = CredentialLiteLLMParams(
            aws_access_key_id="AKIA-xyz",
            aws_secret_access_key="secret-xyz",
            aws_region_name="us-east-1",
            aws_batch_role_arn="arn:aws:iam::123:role/batch",
            s3_bucket_name="my-bucket",
        )
        dumped = params.model_dump(exclude_none=True)
        assert dumped["aws_batch_role_arn"] == "arn:aws:iam::123:role/batch", (
            "aws_batch_role_arn dropped from CredentialLiteLLMParams.model_dump(); "
            "the proxy's model-based batch routing loses the IAM role and Bedrock "
            "create_batch fails with 'AWS IAM role ARN is required'"
        )
        assert dumped["aws_access_key_id"] == "AKIA-xyz"
        assert dumped["aws_secret_access_key"] == "secret-xyz"
        assert dumped["s3_bucket_name"] == "my-bucket"

    def test_bedrock_fields_are_optional(self):
        from litellm.types.router import CredentialLiteLLMParams

        params = CredentialLiteLLMParams(api_key="sk-static")
        dumped = params.model_dump(exclude_none=True)
        assert "aws_batch_role_arn" not in dumped
        assert dumped["api_key"] == "sk-static"


class TestRouterBedrockBatchCredentialResolution:
    def _bedrock_router(self, deployment_id: str):
        from litellm import Router

        return Router(
            model_list=[
                {
                    "model_name": "bedrock-batch-sonnet",
                    "litellm_params": {
                        "model": "bedrock/us.anthropic.claude-sonnet-4-6",
                        "aws_access_key_id": "AKIA-deployment",
                        "aws_secret_access_key": "secret-deployment",
                        "aws_region_name": "us-east-1",
                        "s3_bucket_name": "my-bucket",
                        "aws_batch_role_arn": "arn:aws:iam::123:role/batch",
                    },
                    "model_info": {"id": deployment_id},
                }
            ]
        )

    def test_credentials_preserve_aws_batch_role_arn(self):
        router = self._bedrock_router("bedrock-batch-deployment")

        credentials = router.get_deployment_credentials_with_provider(
            model_id="bedrock-batch-deployment"
        )
        assert credentials is not None
        assert credentials["custom_llm_provider"] == "bedrock"
        assert credentials.get("aws_batch_role_arn") == "arn:aws:iam::123:role/batch", (
            "Router credential resolution dropped aws_batch_role_arn; Bedrock "
            "create_batch cannot build the model-invocation-job request"
        )
        assert credentials.get("aws_access_key_id") == "AKIA-deployment"
        assert credentials.get("aws_secret_access_key") == "secret-deployment"

    def test_credentials_include_deployment_model(self):
        """The resolved credentials must carry the deployment's real model so the
        batch endpoints forward it (not the proxy alias) as the Bedrock modelId."""
        router = self._bedrock_router("bedrock-batch-deployment")

        by_id = router.get_deployment_credentials_with_provider(
            model_id="bedrock-batch-deployment"
        )
        assert by_id is not None
        assert by_id.get("model") == "bedrock/us.anthropic.claude-sonnet-4-6", (
            "Router credential resolution did not return the deployment model; the "
            "batch endpoints would send the proxy model-group name as the Bedrock "
            "modelId, which AWS rejects"
        )

        by_group = router.get_deployment_credentials_with_provider(
            model_id="bedrock-batch-sonnet"
        )
        assert by_group is not None
        assert by_group.get("model") == "bedrock/us.anthropic.claude-sonnet-4-6"

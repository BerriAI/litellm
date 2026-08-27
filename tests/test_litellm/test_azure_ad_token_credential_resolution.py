"""
Regression for #30235.

``Router.get_deployment_credentials_with_provider`` (router.py:8954) is
used by the proxy's ``/v1/files``, ``/v1/batches`` and passthrough
routing code paths to resolve the upstream credentials for a deployment
by model_id::

    return CredentialLiteLLMParams(
        **deployment.litellm_params.model_dump(exclude_none=True)
    ).model_dump(exclude_none=True)

That re-validation is strict. Any field NOT declared on
``CredentialLiteLLMParams`` gets dropped on the way through, even when
it was present on the original ``litellm_params``.

Pre-fix, ``azure_ad_token`` was undeclared, so Azure deployments
configured with OAuth/M2M (``azure_ad_token`` in place of ``api_key``)
silently lost their token on every file upload and the proxy returned::

    Missing credentials. Please pass one of api_key, azure_ad_token,
    azure_ad_token_provider, ...

Tests below pin two things:
1. ``CredentialLiteLLMParams`` directly accepts and round-trips
   ``azure_ad_token``.
2. ``Router.get_deployment_credentials_with_provider`` preserves
   ``azure_ad_token`` from a deployment's ``litellm_params``.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestCredentialLiteLLMParamsAzureAdToken:
    def test_azure_ad_token_round_trips_through_model_dump(self):
        from litellm.types.router import CredentialLiteLLMParams

        params = CredentialLiteLLMParams(
            api_base="https://my.openai.azure.com",
            api_version="2024-08-01-preview",
            azure_ad_token="oauth-bearer-token-xyz",
        )
        dumped = params.model_dump(exclude_none=True)
        assert dumped["azure_ad_token"] == "oauth-bearer-token-xyz", (
            "azure_ad_token dropped from CredentialLiteLLMParams.model_dump() — "
            "every callsite that round-trips litellm_params through this class "
            "will lose the token (#30235)"
        )

    def test_azure_ad_token_is_optional(self):
        """Adding the field must not break deployments that don't use it
        — confirm the default is None and it's excluded by
        ``exclude_none``."""
        from litellm.types.router import CredentialLiteLLMParams

        params = CredentialLiteLLMParams(api_key="sk-static")
        dumped = params.model_dump(exclude_none=True)
        assert "azure_ad_token" not in dumped
        assert dumped["api_key"] == "sk-static"

    def test_round_trip_preserves_full_credential_shape(self):
        """The Router's get_deployment_credentials_with_provider pattern:
        construct from a dict that has azure_ad_token alongside other
        fields, dump, expect azure_ad_token to ride through alongside
        the other declared fields."""
        from litellm.types.router import CredentialLiteLLMParams

        source = {
            "api_base": "https://my.openai.azure.com",
            "api_version": "2024-08-01-preview",
            "azure_ad_token": "tok-123",
            "api_key": None,  # M2M deployment has no static key
        }
        rebuilt = CredentialLiteLLMParams(
            **{k: v for k, v in source.items() if v is not None}
        ).model_dump(exclude_none=True)
        assert rebuilt.get("azure_ad_token") == "tok-123"
        assert rebuilt.get("api_base") == "https://my.openai.azure.com"
        assert "api_key" not in rebuilt


class TestRouterCredentialResolution:
    """The actual fix surface: Router.get_deployment_credentials_with_provider
    must preserve azure_ad_token on the resolved credentials dict so the
    files endpoint can forward it to the Azure files client."""

    def test_credentials_preserve_azure_ad_token(self):
        from litellm import Router

        deployment_id = "azure-m2m-deployment-fixed-uuid"
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-azure-m2m",
                    "litellm_params": {
                        "model": "azure/gpt-4o",
                        "api_base": "https://my.openai.azure.com",
                        "api_version": "2024-08-01-preview",
                        "azure_ad_token": "tok-azure-m2m-xyz",
                    },
                    "model_info": {"id": deployment_id},
                }
            ]
        )

        credentials = router.get_deployment_credentials_with_provider(
            model_id=deployment_id
        )
        assert credentials is not None
        assert credentials.get("azure_ad_token") == "tok-azure-m2m-xyz", (
            "Router credential resolution dropped azure_ad_token; the "
            "files / batches / passthrough callers will not be able to "
            "authenticate against Azure (#30235)"
        )

    def test_credentials_static_api_key_unaffected(self):
        """Don't break the pre-fix happy path: a deployment with a
        static api_key (no azure_ad_token) keeps its api_key and
        azure_ad_token doesn't appear in the dump."""
        from litellm import Router

        deployment_id = "azure-static-key-deployment-fixed-uuid"
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o-azure-static",
                    "litellm_params": {
                        "model": "azure/gpt-4o",
                        "api_base": "https://my.openai.azure.com",
                        "api_version": "2024-08-01-preview",
                        "api_key": "sk-static-key",
                    },
                    "model_info": {"id": deployment_id},
                }
            ]
        )

        credentials = router.get_deployment_credentials_with_provider(
            model_id=deployment_id
        )
        assert credentials is not None
        assert credentials.get("api_key") == "sk-static-key"
        assert "azure_ad_token" not in credentials


class TestRouterCredentialResolutionS3OutputBucket:
    """Same strict-dump trap as azure_ad_token (#30235), for Bedrock batch
    file retrieval (#26335). Bedrock batch outputs land in a per-model
    ``s3_output_bucket_name`` when it differs from the input bucket. The
    file-content retrieval path validates a file id against the buckets in the
    trusted credential snapshot, and that snapshot is built by round-tripping
    the deployment's ``litellm_params`` through ``CredentialLiteLLMParams``. If
    the field is undeclared it is dropped, so the output bucket never reaches
    retrieval and output-bucket file ids are rejected as foreign."""

    def test_credentials_preserve_s3_output_bucket_name(self):
        from litellm import Router

        deployment_id = "bedrock-batch-output-bucket-fixed-uuid"
        router = Router(
            model_list=[
                {
                    "model_name": "bedrock-batch",
                    "litellm_params": {
                        "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                        "s3_bucket_name": "in-bucket",
                        "s3_output_bucket_name": "out-bucket",
                        "aws_region_name": "us-west-2",
                    },
                    "model_info": {"id": deployment_id},
                }
            ]
        )

        credentials = router.get_deployment_credentials_with_provider(
            model_id=deployment_id
        )
        assert credentials is not None
        assert credentials.get("s3_output_bucket_name") == "out-bucket", (
            "Router credential resolution dropped s3_output_bucket_name; "
            "Bedrock batch file-content retrieval will reject output-bucket "
            "file ids as foreign for model-routed deployments (#26335)"
        )
        assert credentials.get("s3_bucket_name") == "in-bucket"

    def test_credentials_without_output_bucket_unaffected(self):
        """A deployment that configures only the input bucket keeps it and does
        not gain a phantom output bucket in the resolved credentials."""
        from litellm import Router

        deployment_id = "bedrock-batch-input-only-fixed-uuid"
        router = Router(
            model_list=[
                {
                    "model_name": "bedrock-batch-input-only",
                    "litellm_params": {
                        "model": "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
                        "s3_bucket_name": "in-bucket",
                        "aws_region_name": "us-west-2",
                    },
                    "model_info": {"id": deployment_id},
                }
            ]
        )

        credentials = router.get_deployment_credentials_with_provider(
            model_id=deployment_id
        )
        assert credentials is not None
        assert credentials.get("s3_bucket_name") == "in-bucket"
        assert "s3_output_bucket_name" not in credentials

"""Single source of truth for the destination-override credential policy.

A non-admin must not redirect a model's endpoint and have the proxy's stored or
ambient credentials sent there. The credential that actually authenticates a
request depends on the destination's provider, so requiring "any credential" is
insufficient: an api_key does not stop Bedrock/SageMaker from signing with
ambient AWS credentials. Shared by the model-management write paths and
/health/test_connection so the two policies cannot drift.
"""

from typing import Tuple

# Caller-controllable endpoint URLs that retarget the outbound request. Mirrors
# the endpoint-redirect subset of auth_utils._BANNED_REQUEST_BODY_PARAMS.
URL_DESTINATION_FIELDS: Tuple[str, ...] = (
    "api_base",
    "base_url",
    "aws_bedrock_runtime_endpoint",
    "aws_sts_endpoint",
    "sagemaker_base_url",
    "s3_endpoint_url",
    "deployment_url",
)

# Credential fields that must never be silently re-pointed at a new endpoint.
CREDENTIAL_FIELDS: Tuple[str, ...] = (
    "api_key",
    "litellm_credential_name",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_web_identity_token",
    "azure_ad_token",
    "vertex_credentials",
)

_AWS_CREDENTIALS: Tuple[str, ...] = (
    "aws_secret_access_key",
    "aws_session_token",
    "aws_web_identity_token",
)
# api_key-class credentials consumed by OpenAI-compatible / generic providers.
_INLINE_CREDENTIALS: Tuple[str, ...] = (
    "api_key",
    "litellm_credential_name",
    "azure_ad_token",
    "vertex_credentials",
)
# AWS providers sign with ambient credentials (instance role / IRSA web identity)
# when no explicit AWS credential is given, so only an AWS credential makes a
# redirect of these fields self-contained.
_CONSUMED_CREDENTIALS = {
    "aws_bedrock_runtime_endpoint": _AWS_CREDENTIALS,
    "aws_sts_endpoint": _AWS_CREDENTIALS,
    "sagemaker_base_url": _AWS_CREDENTIALS,
    "s3_endpoint_url": _AWS_CREDENTIALS,
}


def consumed_credentials_for(destination_field: str) -> Tuple[str, ...]:
    """Credentials the destination's provider actually authenticates with.

    Supplying a credential outside this set does not make redirecting the field
    safe: the provider would fall back to the proxy's ambient/stored credentials
    and send them to the new endpoint.
    """
    return _CONSUMED_CREDENTIALS.get(destination_field, _INLINE_CREDENTIALS)

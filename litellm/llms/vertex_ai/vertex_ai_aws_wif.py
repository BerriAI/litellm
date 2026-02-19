"""
AWS Workload Identity Federation (WIF) auth for Vertex AI.

Handles explicit AWS credentials for GCP WIF token exchange,
bypassing the EC2 instance metadata service.

When aws_* keys are present in the WIF credential JSON, this module
uses BaseAWSLLM to obtain AWS credentials and wraps them in a custom
AwsSecurityCredentialsSupplier for google-auth.
"""

from typing import Dict

GOOGLE_IMPORT_ERROR_MESSAGE = (
    "Google Cloud SDK not found. Install it with: pip install 'litellm[google]' "
    "or pip install google-cloud-aiplatform"
)

# AWS params recognized in WIF credential JSON for explicit auth.
# These match the kwargs accepted by BaseAWSLLM.get_credentials().
_AWS_CREDENTIAL_KEYS = frozenset({
    "aws_access_key_id",
    "aws_secret_access_key",
    "aws_session_token",
    "aws_region_name",
    "aws_session_name",
    "aws_profile_name",
    "aws_role_name",
    "aws_web_identity_token",
    "aws_sts_endpoint",
    "aws_external_id",
})


class VertexAIAwsWifAuth:
    """
    Handles AWS-to-GCP Workload Identity Federation credential creation
    for Vertex AI, using explicit AWS credentials rather than EC2 metadata.
    """

    @staticmethod
    def extract_aws_params(json_obj: dict) -> Dict[str, str]:
        """
        Extract LiteLLM-specific aws_* keys from a WIF credential JSON dict.

        Returns a dict of {param_name: value} for any recognized aws_* keys
        found in the JSON. Returns empty dict if none are present.
        """
        return {
            key: json_obj[key]
            for key in _AWS_CREDENTIAL_KEYS
            if key in json_obj
        }

    @staticmethod
    def credentials_from_explicit_aws(json_obj, aws_params, scopes):
        """
        Create GCP credentials using explicit AWS credentials for WIF.

        Uses BaseAWSLLM to obtain AWS credentials (via STS AssumeRole, profile,
        static keys, etc.), then wraps them in a custom AwsSecurityCredentialsSupplier
        so that google-auth bypasses the EC2 metadata service.

        Args:
            json_obj: The WIF credential JSON dict (contains audience, token_url, etc.)
            aws_params: Dict of aws_* params extracted from json_obj
            scopes: OAuth scopes for the GCP credentials
        """
        try:
            from google.auth import aws
        except ImportError:
            raise ImportError(GOOGLE_IMPORT_ERROR_MESSAGE)

        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
        from litellm.llms.vertex_ai.aws_credentials_supplier import (
            AwsCredentialsSupplier,
        )

        # Validate region first — required for the GCP token exchange.
        # Check before get_credentials() to avoid unnecessary AWS API calls
        # (e.g. STS AssumeRole) on misconfiguration.
        aws_region = aws_params.get("aws_region_name")
        if not aws_region:
            raise ValueError(
                "aws_region_name is required in the WIF credential JSON "
                "when using explicit AWS authentication. Add "
                '"aws_region_name": "<your-region>" to your credential file.'
            )

        # Use BaseAWSLLM to get AWS credentials via any supported auth flow
        base_aws = BaseAWSLLM()
        boto3_credentials = base_aws.get_credentials(**aws_params)

        # Create the custom supplier that wraps boto3 credentials
        supplier = AwsCredentialsSupplier(
            boto3_credentials=boto3_credentials,
            aws_region=aws_region,
        )

        # Build kwargs for aws.Credentials — forward optional fields from JSON
        creds_kwargs = dict(
            audience=json_obj.get("audience"),
            subject_token_type=json_obj.get("subject_token_type"),
            token_url=json_obj.get("token_url"),
            credential_source=None,  # Not using metadata endpoints
            aws_security_credentials_supplier=supplier,
            service_account_impersonation_url=json_obj.get(
                "service_account_impersonation_url"
            ),
        )
        # Forward universe_domain if present (defaults to googleapis.com)
        if "universe_domain" in json_obj:
            creds_kwargs["universe_domain"] = json_obj["universe_domain"]

        creds = aws.Credentials(**creds_kwargs)

        if scopes and hasattr(creds, "requires_scopes") and creds.requires_scopes:
            creds = creds.with_scopes(scopes)

        return creds

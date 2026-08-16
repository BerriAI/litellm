"""
Custom AWS Security Credentials Supplier for Vertex AI WIF.

Wraps boto3/botocore credentials so that google-auth can use them
for the AWS-to-GCP Workload Identity Federation token exchange
without hitting the EC2 instance metadata service.

Requires google-auth >= 2.29.0.
"""

from typing import Callable

from google.auth import aws


class AwsCredentialsSupplier(aws.AwsSecurityCredentialsSupplier):
    """
    Supplies AWS credentials to google-auth's aws.Credentials for WIF
    token exchange.

    This bypasses the default metadata-based credential retrieval,
    allowing WIF to work in environments where EC2 metadata is blocked.

    Accepts a credentials_provider callable that is invoked on every
    get_aws_security_credentials() call, so that refreshed/rotated
    credentials are picked up automatically (important for temporary
    STS tokens).
    """

    def __init__(self, credentials_provider: Callable, aws_region: str):
        """
        Args:
            credentials_provider: A zero-arg callable that returns a
                botocore.credentials.Credentials object (with access_key,
                secret_key, and token attributes).
            aws_region: The AWS region string (e.g. "us-east-1").
        """
        self._credentials_provider = credentials_provider
        self._region = aws_region

    def get_aws_security_credentials(self, context, request):
        """Return current AWS credentials for the GCP token exchange."""
        current = self._credentials_provider()
        return aws.AwsSecurityCredentials(
            access_key_id=current.access_key,
            secret_access_key=current.secret_key,
            session_token=current.token,
        )

    def get_aws_region(self, context, request):
        """Return the AWS region for credential verification."""
        return self._region

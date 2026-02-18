"""
Custom AWS Security Credentials Supplier for Vertex AI WIF.

Wraps boto3/botocore credentials so that google-auth can use them
for the AWS-to-GCP Workload Identity Federation token exchange
without hitting the EC2 instance metadata service.

Requires google-auth >= 2.29.0.
"""

from google.auth import aws


class AwsCredentialsSupplier(aws.AwsSecurityCredentialsSupplier):
    """
    Supplies pre-obtained AWS credentials to google-auth's
    aws.Credentials for WIF token exchange.

    This bypasses the default metadata-based credential retrieval,
    allowing WIF to work in environments where EC2 metadata is blocked.
    """

    def __init__(self, boto3_credentials, aws_region: str):
        """
        Args:
            boto3_credentials: A botocore.credentials.Credentials object
                with access_key, secret_key, and token attributes.
            aws_region: The AWS region string (e.g. "us-east-1").

        Note:
            Credentials are captured as a point-in-time snapshot. When using
            temporary STS credentials (e.g. AssumeRole), the caller is
            responsible for constructing a new supplier before the underlying
            AWS credentials expire.
        """
        self._credentials = boto3_credentials
        self._region = aws_region

    def get_aws_security_credentials(self, context, request):
        """Return AWS credentials for the GCP token exchange."""
        return aws.AwsSecurityCredentials(
            access_key_id=self._credentials.access_key,
            secret_access_key=self._credentials.secret_key,
            session_token=self._credentials.token,
        )

    def get_aws_region(self, context, request):
        """Return the AWS region for credential verification."""
        return self._region

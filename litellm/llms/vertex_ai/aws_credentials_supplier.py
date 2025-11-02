"""
AWS Credentials Supplier for GCP Workload Identity Federation

This module provides a custom AWS credentials supplier that uses boto3 credentials
instead of EC2 metadata endpoints, enabling AWS to GCP federation in environments
where metadata service access is blocked.
"""

from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from botocore.credentials import Credentials as BotoCredentials
else:
    BotoCredentials = Any


class Boto3AwsSecurityCredentialsSupplier:
    """
    Custom AWS credentials supplier that uses boto3 credentials instead of EC2 metadata endpoints.
    
    This allows AWS to GCP Workload Identity Federation without relying on the metadata service
    (http://169.254.169.254). It wraps boto3 credentials obtained via BaseAWSLLM and provides
    them to Google's aws.Credentials class.
    
    Example:
        ```python
        from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
        from google.auth import aws
        
        # Get AWS credentials using BaseAWSLLM (supports all auth flows)
        aws_llm = BaseAWSLLM()
        boto3_creds = aws_llm.get_credentials(
            aws_role_name="arn:aws:iam::123456789012:role/MyRole",
            aws_session_name="my-session",
            aws_region_name="us-east-1"
        )
        
        # Create custom supplier
        supplier = Boto3AwsSecurityCredentialsSupplier(
            boto3_credentials=boto3_creds,
            aws_region="us-east-1"
        )
        
        # Use with Google's aws.Credentials (bypasses metadata)
        gcp_credentials = aws.Credentials(
            audience="//iam.googleapis.com/projects/.../providers/...",
            subject_token_type="urn:ietf:params:aws:token-type:aws4_request",
            token_url="https://sts.googleapis.com/v1/token",
            aws_security_credentials_supplier=supplier,
            credential_source=None,  # Not using metadata
        )
        ```
    """

    def __init__(
        self, boto3_credentials: BotoCredentials, aws_region: str = "us-east-1"
    ) -> None:
        """
        Initialize the AWS credentials supplier.

        Args:
            boto3_credentials: botocore.credentials.Credentials object from boto3/BaseAWSLLM.
                             This can come from any AWS auth flow (role assumption, profile,
                             web identity token, explicit credentials, etc.)
            aws_region: AWS region name. Defaults to "us-east-1"
        """
        self._credentials = boto3_credentials
        self._region = aws_region

    def get_aws_security_credentials(
        self, context: Any, request: Any
    ) -> Mapping[str, str]:
        """
        Get AWS security credentials from the boto3 credentials object.
        
        This method is called by Google's aws.Credentials class to obtain AWS credentials
        for the token exchange process. It extracts the credentials from the boto3
        Credentials object, handling both frozen and unfrozen credential formats.

        Args:
            context: Supplier context (unused, required by interface)
            request: HTTP request object (unused, required by interface)

        Returns:
            Dict containing:
                - access_key_id: AWS access key ID
                - secret_access_key: AWS secret access key
                - security_token: AWS session token (or empty string if not present)
        """
        # Refresh credentials if needed and get frozen credentials
        # Frozen credentials are immutable snapshots of the current credential values
        if hasattr(self._credentials, "get_frozen_credentials"):
            frozen_creds = self._credentials.get_frozen_credentials()
            return {
                "access_key_id": frozen_creds.access_key,
                "secret_access_key": frozen_creds.secret_key,
                "security_token": frozen_creds.token or "",
            }
        else:
            # Fallback for credentials that don't support get_frozen_credentials
            return {
                "access_key_id": self._credentials.access_key,
                "secret_access_key": self._credentials.secret_key,
                "security_token": getattr(self._credentials, "token", "") or "",
            }

    def get_aws_region(self, context: Any, request: Any) -> str:
        """
        Get the AWS region for credential verification.
        
        This method is called by Google's aws.Credentials class to determine which
        AWS region to use for credential verification requests.

        Args:
            context: Supplier context (unused, required by interface)
            request: HTTP request object (unused, required by interface)

        Returns:
            AWS region name (e.g., "us-east-1", "us-west-2")
        """
        return self._region


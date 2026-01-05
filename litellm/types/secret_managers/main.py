import enum
from typing import Dict, List, Literal, Optional

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class KeyManagementSystem(enum.Enum):
    GOOGLE_KMS = "google_kms"
    AZURE_KEY_VAULT = "azure_key_vault"
    AWS_SECRET_MANAGER = "aws_secret_manager"
    GOOGLE_SECRET_MANAGER = "google_secret_manager"
    HASHICORP_VAULT = "hashicorp_vault"
    CYBERARK = "cyberark"
    LOCAL = "local"
    AWS_KMS = "aws_kms"
    CUSTOM = "custom"


class KeyManagementSettings(LiteLLMPydanticObjectBase):
    hosted_keys: Optional[List] = None
    store_virtual_keys: Optional[bool] = False
    """
    If True, virtual keys created by litellm will be stored in the secret manager
    """
    prefix_for_stored_virtual_keys: str = "litellm/"
    """
    If set, this prefix will be used for stored virtual keys in the secret manager
    """

    access_mode: Literal["read_only", "write_only", "read_and_write"] = "read_only"
    """
    Access mode for the secret manager, when write_only will only use for writing secrets
    """

    primary_secret_name: Optional[str] = None
    """
    If set, will read secrets from this primary secret in the secret manager

    eg. on AWS you can store multiple secret values as K/V pairs in a single secret
    """

    description: Optional[str] = None
    """Optional description attached when creating secrets (visible in AWS console)."""

    tags: Optional[Dict[str, str]] = None
    """Optional tags to attach when creating secrets (e.g. {"Environment": "Prod", "Owner": "AI-Platform"})."""

    custom_secret_manager: Optional[str] = None
    """
    Path to custom secret manager class (e.g. "my_secret_manager.InMemorySecretManager")
    Required when key_management_system is "custom"
    """

    # AWS IAM Role Assumption Settings (for AWS Secret Manager)
    aws_region_name: Optional[str] = None
    """AWS region for Secret Manager operations (e.g., 'us-east-1')"""

    aws_role_name: Optional[str] = None
    """ARN of IAM role to assume for Secret Manager access (e.g., 'arn:aws:iam::123456789012:role/MyRole')"""

    aws_session_name: Optional[str] = None
    """Session name for the assumed role session (optional, auto-generated if not provided)"""

    aws_external_id: Optional[str] = None
    """External ID for role assumption (required for cross-account access)"""

    aws_profile_name: Optional[str] = None
    """AWS profile name to use from ~/.aws/credentials"""

    aws_web_identity_token: Optional[str] = None
    """Web identity token for OIDC/IRSA authentication"""

    aws_sts_endpoint: Optional[str] = None
    """Custom STS endpoint URL (useful for VPC endpoints or testing)"""
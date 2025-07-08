import enum
from typing import List, Literal, Optional

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class KeyManagementSystem(enum.Enum):
    GOOGLE_KMS = "google_kms"
    AZURE_KEY_VAULT = "azure_key_vault"
    AWS_SECRET_MANAGER = "aws_secret_manager"
    GOOGLE_SECRET_MANAGER = "google_secret_manager"
    HASHICORP_VAULT = "hashicorp_vault"
    LOCAL = "local"
    AWS_KMS = "aws_kms"


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
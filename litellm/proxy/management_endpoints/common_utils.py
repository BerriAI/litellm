from typing import Any, Union

from litellm.proxy._types import (
    GenerateKeyRequest,
    LiteLLM_ManagementEndpoint_MetadataFields_Premium,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)
from litellm.proxy.utils import _premium_user_check


def _is_user_team_admin(
    user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable
) -> bool:
    for member in team_obj.members_with_roles:
        if (
            member.user_id is not None and member.user_id == user_api_key_dict.user_id
        ) and member.role == "admin":

            return True

    return False


def _set_object_metadata_field(
    object_data: Union[LiteLLM_TeamTable, GenerateKeyRequest],
    field_name: str,
    value: Any,
) -> None:
    """
    Helper function to set metadata fields that require premium user checks

    Args:
        object_data: The team data object to modify
        field_name: Name of the metadata field to set
        value: Value to set for the field
    """
    if field_name in LiteLLM_ManagementEndpoint_MetadataFields_Premium:
        _premium_user_check()
    object_data.metadata = object_data.metadata or {}
    object_data.metadata[field_name] = value

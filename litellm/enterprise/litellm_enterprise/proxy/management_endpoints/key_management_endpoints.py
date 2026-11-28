from typing import Final, Optional

from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import GenerateKeyRequest, LiteLLM_TeamTable


def add_team_member_key_duration(
    team_table: Optional[LiteLLM_TeamTable],
    data: GenerateKeyRequest,
) -> GenerateKeyRequest:
    if team_table is None:
        return data

    if data.user_id is None:  # only apply for team member keys, not service accounts
        return data

    team_duration: Final[object] = (team_table.metadata or {}).get("team_member_key_duration")
    if not isinstance(team_duration, str) or not team_duration.strip():
        return data

    if data.duration is None or duration_in_seconds(data.duration) > duration_in_seconds(team_duration):
        return data.model_copy(update={"duration": team_duration})

    return data


def add_team_organization_id(
    team_table: Optional[LiteLLM_TeamTable],
    data: GenerateKeyRequest,
) -> GenerateKeyRequest:
    if team_table is None:
        return data
    setattr(data, "organization_id", team_table.organization_id)
    return data


def apply_enterprise_key_management_params(
    data: GenerateKeyRequest,
    team_table: Optional[LiteLLM_TeamTable],
) -> GenerateKeyRequest:

    data = add_team_member_key_duration(team_table, data)
    data = add_team_organization_id(team_table, data)
    return data

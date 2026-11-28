from typing import Optional

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

    if team_table.metadata is None:
        return data

    team_duration = team_table.metadata.get("team_member_key_duration")
    # The UI saves a cleared field as "", which means no team default.
    if not isinstance(team_duration, str) or team_duration.strip() == "":
        return data

    # The team value is the default and the longest a member key can last.
    # A shorter requested duration is kept.
    if data.duration is None or data.duration.strip() == "":
        data.duration = team_duration
    else:
        try:
            exceeds_team = duration_in_seconds(data.duration) > duration_in_seconds(team_duration)
        except ValueError:
            # Either value can't be parsed. Use the team value so the cap is never skipped.
            exceeds_team = True
        if exceeds_team:
            data.duration = team_duration

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

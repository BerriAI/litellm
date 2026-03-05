from typing import Optional

from litellm.proxy._types import GenerateKeyRequest, LiteLLM_TeamTable


def add_team_member_key_duration(
    team_table: Optional[LiteLLM_TeamTable],
    data: GenerateKeyRequest,
) -> GenerateKeyRequest:
    if team_table is None:
        return data

    if data.user_id is None:  # only apply for team member keys, not service accounts
        return data

    if (
        team_table.metadata is not None
        and team_table.metadata.get("team_member_key_duration") is not None
    ):
        team_max = team_table.metadata["team_member_key_duration"]
        # Act as a ceiling: only apply team max when the user didn't provide a
        # duration. If the user explicitly supplied a shorter duration (already
        # validated against team max upstream), respect their value.
        if "duration" not in data.model_fields_set:
            data.duration = team_max

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

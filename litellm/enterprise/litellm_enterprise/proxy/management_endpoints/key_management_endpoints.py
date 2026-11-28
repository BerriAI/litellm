from collections.abc import Mapping
from typing import Final, Optional

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy._types import GenerateKeyRequest, LiteLLM_TeamTable

_METADATA: Final = TypeAdapter(Mapping[str, object])


def _seconds(duration: str) -> float | None:
    if duration == "-1":
        return float("inf")
    try:
        return duration_in_seconds(duration)
    except ValueError:
        return None


def _is_within(requested: str | None, cap: str) -> bool:
    if requested is None:
        return False
    requested_seconds: Final = _seconds(requested)
    cap_seconds: Final = _seconds(cap)
    return requested_seconds is not None and cap_seconds is not None and requested_seconds <= cap_seconds


def add_team_member_key_duration(
    team_table: Optional[LiteLLM_TeamTable],
    data: GenerateKeyRequest,
) -> GenerateKeyRequest:
    """The team's key duration is both the default and the longest a member key may live."""
    if team_table is None:
        return data

    if data.user_id is None:  # only apply for team member keys, not service accounts
        return data

    try:
        metadata: Final = _METADATA.validate_python(team_table.metadata or {})
    except ValidationError:
        return data
    team_duration: Final = metadata.get("team_member_key_duration")
    if not isinstance(team_duration, str) or team_duration.strip() == "":
        return data

    if not _is_within(data.duration, team_duration):
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

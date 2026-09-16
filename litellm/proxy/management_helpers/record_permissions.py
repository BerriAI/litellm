from collections.abc import Sequence

from litellm.proxy._types import KeyManagementRoutes, LiteLLM_TeamTable, UserAPIKeyAuth


def can_read_team_records(auth: UserAPIKeyAuth, team: LiteLLM_TeamTable, permission: KeyManagementRoutes) -> bool:
    from litellm.proxy.management_endpoints.common_utils import _is_user_team_admin, _team_member_has_permission

    return _is_user_team_admin(auth, team) or _team_member_has_permission(auth, team, permission.value)


def permitted_record_teams(
    auth: UserAPIKeyAuth, teams: Sequence[LiteLLM_TeamTable], permission: KeyManagementRoutes
) -> tuple[str, ...]:
    return tuple(team.team_id for team in teams if can_read_team_records(auth, team, permission))

from litellm.proxy._types import LiteLLM_TeamTable, UserAPIKeyAuth


def _is_user_team_admin(
    user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable
) -> bool:
    for member in team_obj.members_with_roles:
        if (
            member.user_id is not None and member.user_id == user_api_key_dict.user_id
        ) and member.role == "admin":

            return True

    return False

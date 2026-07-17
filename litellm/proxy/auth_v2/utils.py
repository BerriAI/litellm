from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Dict, List, Optional

from scim2_models import Email, GroupMember, Name
from scim2_models import Group as ScimGroup
from scim2_models import User as ScimUser

from litellm.proxy.auth_v2.authorization import Role
from litellm.proxy.auth_v2.models import TeamRole

if TYPE_CHECKING:
    from litellm.models.team import LiteLLM_TeamTable, Member
    from litellm.models.user import LiteLLM_UserTable


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_ROLE_MAP: Dict[str, Role] = {
    "proxy_admin": Role.PLATFORM_ADMIN,
    "proxy_admin_viewer": Role.PLATFORM_VIEWER,
    "org_admin": Role.ORG_ADMIN,
}


def map_role(value: Optional[object]) -> Optional[Role]:
    """Map a LiteLLM ``user_role`` (string or LitellmUserRoles) to a platform Role."""
    if isinstance(value, str):
        return _ROLE_MAP.get(value)
    return None


def team_role(role: Optional[str]) -> TeamRole:
    return TeamRole.ADMIN if role == "admin" else TeamRole.MEMBER


def member_role(members: "List[Member]", user_id: Optional[str]) -> TeamRole:
    if user_id is not None:
        for member in members:
            if member.user_id == user_id:
                return team_role(member.role)
    return TeamRole.MEMBER


def scim_user_to_db(user: ScimUser) -> Dict[str, object]:
    email = user.emails[0].value if user.emails else None
    metadata: Dict[str, object] = {"scim_active": user.active}
    if user.name is not None:
        metadata["scim_metadata"] = {
            "givenName": user.name.given_name,
            "familyName": user.name.family_name,
        }
    data: Dict[str, object] = {"metadata": metadata}
    if email is not None:
        data["user_email"] = email
    if user.external_id is not None:
        data["sso_user_id"] = user.external_id
    if user.display_name is not None:
        data["user_alias"] = user.display_name
    return data


def db_user_to_scim(user: "LiteLLM_UserTable") -> ScimUser:
    metadata = getattr(user, "metadata", None) or {}
    scim_name = metadata.get("scim_metadata") or {}
    result = ScimUser(
        external_id=user.sso_user_id or user.user_id,
        user_name=user.user_email or user.user_id,
        display_name=user.user_alias,
        active=metadata.get("scim_active", True),
    )
    result.id = user.user_id
    if user.user_email:
        result.emails = [Email(value=user.user_email, primary=True)]
    if scim_name.get("givenName") or scim_name.get("familyName"):
        result.name = Name(
            given_name=scim_name.get("givenName"),
            family_name=scim_name.get("familyName"),
        )
    return result


def scim_group_to_db(group: ScimGroup) -> Dict[str, object]:
    members = [{"user_id": member.value, "role": "user"} for member in (group.members or [])]
    return {"team_alias": group.display_name, "members_with_roles": members}


def db_team_to_scim(team: "LiteLLM_TeamTable") -> ScimGroup:
    result = ScimGroup(display_name=team.team_alias or team.team_id)
    result.id = team.team_id
    members = [GroupMember(value=member.user_id) for member in (team.members_with_roles or []) if member.user_id]
    if members:
        result.members = members
    return result

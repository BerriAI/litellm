"""
Claude Code marketplace visibility: enabled plugins are public, disabled plugins
are private and resolve only for proxy admins or keys granted them via
``object_permission.skills`` or via ``access_skill_names`` on an access group
attached to the key or its team.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from litellm.proxy._types import UI_TEAM_ID, LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin

if TYPE_CHECKING:
    from prisma.types import LiteLLM_ClaudeCodePluginTableWhereInput

    from litellm.proxy._types import LiteLLM_TeamTable


class _SkillRecord(Protocol):
    name: str
    enabled: bool


def _skills_of(permission: LiteLLM_ObjectPermissionTable | None) -> frozenset[str]:
    return frozenset(permission.skills or ()) if permission is not None else frozenset()


@dataclass(frozen=True, slots=True)
class AccessGroupSkills:
    """Plugin names inherited from access groups, split by the scope the group is attached to."""

    key: frozenset[str] = frozenset()
    team: frozenset[str] = frozenset()
    unresolved: bool = False
    """A listed group returned no row, so the policy is incomplete and only public plugins show."""


NO_ACCESS_GROUP_SKILLS: Final = AccessGroupSkills()


def granted_skills(
    user_api_key_dict: UserAPIKeyAuth, inherited: AccessGroupSkills = NO_ACCESS_GROUP_SKILLS
) -> frozenset[str]:
    """Key grant intersected with the team grant when both are non-empty; either alone applies as is.

    An empty list is the Prisma column default, so it means "no private grants configured here".
    """
    key_skills: Final = _skills_of(user_api_key_dict.object_permission) | inherited.key
    team_skills: Final = _skills_of(user_api_key_dict.team_object_permission) | inherited.team
    match (bool(key_skills), bool(team_skills)):
        case (True, True):
            return key_skills & team_skills
        case (True, False):
            return key_skills
        case _:
            return team_skills


@dataclass(frozen=True, slots=True)
class SkillVisibility:
    granted: frozenset[str]
    sees_private: bool

    def allows(self, skill: _SkillRecord) -> bool:
        return skill.enabled or self.sees_private or skill.name in self.granted

    def where(self) -> "LiteLLM_ClaudeCodePluginTableWhereInput":
        if self.sees_private:
            return {}
        if not self.granted:
            return {"enabled": True}
        return {"OR": [{"enabled": True}, {"name": {"in": sorted(self.granted)}}]}


PUBLIC_ONLY: Final = SkillVisibility(granted=frozenset(), sees_private=False)


def skill_visibility(
    user_api_key_dict: UserAPIKeyAuth | None, inherited: AccessGroupSkills = NO_ACCESS_GROUP_SKILLS
) -> SkillVisibility:
    if user_api_key_dict is None:
        return PUBLIC_ONLY
    if is_proxy_admin(user_api_key_dict):
        return SkillVisibility(granted=frozenset(), sees_private=True)
    if inherited.unresolved:
        return PUBLIC_ONLY
    return SkillVisibility(granted=granted_skills(user_api_key_dict, inherited), sees_private=False)


async def _team_object(user_api_key_dict: UserAPIKeyAuth, check_db_only: bool) -> "LiteLLM_TeamTable | None":
    from litellm.proxy.auth.auth_checks import TeamNotFoundError, get_team_object
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    team_id: Final = user_api_key_dict.team_id
    if not team_id or team_id == UI_TEAM_ID or prisma_client is None:
        return None
    try:
        team_obj: Final = await get_team_object(
            team_id=team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
            check_db_only=check_db_only,
        )
    except TeamNotFoundError:
        return None
    return None if team_obj.blocked else team_obj


async def access_group_skills(user_api_key_dict: UserAPIKeyAuth) -> AccessGroupSkills:
    """Resolve ``access_skill_names`` of the groups attached to the key and to its team.

    Key-side groups count only when the group lists the key or its team as an owner, same as
    the model and MCP checks, so a key cannot claim a group it was never assigned to.
    """
    from litellm.proxy.auth.auth_checks import (
        AccessGroupNotFoundError,
        get_authorized_resources_from_key_access_groups,
        get_skill_names_from_access_groups,
    )
    from litellm.proxy.proxy_server import prisma_client, proxy_logging_obj, user_api_key_cache

    if prisma_client is None:
        return NO_ACCESS_GROUP_SKILLS
    check_db_only: Final = bool(user_api_key_dict.requires_fresh_policy)
    team_obj: Final = await _team_object(user_api_key_dict, check_db_only)
    try:
        key_skills: Final = frozenset(
            await get_authorized_resources_from_key_access_groups(
                valid_token=user_api_key_dict,
                team_object=team_obj,
                resource_field="access_skill_names",
                check_db_only=check_db_only,
                strict_errors=True,
            )
        )
        team_skills: Final = await get_skill_names_from_access_groups(
            access_group_ids=tuple(team_obj.access_group_ids or ()) if team_obj is not None else (),
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            proxy_logging_obj=proxy_logging_obj,
            check_db_only=check_db_only,
            strict_errors=True,
        )
        return AccessGroupSkills(key=key_skills, team=team_skills)
    except AccessGroupNotFoundError:
        return AccessGroupSkills(unresolved=True)


AccessGroupSkillsLoader: TypeAlias = Callable[[UserAPIKeyAuth], Awaitable[AccessGroupSkills]]


async def resolve_skill_visibility(
    user_api_key_dict: UserAPIKeyAuth | None,
    load_access_group_skills: AccessGroupSkillsLoader = access_group_skills,
) -> SkillVisibility:
    """``skill_visibility`` with the access-group grants of the key and its team folded in."""
    if user_api_key_dict is None or is_proxy_admin(user_api_key_dict):
        return skill_visibility(user_api_key_dict)
    return skill_visibility(user_api_key_dict, await load_access_group_skills(user_api_key_dict))

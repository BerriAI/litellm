"""
Claude Code marketplace visibility: enabled plugins are public, disabled plugins
are private and resolve only for proxy admins or keys granted them via
``object_permission.skills``.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.proxy.common_utils.resource_ownership import is_proxy_admin

if TYPE_CHECKING:
    from prisma.types import LiteLLM_ClaudeCodePluginTableWhereInput


class _SkillRecord(Protocol):
    name: str
    enabled: bool


def _skills_of(permission: LiteLLM_ObjectPermissionTable | None) -> frozenset[str]:
    return frozenset(permission.skills or ()) if permission is not None else frozenset()


def granted_skills(user_api_key_dict: UserAPIKeyAuth) -> frozenset[str]:
    """Key grant intersected with the team grant when both are non-empty; either alone applies as is.

    An empty list is the Prisma column default for every object-permission row, so it means
    "no private grants configured here" and defers to the other scope, same as the agents check.
    """
    key_skills: Final = _skills_of(user_api_key_dict.object_permission)
    team_skills: Final = _skills_of(user_api_key_dict.team_object_permission)
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


def skill_visibility(user_api_key_dict: UserAPIKeyAuth | None) -> SkillVisibility:
    if user_api_key_dict is None:
        return PUBLIC_ONLY
    if is_proxy_admin(user_api_key_dict):
        return SkillVisibility(granted=frozenset(), sees_private=True)
    return SkillVisibility(granted=granted_skills(user_api_key_dict), sees_private=False)

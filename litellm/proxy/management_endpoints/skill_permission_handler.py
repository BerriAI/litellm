"""
Skill Permission Handler for LiteLLM Proxy.

Handles skill permission checking for keys and teams using object_permission.skills.
Follows the same pattern as AgentRequestHandler in agent_endpoints/auth/agent_permission_handler.py.
"""

from typing import List, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LiteLLM_ObjectPermissionTable,
    LiteLLM_TeamTable,
    UserAPIKeyAuth,
)


class SkillPermissionHandler:
    """
    Class to handle skill permission checking, including:
    1. Key-level skill permissions
    2. Team-level skill permissions

    Follows the same inheritance logic as agents/MCP:
    - If team has restrictions and key has restrictions: use intersection
    - If team has restrictions and key has none: inherit from team
    - If team has no restrictions: use key restrictions
    - If no restrictions: allow all skills
    """

    @staticmethod
    async def get_allowed_skills(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get list of allowed skill names for the given user/key based on permissions.

        Returns:
            List[str]: List of allowed skill names. Empty list means no restrictions (allow all).
        """
        try:
            allowed_skills_for_key = SkillPermissionHandler._get_allowed_skills_for_key(
                user_api_key_auth
            )
            allowed_skills_for_team = (
                await SkillPermissionHandler._get_allowed_skills_for_team(
                    user_api_key_auth
                )
            )

            # If team has skill restrictions, handle inheritance and intersection logic
            if len(allowed_skills_for_team) > 0:
                if len(allowed_skills_for_key) > 0:
                    # Key has its own skill permissions - use intersection with team
                    allowed_skills = [
                        s
                        for s in allowed_skills_for_key
                        if s in allowed_skills_for_team
                    ]
                else:
                    # Key has no skill permissions - inherit from team
                    allowed_skills = allowed_skills_for_team
            else:
                allowed_skills = allowed_skills_for_key

            return list(set(allowed_skills))
        except Exception as e:
            verbose_proxy_logger.exception(f"Failed to get allowed skills: {str(e)}")
            raise

    @staticmethod
    async def is_skill_allowed(
        skill_name: str,
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> bool:
        """
        Check if a specific skill is allowed for the given user/key.

        Args:
            skill_name: The skill name to check
            user_api_key_auth: User authentication info

        Returns:
            bool: True if skill is allowed, False otherwise
        """
        allowed_skills = await SkillPermissionHandler.get_allowed_skills(
            user_api_key_auth
        )

        # Empty list means no restrictions - allow all
        if len(allowed_skills) == 0:
            return True

        return skill_name in allowed_skills

    @staticmethod
    def _get_key_object_permission(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> Optional[LiteLLM_ObjectPermissionTable]:
        """
        Get key object_permission - already loaded by get_key_object() in main auth flow.
        """
        if not user_api_key_auth:
            return None

        return user_api_key_auth.object_permission

    @staticmethod
    async def _get_team_object_permission(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> Optional[LiteLLM_ObjectPermissionTable]:
        """
        Get team object_permission - automatically loaded by get_team_object() in main auth flow.
        """
        from litellm.proxy.auth.auth_checks import get_team_object
        from litellm.proxy.proxy_server import (
            prisma_client,
            proxy_logging_obj,
            user_api_key_cache,
        )

        if not user_api_key_auth or not user_api_key_auth.team_id or not prisma_client:
            return None

        team_obj: Optional[LiteLLM_TeamTable] = await get_team_object(
            team_id=user_api_key_auth.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_auth.parent_otel_span,
            proxy_logging_obj=proxy_logging_obj,
        )

        if not team_obj:
            return None

        return team_obj.object_permission

    @staticmethod
    def _get_allowed_skills_for_key(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get allowed skills for a key from its object_permission.

        Note: object_permission is already loaded by get_key_object() in main auth flow.
        """
        if user_api_key_auth is None:
            return []

        key_object_permission = SkillPermissionHandler._get_key_object_permission(
            user_api_key_auth
        )
        if key_object_permission is not None:
            return key_object_permission.skills or []

        return []

    @staticmethod
    async def _get_allowed_skills_for_team(
        user_api_key_auth: Optional[UserAPIKeyAuth] = None,
    ) -> List[str]:
        """
        Get allowed skills for a team from its object_permission.
        """
        team_object_permission = (
            await SkillPermissionHandler._get_team_object_permission(user_api_key_auth)
        )
        if team_object_permission is not None:
            return team_object_permission.skills or []

        return []

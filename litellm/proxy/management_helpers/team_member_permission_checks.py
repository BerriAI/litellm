from typing import List, Optional

from litellm.caching import DualCache
from litellm.proxy._types import (
    KeyManagementRoutes,
    LiteLLM_TeamTableCachedObj,
    LiteLLM_VerificationToken,
    LiteLLMRoutes,
    LitellmUserRoles,
    Member,
    ProxyErrorTypes,
    ProxyException,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.auth_checks import get_team_object
from litellm.proxy.auth.route_checks import RouteChecks
from litellm.proxy.utils import PrismaClient

DEFAULT_TEAM_MEMBER_PERMISSIONS = [
    KeyManagementRoutes.KEY_INFO,
    KeyManagementRoutes.KEY_HEALTH,
]


class TeamMemberPermissionChecks:
    @staticmethod
    def get_permissions_for_team_member(
        team_member_object: Member,
        team_table: LiteLLM_TeamTableCachedObj,
    ) -> List[KeyManagementRoutes]:
        """
        Returns the permissions for a team member
        """
        if team_table.team_member_permissions and isinstance(
            team_table.team_member_permissions, list
        ):
            return [
                KeyManagementRoutes(permission)
                for permission in team_table.team_member_permissions
            ]

        return DEFAULT_TEAM_MEMBER_PERMISSIONS

    @staticmethod
    def _get_list_of_route_enum_as_str(
        route_enum: List[KeyManagementRoutes],
    ) -> List[str]:
        """
        Returns a list of the route enum as a list of strings
        """
        return [route.value for route in route_enum]

    @staticmethod
    async def can_team_member_execute_key_management_endpoint(
        user_api_key_dict: UserAPIKeyAuth,
        route: KeyManagementRoutes,
        prisma_client: PrismaClient,
        user_api_key_cache: DualCache,
        existing_key_row: LiteLLM_VerificationToken,
    ):
        """
        Main handler for checking if a team member can update a key
        """
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            _get_user_in_team,
        )

        # 1. Don't execute these checks if the user role is proxy admin
        if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN.value:
            return

        # 2. Check if the operation is being done on a team key
        if existing_key_row.team_id is None:
            return

        # 3. Get Team Object from DB
        team_table = await get_team_object(
            team_id=existing_key_row.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            check_db_only=True,
        )

        # 4. Extract `Member` object from `team_table`
        key_assigned_user_in_team = _get_user_in_team(
            team_table=team_table, user_id=user_api_key_dict.user_id
        )

        # 5. Check if the team member has permissions for the endpoint
        TeamMemberPermissionChecks.does_team_member_have_permissions_for_endpoint(
            team_member_object=key_assigned_user_in_team,
            team_table=team_table,
            route=route,
        )

    @staticmethod
    def does_team_member_have_permissions_for_endpoint(
        team_member_object: Optional[Member],
        team_table: LiteLLM_TeamTableCachedObj,
        route: str,
    ) -> Optional[bool]:
        """
        Raises an exception if the team member does not have permissions for calling the endpoint for a team
        """

        # permission checks only run for non-admin users
        # Non-Admin user trying to access information about a team's key
        if team_member_object is None:
            return False
        if team_member_object.role == "admin":
            return True

        _team_member_permissions = (
            TeamMemberPermissionChecks.get_permissions_for_team_member(
                team_member_object=team_member_object,
                team_table=team_table,
            )
        )
        team_member_permissions = (
            TeamMemberPermissionChecks._get_list_of_route_enum_as_str(
                _team_member_permissions
            )
        )

        if not RouteChecks.check_route_access(
            route=route, allowed_routes=team_member_permissions
        ):
            raise ProxyException(
                message=f"Team member does not have permissions for endpoint: {route}. You only have access to the following endpoints: {team_member_permissions} for team {team_table.team_id}. To create keys for this team, please ask your proxy admin to check the team member permission settings and update the settings to allow team member users to create keys.",
                type=ProxyErrorTypes.team_member_permission_error,
                param=route,
                code=401,
            )

        return True

    @staticmethod
    async def user_belongs_to_keys_team(
        user_api_key_dict: UserAPIKeyAuth,
        existing_key_row: LiteLLM_VerificationToken,
    ) -> bool:
        """
        Returns True if the user belongs to the team that the key is assigned to
        """
        from litellm.proxy.management_endpoints.key_management_endpoints import (
            _get_user_in_team,
        )
        from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

        if existing_key_row.team_id is None:
            return False
        team_table = await get_team_object(
            team_id=existing_key_row.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            parent_otel_span=user_api_key_dict.parent_otel_span,
            check_db_only=True,
        )

        # 4. Extract `Member` object from `team_table`
        team_member_object = _get_user_in_team(
            team_table=team_table, user_id=user_api_key_dict.user_id
        )
        return team_member_object is not None

    @staticmethod
    def get_all_available_team_member_permissions() -> List[str]:
        """
        Returns all available team member permissions
        """
        all_available_permissions = []
        for route in LiteLLMRoutes.key_management_routes.value:
            all_available_permissions.append(route)
        return all_available_permissions

    @staticmethod
    def default_team_member_permissions() -> List[str]:
        return [route.value for route in DEFAULT_TEAM_MEMBER_PERMISSIONS]

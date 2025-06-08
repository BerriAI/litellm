from typing import List, Union

from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    Member,
    NewUserResponse,
)
from litellm.types.proxy.management_endpoints.scim_v2 import *


class ScimTransformations:
    DEFAULT_SCIM_NAME = "Unknown User"
    DEFAULT_SCIM_FAMILY_NAME = "Unknown Family Name"
    DEFAULT_SCIM_DISPLAY_NAME = "Unknown Display Name"
    DEFAULT_SCIM_MEMBER_VALUE = "Unknown Member Value"

    @staticmethod
    async def transform_litellm_user_to_scim_user(
        user: Union[LiteLLM_UserTable, NewUserResponse],
    ) -> SCIMUser:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500, detail={"error": "No database connected"}
            )

        # Get user's teams/groups
        groups = []
        for team_id in user.teams or []:
            team = await prisma_client.db.litellm_teamtable.find_unique(
                where={"team_id": team_id}
            )
            if team:
                team_alias = getattr(team, "team_alias", team.team_id)
                groups.append(SCIMUserGroup(value=team.team_id, display=team_alias))

        user_created_at = user.created_at.isoformat() if user.created_at else None
        user_updated_at = user.updated_at.isoformat() if user.updated_at else None

        emails = []
        if user.user_email:
            emails.append(SCIMUserEmail(value=user.user_email, primary=True))

        return SCIMUser(
            schemas=["urn:ietf:params:scim:schemas:core:2.0:User"],
            id=user.user_id,
            userName=ScimTransformations._get_scim_user_name(user),
            displayName=ScimTransformations._get_scim_user_name(user),
            name=SCIMUserName(
                familyName=ScimTransformations._get_scim_family_name(user),
                givenName=ScimTransformations._get_scim_given_name(user),
            ),
            emails=emails,
            groups=groups,
            active=True,
            meta={
                "resourceType": "User",
                "created": user_created_at,
                "lastModified": user_updated_at,
            },
        )

    @staticmethod
    def _get_scim_user_name(user: Union[LiteLLM_UserTable, NewUserResponse]) -> str:
        """
        SCIM requires a display name with length > 0

        We use the same userName and displayName for SCIM users
        """
        if user.user_email and len(user.user_email) > 0:
            return user.user_email
        return ScimTransformations.DEFAULT_SCIM_DISPLAY_NAME

    @staticmethod
    def _get_scim_family_name(user: Union[LiteLLM_UserTable, NewUserResponse]) -> str:
        """
        SCIM requires a family name with length > 0
        """
        metadata = user.metadata or {}
        if "scim_metadata" in metadata:
            scim_metadata: LiteLLM_UserScimMetadata = LiteLLM_UserScimMetadata(
                **metadata["scim_metadata"]
            )
            if scim_metadata.familyName and len(scim_metadata.familyName) > 0:
                return scim_metadata.familyName

        if user.user_alias and len(user.user_alias) > 0:
            return user.user_alias
        return ScimTransformations.DEFAULT_SCIM_FAMILY_NAME

    @staticmethod
    def _get_scim_given_name(user: Union[LiteLLM_UserTable, NewUserResponse]) -> str:
        """
        SCIM requires a given name with length > 0
        """
        metadata = user.metadata or {}
        if "scim_metadata" in metadata:
            scim_metadata: LiteLLM_UserScimMetadata = LiteLLM_UserScimMetadata(
                **metadata["scim_metadata"]
            )
            if scim_metadata.givenName and len(scim_metadata.givenName) > 0:
                return scim_metadata.givenName

        if user.user_alias and len(user.user_alias) > 0:
            return user.user_alias or ScimTransformations.DEFAULT_SCIM_NAME
        return ScimTransformations.DEFAULT_SCIM_NAME

    @staticmethod
    async def transform_litellm_team_to_scim_group(
        team: Union[LiteLLM_TeamTable, dict],
    ) -> SCIMGroup:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(
                status_code=500, detail={"error": "No database connected"}
            )

        if isinstance(team, dict):
            team = LiteLLM_TeamTable(**team)

        # Get team members
        scim_members: List[SCIMMember] = []
        for member in team.members_with_roles or []:
            scim_members.append(
                SCIMMember(
                    value=ScimTransformations._get_scim_member_value(member),
                    display=member.user_email,
                )
            )

        team_alias = getattr(team, "team_alias", team.team_id)
        team_created_at = team.created_at.isoformat() if team.created_at else None
        team_updated_at = team.updated_at.isoformat() if team.updated_at else None

        return SCIMGroup(
            schemas=["urn:ietf:params:scim:schemas:core:2.0:Group"],
            id=team.team_id,
            displayName=team_alias,
            members=scim_members,
            meta={
                "resourceType": "Group",
                "created": team_created_at,
                "lastModified": team_updated_at,
            },
        )

    @staticmethod
    def _get_scim_member_value(member: Member) -> str:
        if member.user_email:
            return member.user_email
        return ScimTransformations.DEFAULT_SCIM_MEMBER_VALUE

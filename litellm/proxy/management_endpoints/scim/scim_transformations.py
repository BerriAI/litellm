from collections.abc import Callable
from typing import Final, TypeVar

from pydantic import ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    LiteLLM_TeamTable,
    LiteLLM_UserTable,
    Member,
    NewUserResponse,
)
from litellm.repositories.team_repository import TeamRepository
from litellm.types.proxy.management_endpoints.scim_v2 import *

T = TypeVar("T")


class ScimTransformations:
    DEFAULT_SCIM_NAME = "Unknown User"
    DEFAULT_SCIM_FAMILY_NAME = "Unknown Family Name"
    DEFAULT_SCIM_DISPLAY_NAME = "Unknown Display Name"
    DEFAULT_SCIM_MEMBER_VALUE = "Unknown Member Value"

    @staticmethod
    async def transform_litellm_user_to_scim_user(
        user: LiteLLM_UserTable | NewUserResponse,
    ) -> SCIMUser:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No database connected"})

        # Get user's teams/groups
        groups: Final[list[SCIMUserGroup]] = []
        team_ids: Final[list[str]] = user.teams or []  # mutable-ok: scim reads the user row's team ids
        for team_id in team_ids:
            team = await TeamRepository(prisma_client).table.find_unique(where={"team_id": team_id})
            if team:
                team_alias = getattr(team, "team_alias", team.team_id)
                groups.append(SCIMUserGroup(value=team.team_id, display=team_alias))

        user_created_at: Final = user.created_at.isoformat() if user.created_at else None
        user_updated_at: Final = user.updated_at.isoformat() if user.updated_at else None

        emails: Final[list[SCIMUserEmail]] = []
        # Only add email if it's a valid email address (contains @)
        # user_email can be a UUID when users are created without an email
        if user.user_email and "@" in user.user_email:
            emails.append(SCIMUserEmail(value=user.user_email, primary=True))

        metadata: Final = user.metadata or {}
        scim_active: Final = metadata.get("scim_active")
        active: Final = True if scim_active is None else bool(scim_active)

        schemas: Final = ["urn:ietf:params:scim:schemas:core:2.0:User"]
        enterprise_user: Final = ScimTransformations._parse_directory_metadata(
            user, SCIM_ENTERPRISE_METADATA_KEY, SCIMEnterpriseUser.model_validate
        )
        if enterprise_user is not None:
            schemas.append(SCIM_ENTERPRISE_USER_SCHEMA)

        entitlements: Final = ScimTransformations._parse_directory_metadata(
            user, SCIM_ENTITLEMENTS_METADATA_KEY, SCIM_MULTI_VALUED_LIST_ADAPTER.validate_python
        )
        roles: Final = ScimTransformations._parse_directory_metadata(
            user, SCIM_ROLES_METADATA_KEY, SCIM_MULTI_VALUED_LIST_ADAPTER.validate_python
        )
        addresses: Final = ScimTransformations._parse_directory_metadata(
            user, SCIM_ADDRESSES_METADATA_KEY, SCIM_ADDRESS_LIST_ADAPTER.validate_python
        )

        return SCIMUser(
            schemas=schemas,
            id=user.user_id,
            externalId=ScimTransformations._get_scim_external_id(user),
            userName=ScimTransformations._get_scim_user_name(user),
            displayName=ScimTransformations._get_scim_display_name(user),
            name=SCIMUserName(
                familyName=ScimTransformations._get_scim_family_name(user),
                givenName=ScimTransformations._get_scim_given_name(user),
                formatted=ScimTransformations._get_scim_profile_str(user, "formatted", "name.formatted"),
                middleName=ScimTransformations._get_scim_profile_str(user, "middleName"),
                honorificPrefix=ScimTransformations._get_scim_profile_str(user, "honorificPrefix"),
                honorificSuffix=ScimTransformations._get_scim_profile_str(user, "honorificSuffix"),
            ),
            title=ScimTransformations._get_scim_profile_str(user, "title", "title"),
            emails=emails,
            addresses=addresses,
            groups=groups,
            active=active,
            entitlements=entitlements,
            roles=roles,
            enterprise_user=enterprise_user,
            meta={
                "resourceType": "User",
                "created": user_created_at,
                "lastModified": user_updated_at,
            },
        )

    @staticmethod
    def _non_empty_str(value: object) -> str | None:
        return value if isinstance(value, str) and len(value) > 0 else None

    @staticmethod
    def _get_scim_metadata(
        user: LiteLLM_UserTable | NewUserResponse,
    ) -> LiteLLM_UserScimMetadata | None:
        metadata: Final = user.metadata or {}
        raw: Final = metadata.get("scim_metadata")
        if not isinstance(raw, dict):
            return None
        try:
            return LiteLLM_UserScimMetadata.model_validate(raw)
        except ValidationError:
            verbose_proxy_logger.warning(
                "Skipping malformed scim_metadata on user %s in SCIM response",
                user.user_id,
            )
            return None

    @staticmethod
    def _parse_directory_metadata(
        user: LiteLLM_UserTable | NewUserResponse,
        key: str,
        validate: Callable[[object], T],
    ) -> T | None:
        """A SCIM directory attribute parsed from user metadata, or None when absent or malformed.

        Metadata is writable outside the SCIM surface, so a malformed value on one user must not
        fail the whole directory response; the attribute is omitted and the corruption logged.
        """
        metadata: Final = user.metadata or {}
        raw: Final = metadata.get(key)
        if not raw:
            return None
        try:
            return validate(raw)
        except ValidationError:
            verbose_proxy_logger.warning(
                "Skipping malformed %s metadata on user %s in SCIM response",
                key,
                user.user_id,
            )
            return None

    @staticmethod
    def _get_scim_user_name(user: LiteLLM_UserTable | NewUserResponse) -> str:
        """
        SCIM requires a userName with length > 0.

        Prefer the stored SCIM userName (e.g. Entra objectId) and fall back to user_email.
        """
        scim_metadata: Final = ScimTransformations._get_scim_metadata(user)
        if scim_metadata is not None:
            stored_user_name: Final = ScimTransformations._non_empty_str(scim_metadata.userName)
            if stored_user_name is not None:
                return stored_user_name
        if user.user_email and len(user.user_email) > 0:
            return user.user_email
        return ScimTransformations.DEFAULT_SCIM_DISPLAY_NAME

    @staticmethod
    def _get_scim_display_name(user: LiteLLM_UserTable | NewUserResponse) -> str:
        scim_metadata: Final = ScimTransformations._get_scim_metadata(user)
        if scim_metadata is not None:
            stored_display_name: Final = ScimTransformations._non_empty_str(scim_metadata.displayName)
            if stored_display_name is not None:
                return stored_display_name
        return ScimTransformations._get_scim_user_name(user)

    @staticmethod
    def _get_scim_external_id(user: LiteLLM_UserTable | NewUserResponse) -> str | None:
        sso_user_id: Final = ScimTransformations._non_empty_str(getattr(user, "sso_user_id", None))
        if sso_user_id is not None:
            return sso_user_id
        scim_metadata: Final = ScimTransformations._get_scim_metadata(user)
        if scim_metadata is not None:
            return ScimTransformations._non_empty_str(scim_metadata.externalId)
        return None

    @staticmethod
    def _get_scim_profile_str(
        user: LiteLLM_UserTable | NewUserResponse,
        field_name: str,
        legacy_metadata_key: str | None = None,
    ) -> str | None:
        scim_metadata: Final = ScimTransformations._get_scim_metadata(user)
        if scim_metadata is not None:
            val: Final = ScimTransformations._non_empty_str(getattr(scim_metadata, field_name, None))
            if val is not None:
                return val
        if legacy_metadata_key is not None:
            metadata: Final = user.metadata or {}
            return ScimTransformations._non_empty_str(metadata.get(legacy_metadata_key))
        return None

    @staticmethod
    def _get_scim_family_name(user: LiteLLM_UserTable | NewUserResponse) -> str:
        """
        SCIM requires a family name with length > 0
        """
        scim_metadata: Final = ScimTransformations._get_scim_metadata(user)
        if scim_metadata is not None and scim_metadata.familyName and len(scim_metadata.familyName) > 0:
            return scim_metadata.familyName

        if user.user_alias and len(user.user_alias) > 0:
            return user.user_alias
        return ScimTransformations.DEFAULT_SCIM_FAMILY_NAME

    @staticmethod
    def _get_scim_given_name(user: LiteLLM_UserTable | NewUserResponse) -> str:
        """
        SCIM requires a given name with length > 0
        """
        scim_metadata: Final = ScimTransformations._get_scim_metadata(user)
        if scim_metadata is not None and scim_metadata.givenName and len(scim_metadata.givenName) > 0:
            return scim_metadata.givenName

        if user.user_alias and len(user.user_alias) > 0:
            return user.user_alias or ScimTransformations.DEFAULT_SCIM_NAME
        return ScimTransformations.DEFAULT_SCIM_NAME

    @staticmethod
    async def transform_litellm_team_to_scim_group(
        team: LiteLLM_TeamTable | dict,
    ) -> SCIMGroup:
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise HTTPException(status_code=500, detail={"error": "No database connected"})

        if isinstance(team, dict):
            team = LiteLLM_TeamTable(**team)

        # Get team members with proper display names
        scim_members: Final[list[SCIMMember]] = []
        for member in team.members_with_roles or []:
            if isinstance(member, dict):
                member = Member(**member)

            scim_members.append(
                SCIMMember(
                    value=ScimTransformations._get_scim_member_value(member),
                    display=ScimTransformations._get_scim_member_display(member),
                    type="User",
                )
            )

        team_alias: Final = getattr(team, "team_alias", team.team_id)
        team_created_at: Final = team.created_at.isoformat() if team.created_at else None
        team_updated_at: Final = team.updated_at.isoformat() if team.updated_at else None

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
        """The member's SCIM resource id, which LiteLLM serves as user_id (RFC 7643 §8.7.1)."""
        return member.user_id or ScimTransformations.DEFAULT_SCIM_MEMBER_VALUE

    @staticmethod
    def _get_scim_member_display(member: Member) -> str:
        """
        Get the SCIM member display. Use user_email if available, otherwise use user_id.
        SCIM member display should be the display name for the user.
        """
        if hasattr(member, "user_email") and member.user_email:
            return member.user_email
        elif hasattr(member, "user_id"):
            return member.user_id or ScimTransformations.DEFAULT_SCIM_MEMBER_VALUE
        return ScimTransformations.DEFAULT_SCIM_MEMBER_VALUE

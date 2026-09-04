from typing import Literal

from pydantic import Field
from typing_extensions import TypedDict

from litellm.proxy._types import KeyManagementRoutes, LitellmUserRoles
from litellm.types.utils import LiteLLMPydanticObjectBase


class LiteLLM_UpperboundKeyGenerateParams(LiteLLMPydanticObjectBase):
    """
    Set default upperbound to max budget a key called via `/key/generate` can be.

    Args:
        max_budget (Optional[float], optional): Max budget a key can be. Defaults to None.
        budget_duration (Optional[str], optional): Duration of the budget. Defaults to None.
        duration (Optional[str], optional): Duration of the key. Defaults to None.
        max_parallel_requests (Optional[int], optional): Max number of requests that can be made in parallel. Defaults to None.
        tpm_limit (Optional[int], optional): Tpm limit. Defaults to None.
        rpm_limit (Optional[int], optional): Rpm limit. Defaults to None.
    """

    max_budget: float | None = None
    budget_duration: str | None = None
    duration: str | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None


class MicrosoftGraphAPIUserGroupDirectoryObject(TypedDict, total=False):
    """Model for Microsoft Graph API directory object"""

    odata_type: str | None
    id: str | None
    deletedDateTime: str | None
    description: str | None
    displayName: str | None
    roleTemplateId: str | None


class MicrosoftGraphAPIUserGroupResponse(TypedDict, total=False):
    """Model for Microsoft Graph API user groups response"""

    odata_context: str | None
    odata_nextLink: str | None
    value: list[MicrosoftGraphAPIUserGroupDirectoryObject] | None


class MicrosoftServicePrincipalTeam(TypedDict, total=False):
    """Model for Microsoft Service Principal Team"""

    principalDisplayName: str | None
    principalId: str | None


class AccessControl_UI_AccessMode(LiteLLMPydanticObjectBase):
    """Model for Controlling UI Access Mode via SSO Groups"""

    type: Literal["restricted_sso_group"]
    restricted_sso_group: str
    sso_group_jwt_field: str


class RoleMappings(LiteLLMPydanticObjectBase):
    """
    Configuration for mapping SSO groups to LiteLLM roles.

    The system will look at the group_claim field in the SSO token to determine
    which role to assign the user based on the roles mapping.
    """

    provider: str = Field(description="SSO Provider name (e.g., 'google', 'microsoft', 'generic')")
    group_claim: str = Field(
        description="The field name in the SSO token that contains the groups array (e.g., 'groups', 'roles')"
    )
    default_role: LitellmUserRoles | None = Field(
        default=None,
        description="Default role to assign if user's groups don't match any role mappings. Must be a valid LitellmUserRoles value (e.g., 'proxy_admin', 'internal_user', 'proxy_admin_viewer')",
    )
    roles: dict[LitellmUserRoles, list[str]] = Field(
        default_factory=dict,
        description="Mapping of LiteLLM role names to arrays of SSO group names. Example: {'proxy_admin': ['group-1', 'group-2'], 'proxy_admin_viewer': ['group-3']}",
    )


class TeamMappings(LiteLLMPydanticObjectBase):
    """
    Configuration for mapping SSO JWT fields to team IDs.

    This allows configuring team_ids_jwt_field via the database instead of
    requiring config file changes and restarts.
    """

    team_ids_jwt_field: str | None = Field(
        default=None,
        description="The field name in the SSO/JWT token that contains the team IDs array (e.g., 'groups', 'teams'). Supports dot notation for nested fields.",
    )


class SSOConfig(LiteLLMPydanticObjectBase):
    """
    Configuration for SSO environment variables and settings
    """

    # Google SSO
    google_client_id: str | None = Field(
        default=None,
        description="Google OAuth Client ID for SSO authentication",
    )
    google_client_secret: str | None = Field(
        default=None,
        description="Google OAuth Client Secret for SSO authentication",
    )

    # Microsoft SSO
    microsoft_client_id: str | None = Field(
        default=None,
        description="Microsoft OAuth Client ID for SSO authentication",
    )
    microsoft_client_secret: str | None = Field(
        default=None,
        description="Microsoft OAuth Client Secret for SSO authentication",
    )
    microsoft_tenant: str | None = Field(
        default=None,
        description="Microsoft Azure Tenant ID for SSO authentication",
    )

    # Generic/Okta SSO
    generic_client_id: str | None = Field(
        default=None,
        description="Generic OAuth Client ID for SSO authentication (used for Okta and other providers)",
    )
    generic_client_secret: str | None = Field(
        default=None,
        description="Generic OAuth Client Secret for SSO authentication",
    )
    generic_authorization_endpoint: str | None = Field(
        default=None,
        description="Authorization endpoint URL for generic OAuth provider",
    )
    generic_token_endpoint: str | None = Field(
        default=None,
        description="Token endpoint URL for generic OAuth provider",
    )
    generic_userinfo_endpoint: str | None = Field(
        default=None,
        description="User info endpoint URL for generic OAuth provider",
    )
    generic_scope: str | None = Field(
        default=None,
        description="Space-separated OAuth scopes requested from the generic provider, e.g. 'openid email profile'",
    )

    # SAML SSO
    saml_idp_metadata_url: str | None = Field(
        default=None,
        description="URL of the SAML IdP metadata to fetch and parse for SSO authentication",
    )
    saml_idp_metadata_xml: str | None = Field(
        default=None,
        description="Inline SAML IdP metadata XML, used when a metadata URL is not available",
    )
    saml_sp_entity_id: str | None = Field(
        default=None,
        description="SAML Service Provider entityID; defaults to the proxy's /sso/saml/metadata URL",
    )
    saml_allow_unsolicited: str | None = Field(
        default=None,
        description="'true' to accept IdP-initiated (unsolicited) SAML responses, which cannot be browser-bound against login CSRF",
    )

    # Common settings
    proxy_base_url: str | None = Field(
        default=None,
        description="Base URL of the proxy server for SSO redirects",
    )
    user_email: str | None = Field(
        default=None,
        description="Email of the proxy admin user",
    )

    # Access Mode
    ui_access_mode: AccessControl_UI_AccessMode | str | None = Field(
        default=None,
        description="Access mode for the UI",
    )

    # Role Mappings
    role_mappings: RoleMappings | None = Field(
        default=None,
        description="Configuration for mapping SSO groups to LiteLLM roles based on group claims in the SSO token",
    )

    # Team Mappings
    team_mappings: TeamMappings | None = Field(
        default=None,
        description="Configuration for mapping SSO JWT fields to team IDs. Takes precedence over config file settings.",
    )


class DefaultTeamSSOParams(LiteLLMPydanticObjectBase):
    """
    Default parameters applied to every /team/new call for fields not explicitly provided in the request.
    `models` is the exception: it only applies to teams automatically created by LiteLLM via SSO Groups.
    """

    models: list[str] = Field(
        default=[],
        description="Default list of models for teams automatically created via SSO Groups",
    )
    max_budget: float | None = Field(
        default=None,
        description="Default maximum budget (in USD) for new teams, when not explicitly provided",
    )
    budget_duration: str | None = Field(
        default=None,
        description="Default budget duration for new teams, when not explicitly provided (e.g. '24h', '7d', '30d')",
    )
    tpm_limit: int | None = Field(
        default=None,
        description="Default tpm limit for new teams, when not explicitly provided",
    )
    rpm_limit: int | None = Field(
        default=None,
        description="Default rpm limit for new teams, when not explicitly provided",
    )
    team_member_permissions: list[KeyManagementRoutes] | None = Field(
        default=None,
        description="Default permissions granted to members of newly created teams (e.g. /key/generate, /key/update, /key/delete). /key/info and /key/health are always included.",
    )
    organization_id: str | None = Field(
        default=None,
        description="Default organization for new teams created without an explicit organization",
    )

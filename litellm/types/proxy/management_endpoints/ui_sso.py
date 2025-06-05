from typing import List, Literal, Optional, TypedDict

from pydantic import Field

from litellm.proxy._types import LiteLLMPydanticObjectBase, LitellmUserRoles


class MicrosoftGraphAPIUserGroupDirectoryObject(TypedDict, total=False):
    """Model for Microsoft Graph API directory object"""

    odata_type: Optional[str]
    id: Optional[str]
    deletedDateTime: Optional[str]
    description: Optional[str]
    displayName: Optional[str]
    roleTemplateId: Optional[str]


class MicrosoftGraphAPIUserGroupResponse(TypedDict, total=False):
    """Model for Microsoft Graph API user groups response"""

    odata_context: Optional[str]
    odata_nextLink: Optional[str]
    value: Optional[List[MicrosoftGraphAPIUserGroupDirectoryObject]]


class MicrosoftServicePrincipalTeam(TypedDict, total=False):
    """Model for Microsoft Service Principal Team"""

    principalDisplayName: Optional[str]
    principalId: Optional[str]


class SSOConfig(LiteLLMPydanticObjectBase):
    """
    Configuration for SSO environment variables and settings
    """

    # Google SSO
    google_client_id: Optional[str] = Field(
        default=None,
        description="Google OAuth Client ID for SSO authentication",
    )
    google_client_secret: Optional[str] = Field(
        default=None,
        description="Google OAuth Client Secret for SSO authentication",
    )
    
    # Microsoft SSO
    microsoft_client_id: Optional[str] = Field(
        default=None,
        description="Microsoft OAuth Client ID for SSO authentication",
    )
    microsoft_client_secret: Optional[str] = Field(
        default=None,
        description="Microsoft OAuth Client Secret for SSO authentication",
    )
    microsoft_tenant: Optional[str] = Field(
        default=None,
        description="Microsoft Azure Tenant ID for SSO authentication",
    )
    
    # Generic/Okta SSO
    generic_client_id: Optional[str] = Field(
        default=None,
        description="Generic OAuth Client ID for SSO authentication (used for Okta and other providers)",
    )
    generic_client_secret: Optional[str] = Field(
        default=None,
        description="Generic OAuth Client Secret for SSO authentication",
    )
    generic_authorization_endpoint: Optional[str] = Field(
        default=None,
        description="Authorization endpoint URL for generic OAuth provider",
    )
    generic_token_endpoint: Optional[str] = Field(
        default=None,
        description="Token endpoint URL for generic OAuth provider",
    )
    generic_userinfo_endpoint: Optional[str] = Field(
        default=None,
        description="User info endpoint URL for generic OAuth provider",
    )
    
    # Common settings
    proxy_base_url: Optional[str] = Field(
        default=None,
        description="Base URL of the proxy server for SSO redirects",
    )
    user_email: Optional[str] = Field(
        default=None,
        description="Email of the proxy admin user",
    )


class DefaultTeamSSOParams(LiteLLMPydanticObjectBase):
    """
    Default parameters to apply when a new team is automatically created by LiteLLM via SSO Groups
    """

    models: List[str] = Field(
        default=[],
        description="Default list of models that new automatically created teams can access",
    )
    max_budget: Optional[float] = Field(
        default=None,
        description="Default maximum budget (in USD) for new automatically created teams",
    )
    budget_duration: Optional[str] = Field(
        default=None,
        description="Default budget duration for new automatically created teams (e.g. 'daily', 'weekly', 'monthly')",
    )
    tpm_limit: Optional[int] = Field(
        default=None,
        description="Default tpm limit for new automatically created teams",
    )
    rpm_limit: Optional[int] = Field(
        default=None,
        description="Default rpm limit for new automatically created teams",
    )

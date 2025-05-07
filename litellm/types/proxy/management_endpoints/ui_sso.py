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

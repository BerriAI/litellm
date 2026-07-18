"""
Domain models for LiteLLM backend.
"""

from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.models.budget import (
    LiteLLM_BudgetTable,
    LiteLLM_BudgetTableFull,
    LiteLLM_TeamMemberTable,
)
from litellm.models.config import LiteLLM_Config
from litellm.models.credentials import (
    CreateCredentialItem,
    CredentialBase,
    CredentialItem,
)
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.managed_files import (
    LiteLLM_ManagedFileTable,
    LiteLLM_ManagedObjectTable,
    LiteLLM_ManagedVectorStoresTable,
    LiteLLM_ManagedVectorStoreTable,
)
from litellm.models.mcp_server import LiteLLM_MCPServerTable
from litellm.models.model import LiteLLM_ProxyModelTable
from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.models.organization_membership import LiteLLM_OrganizationMembershipTable
from litellm.models.project import LiteLLM_ProjectTable
from litellm.models.skills import LiteLLM_SkillsTable
from litellm.models.spend_logs import LiteLLM_ErrorLogs, LiteLLM_SpendLogs
from litellm.models.tag import LiteLLM_TagTable
from litellm.models.team import LiteLLM_TeamTable
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.models.user import LiteLLM_UserTable
from litellm.models.verification_token import LiteLLM_VerificationToken

__all__ = [
    "LiteLLM_AccessGroupTable",
    "LiteLLM_BudgetTable",
    "LiteLLM_BudgetTableFull",
    "LiteLLM_TeamMemberTable",
    "LiteLLM_Config",
    "CredentialBase",
    "CredentialItem",
    "CreateCredentialItem",
    "LiteLLM_EndUserTable",
    "LiteLLM_ManagedFileTable",
    "LiteLLM_ManagedObjectTable",
    "LiteLLM_ManagedVectorStoreTable",
    "LiteLLM_ManagedVectorStoresTable",
    "LiteLLM_MCPServerTable",
    "LiteLLM_ProxyModelTable",
    "LiteLLM_ObjectPermissionTable",
    "LiteLLM_OrganizationTable",
    "LiteLLM_OrganizationMembershipTable",
    "LiteLLM_ProjectTable",
    "LiteLLM_SkillsTable",
    "LiteLLM_ErrorLogs",
    "LiteLLM_SpendLogs",
    "LiteLLM_TagTable",
    "LiteLLM_TeamTable",
    "LiteLLM_TeamMembership",
    "LiteLLM_UserTable",
    "LiteLLM_VerificationToken",
]

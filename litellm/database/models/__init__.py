"""
Domain models for LiteLLM backend.
"""

from litellm.database.models.access_group import LiteLLM_AccessGroupTable
from litellm.database.models.budget import (
    LiteLLM_BudgetTable,
    LiteLLM_BudgetTableFull,
    LiteLLM_TeamMemberTable,
)
from litellm.database.models.config import LiteLLM_Config
from litellm.database.models.credentials import (
    CreateCredentialItem,
    CredentialBase,
    CredentialItem,
)
from litellm.database.models.end_user import LiteLLM_EndUserTable
from litellm.database.models.managed_files import (
    LiteLLM_ManagedFileTable,
    LiteLLM_ManagedObjectTable,
    LiteLLM_ManagedVectorStoresTable,
    LiteLLM_ManagedVectorStoreTable,
)
from litellm.database.models.mcp_server import LiteLLM_MCPServerTable
from litellm.database.models.model import LiteLLM_ProxyModelTable
from litellm.database.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.database.models.organization import LiteLLM_OrganizationTable
from litellm.database.models.organization_membership import LiteLLM_OrganizationMembershipTable
from litellm.database.models.project import LiteLLM_ProjectTable
from litellm.database.models.skills import LiteLLM_SkillsTable
from litellm.database.models.spend_logs import LiteLLM_ErrorLogs, LiteLLM_SpendLogs
from litellm.database.models.tag import LiteLLM_TagTable
from litellm.database.models.team import LiteLLM_TeamTable
from litellm.database.models.team_membership import LiteLLM_TeamMembership
from litellm.database.models.user import LiteLLM_UserTable
from litellm.database.models.verification_token import LiteLLM_VerificationToken

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

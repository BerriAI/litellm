"""
Passthrough table repositories.

Each repository centralizes access to a single Prisma table behind a ``table``
property, making the repository the one place that names the underlying table.
These are thin wrappers for tables that do not (yet) need domain-specific query
methods; richer repositories live in their own modules.
"""

from typing import TYPE_CHECKING, Any, Final, Generic

from litellm.proxy.common_utils.config_sync_pubsub import wrap_table_actions_for_config_sync
from litellm.repositories.prisma_protocols import RowT_co, TableActions

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class PrismaTableRepository(Generic[RowT_co]):
    """Base for repositories that expose a single Prisma table."""

    table_name: str

    def __init__(self, prisma_client: Any):
        self._prisma_client = prisma_client

    @property
    def prisma_client(self) -> Any:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    def table(self) -> TableActions[RowT_co]:
        actions: Final[TableActions[RowT_co]] = getattr(self.prisma_client.db, self.table_name)
        return wrap_table_actions_for_config_sync(actions=actions, table_name=self.table_name)


class PolicyRepository(PrismaTableRepository["prisma_models.LiteLLM_PolicyTable"]):
    table_name = "litellm_policytable"


class AgentsRepository(PrismaTableRepository["prisma_models.LiteLLM_AgentsTable"]):
    table_name = "litellm_agentstable"


class ObjectPermissionRepository(PrismaTableRepository["prisma_models.LiteLLM_ObjectPermissionTable"]):
    table_name = "litellm_objectpermissiontable"


class GuardrailsRepository(PrismaTableRepository["prisma_models.LiteLLM_GuardrailsTable"]):
    table_name = "litellm_guardrailstable"


class MCPServerRepository(PrismaTableRepository["prisma_models.LiteLLM_MCPServerTable"]):
    table_name = "litellm_mcpservertable"


class ManagedObjectRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedObjectTable"]):
    table_name = "litellm_managedobjecttable"


class OrganizationMembershipRepository(PrismaTableRepository["prisma_models.LiteLLM_OrganizationMembership"]):
    table_name = "litellm_organizationmembership"


class SpendLogsRepository(PrismaTableRepository["prisma_models.LiteLLM_SpendLogs"]):
    table_name = "litellm_spendlogs"


class ClaudeCodePluginRepository(PrismaTableRepository["prisma_models.LiteLLM_ClaudeCodePluginTable"]):
    table_name = "litellm_claudecodeplugintable"


class TeamMembershipRepository(PrismaTableRepository["prisma_models.LiteLLM_TeamMembership"]):
    table_name = "litellm_teammembership"


class EndUserRepository(PrismaTableRepository["prisma_models.LiteLLM_EndUserTable"]):
    table_name = "litellm_endusertable"


class ManagedVectorStoresRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedVectorStoresTable"]):
    table_name = "litellm_managedvectorstorestable"


class MCPUserCredentialsRepository(PrismaTableRepository["prisma_models.LiteLLM_MCPUserCredentials"]):
    table_name = "litellm_mcpusercredentials"


class MCPServerOAuthClientRepository(PrismaTableRepository["prisma_models.LiteLLM_MCPServerOAuthClient"]):
    table_name = "litellm_mcpserveroauthclient"


class PromptRepository(PrismaTableRepository["prisma_models.LiteLLM_PromptTable"]):
    table_name = "litellm_prompttable"


class TagRepository(PrismaTableRepository["prisma_models.LiteLLM_TagTable"]):
    table_name = "litellm_tagtable"


class InvitationLinkRepository(PrismaTableRepository["prisma_models.LiteLLM_InvitationLink"]):
    table_name = "litellm_invitationlink"


class JWTKeyMappingRepository(PrismaTableRepository["prisma_models.LiteLLM_JWTKeyMapping"]):
    table_name = "litellm_jwtkeymapping"


class ManagedFileRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedFileTable"]):
    table_name = "litellm_managedfiletable"


class MemoryRepository(PrismaTableRepository["prisma_models.LiteLLM_MemoryTable"]):
    table_name = "litellm_memorytable"


class SearchToolsRepository(PrismaTableRepository["prisma_models.LiteLLM_SearchToolsTable"]):
    table_name = "litellm_searchtoolstable"


class ConfigOverridesRepository(PrismaTableRepository["prisma_models.LiteLLM_ConfigOverrides"]):
    table_name = "litellm_configoverrides"


class MCPToolsetRepository(PrismaTableRepository["prisma_models.LiteLLM_MCPToolsetTable"]):
    table_name = "litellm_mcptoolsettable"


class ToolRepository(PrismaTableRepository["prisma_models.LiteLLM_ToolTable"]):
    table_name = "litellm_tooltable"


class DeletedVerificationTokenRepository(PrismaTableRepository["prisma_models.LiteLLM_DeletedVerificationToken"]):
    table_name = "litellm_deletedverificationtoken"


class WorkflowRunRepository(PrismaTableRepository["prisma_models.LiteLLM_WorkflowRun"]):
    table_name = "litellm_workflowrun"


class ModelTableRepository(PrismaTableRepository["prisma_models.LiteLLM_ModelTable"]):
    table_name = "litellm_modeltable"


class AccessGroupRepository(PrismaTableRepository["prisma_models.LiteLLM_AccessGroupTable"]):
    table_name = "litellm_accessgrouptable"


class SSOConfigRepository(PrismaTableRepository["prisma_models.LiteLLM_SSOConfig"]):
    table_name = "litellm_ssoconfig"


class UISettingsRepository(PrismaTableRepository["prisma_models.LiteLLM_UISettings"]):
    table_name = "litellm_uisettings"


class DailyGuardrailMetricsRepository(PrismaTableRepository["prisma_models.LiteLLM_DailyGuardrailMetrics"]):
    table_name = "litellm_dailyguardrailmetrics"


class DailyGuardrailUsageUnitsRepository(PrismaTableRepository["prisma_models.LiteLLM_DailyGuardrailUsageUnits"]):
    table_name = "litellm_dailyguardrailusageunits"


class PolicyAttachmentRepository(PrismaTableRepository["prisma_models.LiteLLM_PolicyAttachmentTable"]):
    table_name = "litellm_policyattachmenttable"


class DeletedTeamRepository(PrismaTableRepository["prisma_models.LiteLLM_DeletedTeamTable"]):
    table_name = "litellm_deletedteamtable"


class SkillsRepository(PrismaTableRepository["prisma_models.LiteLLM_SkillsTable"]):
    table_name = "litellm_skillstable"


class CacheConfigRepository(PrismaTableRepository["prisma_models.LiteLLM_CacheConfig"]):
    table_name = "litellm_cacheconfig"


class ManagedVectorStoreIndexRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedVectorStoreIndexTable"]):
    table_name = "litellm_managedvectorstoreindextable"


class WorkflowMessageRepository(PrismaTableRepository["prisma_models.LiteLLM_WorkflowMessage"]):
    table_name = "litellm_workflowmessage"


class DailyTagSpendRepository(PrismaTableRepository["prisma_models.LiteLLM_DailyTagSpend"]):
    table_name = "litellm_dailytagspend"


class SpendLogToolIndexRepository(PrismaTableRepository["prisma_models.LiteLLM_SpendLogToolIndex"]):
    table_name = "litellm_spendlogtoolindex"


class DailyToolSpendRepository(PrismaTableRepository["prisma_models.LiteLLM_DailyToolSpend"]):
    table_name = "litellm_dailytoolspend"


class SpendLogGuardrailIndexRepository(PrismaTableRepository["prisma_models.LiteLLM_SpendLogGuardrailIndex"]):
    table_name = "litellm_spendlogguardrailindex"


class UserNotificationsRepository(PrismaTableRepository["prisma_models.LiteLLM_UserNotifications"]):
    table_name = "litellm_usernotifications"


class HealthCheckRepository(PrismaTableRepository["prisma_models.LiteLLM_HealthCheckTable"]):
    table_name = "litellm_healthchecktable"


class DeprecatedVerificationTokenRepository(PrismaTableRepository["prisma_models.LiteLLM_DeprecatedVerificationToken"]):
    table_name = "litellm_deprecatedverificationtoken"


class WorkflowEventRepository(PrismaTableRepository["prisma_models.LiteLLM_WorkflowEvent"]):
    table_name = "litellm_workflowevent"


class DailyPolicyMetricsRepository(PrismaTableRepository["prisma_models.LiteLLM_DailyPolicyMetrics"]):
    table_name = "litellm_dailypolicymetrics"


class AdaptiveRouterStateRepository(PrismaTableRepository["prisma_models.LiteLLM_AdaptiveRouterState"]):
    table_name = "litellm_adaptiverouterstate"


class AuditLogRepository(PrismaTableRepository["prisma_models.LiteLLM_AuditLog"]):
    table_name = "litellm_auditlog"


class AdaptiveRouterSessionRepository(PrismaTableRepository["prisma_models.LiteLLM_AdaptiveRouterSession"]):
    table_name = "litellm_adaptiveroutersession"

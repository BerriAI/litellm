"""
Passthrough table repositories.

Each repository centralizes access to a single Prisma table behind a ``table``
property, making the repository the one place that names the underlying table.
These are thin wrappers for tables that do not (yet) need domain-specific query
methods; richer repositories live in their own modules.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final, Generic, NamedTuple, Protocol

from litellm.litellm_core_utils.core_helpers import require_attributes
from litellm.repositories import rust_database
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.prisma_protocols import RowT_co, TableActions
from litellm.rust_bridge._native import RustAgentIdentityRepository, RustAuditLogRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class _PrismaReaderClientView(Protocol):
    @property
    def db(self) -> object: ...


class _PrismaWriterClientView(Protocol):
    @property
    def writer_db(self) -> object: ...


class PrismaTableRepository(Generic[RowT_co]):
    """Base for repositories that expose a single Prisma table."""

    table_name: str

    def __init__(self, prisma_client: object, *, use_writer: bool = False) -> None:
        self._prisma_client = prisma_client
        self._use_writer = use_writer

    def _connected_client(self) -> object:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    def prisma_client(self) -> object:
        return self._connected_client()

    def _selected_database(self) -> object:
        client: Final = self._connected_client()
        if self._use_writer:
            return require_attributes(client, _PrismaWriterClientView, "writer_db").writer_db
        return require_attributes(client, _PrismaReaderClientView, "db").db

    @property
    def table(self) -> TableActions[RowT_co]:
        from litellm.proxy.common_utils.config_sync_pubsub import wrap_table_actions_for_config_sync

        actions: Final[TableActions[RowT_co]] = getattr(self._selected_database(), self.table_name)
        return wrap_table_actions_for_config_sync(actions=actions, table_name=self.table_name)


class CustomerOrganizationRepository(PrismaTableRepository["prisma_models.LiteLLM_CustomerOrganizationTable"]):
    table_name = "litellm_customerorganizationtable"


class CustomerMemberRepository(PrismaTableRepository["prisma_models.LiteLLM_CustomerMemberTable"]):
    table_name = "litellm_customermembertable"


class CustomerTeamRepository(PrismaTableRepository["prisma_models.LiteLLM_CustomerTeamTable"]):
    table_name = "litellm_customerteamtable"


class CustomerBudgetIncreaseRequestRepository(
    PrismaTableRepository["prisma_models.LiteLLM_CustomerBudgetIncreaseRequest"]
):
    table_name = "litellm_customerbudgetincreaserequest"


class PolicyRepository(PrismaTableRepository["prisma_models.LiteLLM_PolicyTable"]):
    table_name = "litellm_policytable"


class AgentsRepository(PrismaTableRepository["prisma_models.LiteLLM_AgentsTable"]):
    table_name = "litellm_agentstable"


class AgentIdentityRepository(PrismaTableRepository["prisma_models.LiteLLM_AgentIdentity"]):
    table_name = "litellm_agentidentity"

    async def record_authentication(self, agent_id: str, revision: str) -> bool:
        """Stamps the identity as authenticated now. False when it is no longer the active one at ``revision``, or
        its agent is disabled or no longer identity managed."""
        return await rust_database.required_repository(RustAgentIdentityRepository).record_authentication(
            agent_id, revision
        )


class RetiredAgentIdentityRepository(PrismaTableRepository["prisma_models.LiteLLM_RetiredAgentIdentity"]):
    table_name = "litellm_retiredagentidentity"


class VerifiedSubjectRepository(PrismaTableRepository["prisma_models.LiteLLM_VerifiedSubject"]):
    table_name = "litellm_verifiedsubject"


class ObjectPermissionRepository(PrismaTableRepository["prisma_models.LiteLLM_ObjectPermissionTable"]):
    table_name = "litellm_objectpermissiontable"


class GuardrailsRepository(PrismaTableRepository["prisma_models.LiteLLM_GuardrailsTable"]):
    table_name = "litellm_guardrailstable"

    async def find_active_guardrail_names(self, names: frozenset[str]) -> frozenset[str]:
        rows: Final = await find_many_in(self.table, "guardrail_name", sorted(names), where={"status": "active"})
        return frozenset(row.guardrail_name for row in rows)


class MCPServerRepository(PrismaTableRepository["prisma_models.LiteLLM_MCPServerTable"]):
    table_name = "litellm_mcpservertable"


class ManagedObjectRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedObjectTable"]):
    table_name = "litellm_managedobjecttable"


class OrganizationMembershipRepository(PrismaTableRepository["prisma_models.LiteLLM_OrganizationMembership"]):
    table_name = "litellm_organizationmembership"


class ClaudeCodePluginRepository(PrismaTableRepository["prisma_models.LiteLLM_ClaudeCodePluginTable"]):
    table_name = "litellm_claudecodeplugintable"


class ManagedVectorStoresRepository(PrismaTableRepository["prisma_models.LiteLLM_ManagedVectorStoresTable"]):
    table_name = "litellm_managedvectorstorestable"


class MCPUserCredentialsRepository(PrismaTableRepository["prisma_models.LiteLLM_MCPUserCredentials"]):
    table_name = "litellm_mcpusercredentials"


class MCPServerOAuthClientRepository(PrismaTableRepository["prisma_models.LiteLLM_MCPServerOAuthClient"]):
    table_name = "litellm_mcpserveroauthclient"


class PromptRepository(PrismaTableRepository["prisma_models.LiteLLM_PromptTable"]):
    table_name = "litellm_prompttable"


class InvitationLinkRepository(PrismaTableRepository["prisma_models.LiteLLM_InvitationLink"]):
    table_name = "litellm_invitationlink"


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


class TeamRepository(PrismaTableRepository["prisma_models.LiteLLM_TeamTable"]):
    table_name = "litellm_teamtable"


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


class DailyModelUsageRepository(PrismaTableRepository["prisma_models.LiteLLM_DailyModelUsage"]):
    table_name = "litellm_dailymodelusage"


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


class NewAuditLog(NamedTuple):
    """One audit log row to insert. ``before_value`` and ``updated_values`` are JSON text. A ``changed_by`` or
    ``changed_by_api_key`` of None is stored as empty, and an ``updated_at`` without a zone is read as UTC."""

    id: str
    updated_at: datetime
    changed_by: str | None
    changed_by_api_key: str | None
    action: str
    table_name: str
    object_id: str
    before_value: str | None
    updated_values: str | None


_EPOCH: Final = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _audit_log_row(row: NewAuditLog) -> tuple[str, int, str | None, str | None, str, str, str, str | None, str | None]:
    updated_at: Final = (
        row.updated_at if row.updated_at.tzinfo is not None else row.updated_at.replace(tzinfo=timezone.utc)
    )
    return (
        row.id,
        (updated_at - _EPOCH) // timedelta(microseconds=1),
        row.changed_by,
        row.changed_by_api_key,
        row.action,
        row.table_name,
        row.object_id,
        row.before_value,
        row.updated_values,
    )


class AuditLogRepository(PrismaTableRepository["prisma_models.LiteLLM_AuditLog"]):
    table_name = "litellm_auditlog"

    async def insert(self, row: NewAuditLog) -> None:
        await rust_database.required_repository(RustAuditLogRepository).insert(_audit_log_row(row))

    async def insert_many(self, rows: Sequence[NewAuditLog]) -> None:
        """Inserts every row in one statement, so either all of them are stored or none is."""
        await rust_database.required_repository(RustAuditLogRepository).insert_many(
            tuple(_audit_log_row(row) for row in rows)
        )


class AdaptiveRouterSessionRepository(PrismaTableRepository["prisma_models.LiteLLM_AdaptiveRouterSession"]):
    table_name = "litellm_adaptiveroutersession"


class BackgroundInteractionSettlementRepository(
    PrismaTableRepository["prisma_models.LiteLLM_BackgroundInteractionSettlement"]
):
    table_name = "litellm_backgroundinteractionsettlement"


class BackgroundResponseSettlementRepository(
    PrismaTableRepository["prisma_models.LiteLLM_BackgroundResponseSettlement"]
):
    table_name = "litellm_backgroundresponsesettlement"


class BackgroundResponseCreatorRepository(PrismaTableRepository["prisma_models.LiteLLM_BackgroundResponseCreator"]):
    table_name = "litellm_backgroundresponsecreator"


class RetiredAgentRepository(PrismaTableRepository["prisma_models.LiteLLM_RetiredAgent"]):
    table_name = "litellm_retiredagent"


class DailyUserSpendRepository(PrismaTableRepository["prisma_models.LiteLLM_DailyUserSpend"]):
    table_name = "litellm_dailyuserspend"

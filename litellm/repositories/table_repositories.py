"""
Passthrough table repositories.

Each repository centralizes access to a single Prisma table behind a ``table``
property, making the repository the one place that names the underlying table.
These are thin wrappers for tables that do not (yet) need domain-specific query
methods; richer repositories live in their own modules.
"""

from collections.abc import Mapping, Sequence
from datetime import timedelta
from itertools import groupby
from typing import TYPE_CHECKING, Any, Final, Generic, Protocol

from pydantic import TypeAdapter

from litellm.litellm_core_utils.core_helpers import require_attributes
from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.repositories.accounting_delivery_repository import LedgerDatabase, LedgerTransaction
from litellm.repositories.prisma_protocols import RowT_co, TableActions

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
    def prisma_client(self) -> Any:
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


class PolicyRepository(PrismaTableRepository["prisma_models.LiteLLM_PolicyTable"]):
    table_name = "litellm_policytable"


class AgentsRepository(PrismaTableRepository["prisma_models.LiteLLM_AgentsTable"]):
    table_name = "litellm_agentstable"


class AgentIdentityRepository(PrismaTableRepository["prisma_models.LiteLLM_AgentIdentity"]):
    table_name = "litellm_agentidentity"


class RetiredAgentIdentityRepository(PrismaTableRepository["prisma_models.LiteLLM_RetiredAgentIdentity"]):
    table_name = "litellm_retiredagentidentity"


class VerifiedSubjectRepository(PrismaTableRepository["prisma_models.LiteLLM_VerifiedSubject"]):
    table_name = "litellm_verifiedsubject"


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


_SPEND_LOG_FIELDS: Final = TypeAdapter(dict[str, object])


def _accounting_payload(value: object) -> Mapping[str, object]:
    return (
        _SPEND_LOG_FIELDS.validate_json(value) if isinstance(value, str) else _SPEND_LOG_FIELDS.validate_python(value)
    )


_ACCOUNTING_ONLY_FIELDS: Final = frozenset({"accounting_payload", "accounting_log_disabled", "accounting_delivery_id"})


def _settles_once(row: Mapping[str, object]) -> bool:
    """A batch or video cost row: update_database gives every other ledger row its own delivery id."""
    return "accounting_delivery_id" not in row


def _delivery_id(row: Mapping[str, object]) -> str:
    return str(row["accounting_delivery_id"]) if not _settles_once(row) else str(row["request_id"])


class SpendLogsRepository(PrismaTableRepository["prisma_models.LiteLLM_SpendLogs"]):
    table_name = "litellm_spendlogs"

    async def insert_with_deliveries(
        self, rows: Sequence[Mapping[str, object]], *, write_spend_logs: bool = True
    ) -> int:
        legacy: Final = tuple(row for row in rows if "accounting_payload" not in row)
        legacy_count: Final = await self.table.create_many(data=legacy, skip_duplicates=True) if legacy else 0
        ledger: Final = tuple(
            sorted(
                {_delivery_id(row): row for row in reversed(rows) if "accounting_payload" in row}.values(),
                key=_delivery_id,
            )
        )
        if not ledger:
            return legacy_count
        database: Final[LedgerDatabase] = self.prisma_client.writer_db
        async with database.tx(timeout=timedelta(seconds=60)) as transaction:
            accepted: Final = await self._insert_ledger(transaction, ledger, write_spend_logs=write_spend_logs)
        return legacy_count + accepted

    @staticmethod
    async def _insert_ledger(
        transaction: LedgerTransaction, rows: Sequence[Mapping[str, object]], *, write_spend_logs: bool
    ) -> int:
        from litellm.proxy.db.db_span import db_span
        from litellm.repositories.spend_log_insert import spend_log_insert_statement

        payloads: Final = tuple(
            {
                "delivery_id": _delivery_id(row),
                "request_id": row["request_id"],
                "accounting_payload": _accounting_payload(row["accounting_payload"]),
                "is_batch": _settles_once(row),
            }
            for row in rows
        )
        inserted_deliveries: Final = await transaction.query_raw(
            'INSERT INTO "LiteLLM_SpendLogDelivery" (delivery_id, request_id, accounting_payload, is_batch) '
            "SELECT delivery_id, request_id, accounting_payload, is_batch FROM jsonb_to_recordset($1::jsonb) "
            "AS inputs(delivery_id text, request_id text, accounting_payload jsonb, is_batch boolean) "
            "ON CONFLICT (delivery_id) DO NOTHING RETURNING delivery_id",
            safe_dumps(payloads, non_finite_as_null=True),
        )
        accepted_ids: Final = frozenset(str(row["delivery_id"]) for row in inserted_deliveries)
        charged_by_legacy: Final = await SpendLogsRepository._jobs_charged_by_legacy(
            transaction,
            tuple(_delivery_id(row) for row in rows if _settles_once(row) and _delivery_id(row) in accepted_ids),
        )
        if charged_by_legacy:
            await transaction.execute_raw(
                'DELETE FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id = ANY($1::text[])', list(charged_by_legacy)
            )
        accepted: Final = accepted_ids - charged_by_legacy
        local_rows: Final = tuple(
            {key: value for key, value in row.items() if key not in _ACCOUNTING_ONLY_FIELDS}
            for row in rows
            if _delivery_id(row) in accepted and row.get("accounting_log_disabled") is not True
        )
        if write_spend_logs and local_rows:
            groups: Final = tuple(
                tuple(group)
                for _, group in groupby(
                    sorted(local_rows, key=lambda row: tuple(sorted(row))), key=lambda row: tuple(sorted(row))
                )
            )
            async with db_span("insert_spend_logs", "LiteLLM_SpendLogs"):
                for group in groups:
                    await transaction.query_raw(*spend_log_insert_statement(group))
        return len(accepted)

    @staticmethod
    async def _jobs_charged_by_legacy(transaction: LedgerTransaction, cost_row_ids: Sequence[str]) -> frozenset[str]:
        """Batches and videos a pre-ledger poll already charged through their SpendLogs cost row."""
        if not cost_row_ids:
            return frozenset()
        rows: Final = await transaction.query_raw(
            'SELECT DISTINCT request_id FROM "LiteLLM_SpendLogs" WHERE request_id = ANY($1::text[]) '
            "AND status = 'success' AND spend > 0",
            list(cost_row_ids),
        )
        return frozenset(str(row["request_id"]) for row in rows)

    async def has_delivery(self, delivery_id: str) -> bool:
        database: Final[LedgerTransaction] = self.prisma_client.writer_db
        rows: Final = await database.query_raw(
            'SELECT 1 FROM "LiteLLM_SpendLogDelivery" WHERE delivery_id = $1', delivery_id
        )
        return bool(rows)


class BudgetWindowSpendRepository(PrismaTableRepository["prisma_models.LiteLLM_BudgetWindowSpend"]):
    table_name = "litellm_budgetwindowspend"


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


class ModelAccessGroupBudgetRepository(PrismaTableRepository["prisma_models.LiteLLM_ModelAccessGroupBudgetTable"]):
    table_name = "litellm_modelaccessgroupbudgettable"


class InvitationLinkRepository(PrismaTableRepository["prisma_models.LiteLLM_InvitationLink"]):
    table_name = "litellm_invitationlink"


class JWTKeyMappingRepository(PrismaTableRepository["prisma_models.LiteLLM_JWTKeyMapping"]):
    table_name = "litellm_jwtkeymapping"


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

    async def find_by_name(self, name: str) -> LiteLLM_AccessGroupTable | None:
        row: Final = await self.table.find_unique(where={"access_group_name": name})
        return LiteLLM_AccessGroupTable.model_validate(row, from_attributes=True) if row is not None else None


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


class AuditLogRepository(PrismaTableRepository["prisma_models.LiteLLM_AuditLog"]):
    table_name = "litellm_auditlog"


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


class RetiredAgentRepository(PrismaTableRepository["prisma_models.LiteLLM_RetiredAgent"]):
    table_name = "litellm_retiredagent"


class DailyUserSpendRepository(PrismaTableRepository["prisma_models.LiteLLM_DailyUserSpend"]):
    table_name = "litellm_dailyuserspend"

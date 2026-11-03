"""
Passthrough table repositories.

Each repository centralizes access to a single Prisma table behind a ``table``
property, making the repository the one place that names the underlying table.
These are thin wrappers for tables that do not (yet) need domain-specific query
methods; richer repositories live in their own modules.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from itertools import chain, groupby
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
)

from pydantic import TypeAdapter

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.repositories.accounting_delivery_repository import LedgerDatabase, LedgerTransaction
from litellm.repositories.prisma_protocols import RowT_co, TableActions

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class PrismaTableRepository(Generic[RowT_co]):
    """Base for repositories that expose a single Prisma table."""

    table_name: str

    def __init__(self, prisma_client: object, *, use_writer: bool = False) -> None:
        self._prisma_client = prisma_client
        self._use_writer = use_writer

    @property
    def prisma_client(self) -> Any:
        if self._prisma_client is None:
            raise RuntimeError("No DB Connected. See - https://docs.litellm.ai/docs/proxy/virtual_keys")
        return self._prisma_client

    @property
    def table(self) -> TableActions[RowT_co]:
        from litellm.proxy.common_utils.config_sync_pubsub import wrap_table_actions_for_config_sync

        actions: Final[TableActions[RowT_co]] = getattr(
            self.prisma_client.writer_db if self._use_writer else self.prisma_client.db, self.table_name
        )
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


@dataclass(frozen=True, slots=True)
class BatchTakeover:
    charged: bool
    accounting_path: Literal["legacy", "ledger"]


_SPEND_LOG_ROWS: Final = TypeAdapter(tuple[dict[str, object], ...])
_SPEND_LOG_FIELDS: Final = TypeAdapter(dict[str, object])


def _accounting_payload(value: object) -> Mapping[str, object]:
    return (
        _SPEND_LOG_FIELDS.validate_json(value) if isinstance(value, str) else _SPEND_LOG_FIELDS.validate_python(value)
    )


def _charges_a_batch(row: Mapping[str, object]) -> bool:
    spend: Final = row.get("spend")
    return row.get("call_type") == "aretrieve_batch" and isinstance(spend, (int, float)) and spend > 0


class SpendLogsRepository(PrismaTableRepository["prisma_models.LiteLLM_SpendLogs"]):
    table_name = "litellm_spendlogs"

    async def insert_with_deliveries(
        self, rows: Sequence[Mapping[str, object]], *, write_spend_logs: bool = True
    ) -> int:
        legacy: Final = tuple(row for row in rows if row.get("accounting_path") != "ledger")
        legacy_count: Final = await self.table.create_many(data=legacy, skip_duplicates=True) if legacy else 0
        ledger: Final = tuple(
            sorted(
                {
                    str(row["request_id"]): row for row in reversed(rows) if row.get("accounting_path") == "ledger"
                }.values(),
                key=lambda row: str(row["request_id"]),
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
                "request_id": row["request_id"],
                "accounting_payload": _accounting_payload(row["accounting_payload"]),
                "is_batch": row.get("call_type") == "aretrieve_batch",
            }
            for row in rows
        )
        inserted_deliveries: Final = await transaction.query_raw(
            'INSERT INTO "LiteLLM_SpendLogDelivery" (request_id, accounting_payload, is_batch) '
            "SELECT request_id, accounting_payload, is_batch FROM jsonb_to_recordset($1::jsonb) "
            "AS inputs(request_id text, accounting_payload jsonb, is_batch boolean) "
            "ON CONFLICT (request_id) DO NOTHING RETURNING request_id",
            safe_dumps(payloads, non_finite_as_null=True),
        )
        accepted: Final = frozenset(str(row["request_id"]) for row in inserted_deliveries)
        if not accepted or not write_spend_logs:
            return len(accepted)
        local_rows: Final = tuple(
            {key: value for key, value in row.items() if key not in {"accounting_payload", "accounting_log_disabled"}}
            for row in rows
            if row["request_id"] in accepted and row.get("accounting_log_disabled") is not True
        )
        groups: Final = tuple(
            tuple(group)
            for _, group in groupby(
                sorted(local_rows, key=lambda row: tuple(sorted(row))), key=lambda row: tuple(sorted(row))
            )
        )
        async with db_span("insert_spend_logs", "LiteLLM_SpendLogs"):
            inserted_groups: Final = tuple(
                [await transaction.query_raw(*spend_log_insert_statement(group)) for group in groups]
            )
        inserted: Final = frozenset(str(row["request_id"]) for row in chain.from_iterable(inserted_groups))
        taken_over: Final = await SpendLogsRepository._take_over_uncharged_batches(
            transaction, tuple(row for row in local_rows if row["request_id"] not in inserted)
        )
        conflicts: Final = tuple(
            str(row["request_id"])
            for row in local_rows
            if row["request_id"] not in inserted and row["request_id"] not in taken_over
        )
        if conflicts:
            await transaction.execute_raw(
                'DELETE FROM "LiteLLM_SpendLogDelivery" WHERE request_id = ANY($1::text[])', conflicts
            )
        return len(accepted) - len(conflicts)

    @staticmethod
    async def _take_over_uncharged_batches(
        transaction: LedgerTransaction, conflicting: Sequence[Mapping[str, object]]
    ) -> frozenset[str]:
        from litellm.repositories.spend_log_insert import spend_log_takeover_statement

        updated: Final = tuple(
            [
                await transaction.query_raw(*spend_log_takeover_statement(row))
                for row in conflicting
                if _charges_a_batch(row)
            ]
        )
        return frozenset(str(row["request_id"]) for row in chain.from_iterable(updated))

    async def batch_delivery(self, request_id: str) -> Mapping[str, object] | None:
        database: Final[LedgerTransaction] = self.prisma_client.writer_db
        rows: Final = await database.query_raw(
            'SELECT accounting_payload FROM "LiteLLM_SpendLogDelivery" WHERE request_id = $1', request_id
        )
        return _SPEND_LOG_FIELDS.validate_python(rows[0]["accounting_payload"]) if rows else None

    async def take_over_batch(self, row: Mapping[str, object]) -> BatchTakeover:
        from litellm.proxy.db.db_span import db_span
        from litellm.repositories.spend_log_insert import spend_log_takeover_statement

        database: Final[LedgerDatabase] = self.prisma_client.writer_db
        async with database.tx(timeout=timedelta(seconds=60)) as transaction:
            receipts: Final = await transaction.query_raw(
                'SELECT state::text, accounting_payload FROM "LiteLLM_SpendLogDelivery" WHERE request_id = $1 FOR UPDATE',
                row["request_id"],
            )
            if not receipts:
                return BatchTakeover(False, "legacy")
            old: Final = _SPEND_LOG_FIELDS.validate_python(receipts[0]["accounting_payload"])
            new: Final = _accounting_payload(row["accounting_payload"])
            if old.get("call_type") != "aretrieve_batch" or old.get("status") != "success":
                return BatchTakeover(False, "ledger")
            spend: Final = new.get("spend")
            if old.get("spend") != 0 or not isinstance(spend, (int, float)) or spend <= 0:
                return BatchTakeover(False, "ledger")
            local_row: Final = {
                key: value for key, value in row.items() if key not in {"accounting_payload", "accounting_log_disabled"}
            }
            async with db_span("update", "LiteLLM_SpendLogs"):
                await transaction.query_raw(*spend_log_takeover_statement(local_row))
            await transaction.execute_raw(
                'UPDATE "LiteLLM_SpendLogDelivery" SET accounting_payload = $2::jsonb, '
                "previous_payload = CASE WHEN state = 'applied' THEN accounting_payload ELSE previous_payload END, "
                "state = CASE WHEN state = 'applied' THEN 'pending'::\"AccountingDeliveryState\" ELSE state END, "
                "applied_at = CASE WHEN state = 'applied' THEN NULL ELSE applied_at END "
                "WHERE request_id = $1",
                row["request_id"],
                safe_dumps(new, non_finite_as_null=True),
            )
            return BatchTakeover(True, "ledger")


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


class RetiredAgentRepository(PrismaTableRepository["prisma_models.LiteLLM_RetiredAgent"]):
    table_name = "litellm_retiredagent"

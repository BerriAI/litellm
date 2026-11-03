"""
Passthrough table repositories.

Each repository centralizes access to a single Prisma table behind a ``table``
property, making the repository the one place that names the underlying table.
These are thin wrappers for tables that do not (yet) need domain-specific query
methods; richer repositories live in their own modules.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import groupby
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Generic,
    Literal,
    Protocol,
    cast,  # noqa: TID251  # validated partition identifiers cross the LiteralString SQL boundary
)

from pydantic import TypeAdapter
from typing_extensions import LiteralString

from litellm.litellm_core_utils.safe_json_dumps import safe_dumps
from litellm.repositories.accounting_delivery_repository import LedgerDatabase
from litellm.repositories.prisma_protocols import RowT_co, TableActions

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # used by quoted base-class subscripts


class _SpendLogInsertDatabase(Protocol):
    async def query_raw(self, query: LiteralString, *args: object) -> Sequence[Mapping[str, object]]: ...


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


class SpendLogCleanupTransaction(Protocol):
    async def query_raw(self, query: LiteralString, *args: object) -> Sequence[Mapping[str, object]]: ...

    async def execute_raw(self, query: LiteralString, *args: object) -> int: ...


@dataclass(frozen=True, slots=True)
class SpendLogDeletion:
    deleted: int
    postponed: bool


class SpendLogsRepository(PrismaTableRepository["prisma_models.LiteLLM_SpendLogs"]):
    table_name = "litellm_spendlogs"

    @staticmethod
    async def delete_settled_before(
        transaction: SpendLogCleanupTransaction, cutoff: datetime, batch_size: int
    ) -> SpendLogDeletion:
        candidates: Final = await transaction.query_raw(
            'SELECT request_id, "startTime" FROM "LiteLLM_SpendLogs" logs '
            'WHERE "startTime" < $1::timestamp AND NOT EXISTS ('
            'SELECT 1 FROM "LiteLLM_AccountingDelivery" delivery WHERE delivery.request_id = logs.request_id '
            "AND delivery.state IN ('pending', 'parked')) "
            'ORDER BY "startTime", request_id LIMIT $2::integer FOR UPDATE OF logs SKIP LOCKED',
            cutoff,
            batch_size,
        )
        serialized: Final = safe_dumps(
            tuple(
                {
                    "request_id": row["request_id"],
                    "startTime": value.isoformat() if isinstance(value := row["startTime"], datetime) else value,
                }
                for row in candidates
            )
        )
        deleted: Final = await transaction.execute_raw(
            'DELETE FROM "LiteLLM_SpendLogs" logs USING jsonb_to_recordset($1::jsonb) '
            'AS candidate(request_id text, "startTime" timestamp) '
            'WHERE logs.request_id = candidate.request_id AND logs."startTime" = candidate."startTime" '
            'AND NOT EXISTS (SELECT 1 FROM "LiteLLM_AccountingDelivery" delivery '
            "WHERE delivery.request_id = logs.request_id AND delivery.state IN ('pending', 'parked'))",
            serialized,
        )
        protected: Final = await transaction.query_raw(
            'SELECT EXISTS(SELECT 1 FROM "LiteLLM_SpendLogs" logs JOIN "LiteLLM_AccountingDelivery" delivery '
            'ON delivery.request_id = logs.request_id WHERE logs."startTime" < $1::timestamp '
            "AND delivery.state IN ('pending', 'parked')) AS protected",
            cutoff,
        )
        return SpendLogDeletion(deleted, bool(protected and protected[0]["protected"]))

    @staticmethod
    async def drop_settled_partition(transaction: SpendLogCleanupTransaction, name: str) -> bool:
        if re.fullmatch(r"LiteLLM_SpendLogs_p[0-9]{8}", name) is None:
            raise ValueError("Unexpected SpendLogs partition name")
        partition: Final = cast(  # cast-ok: the strict partition-name allowlist above excludes SQL syntax
            LiteralString, name
        )
        await transaction.execute_raw(f'LOCK TABLE "{partition}" IN ACCESS EXCLUSIVE MODE')
        protected: Final = await transaction.query_raw(
            f'SELECT EXISTS(SELECT 1 FROM "{partition}" logs JOIN "LiteLLM_AccountingDelivery" delivery '
            "ON delivery.request_id = logs.request_id "
            "WHERE delivery.state IN ('pending', 'parked')) AS protected"
        )
        if protected and protected[0]["protected"]:
            return False
        await transaction.execute_raw(f'DROP TABLE "{partition}"')
        return True

    async def insert_with_deliveries(self, rows: Sequence[Mapping[str, object]]) -> int:
        from litellm.proxy.db.db_span import db_span
        from litellm.repositories.spend_log_insert import spend_log_insert_statement

        if not any(row.get("accounting_path") == "ledger" for row in rows):
            return await self.table.create_many(data=rows, skip_duplicates=True)
        database: Final[_SpendLogInsertDatabase] = self.prisma_client.writer_db
        async with db_span("insert_spend_logs", "LiteLLM_SpendLogs"):
            counts: Final = tuple(
                [
                    len(await database.query_raw(*spend_log_insert_statement(tuple(group))))
                    for _, group in groupby(
                        sorted(rows, key=lambda row: tuple(sorted(row))), key=lambda row: tuple(sorted(row))
                    )
                ]
            )
        return sum(counts)

    async def take_over_batch(self, row: Mapping[str, object]) -> BatchTakeover:
        from litellm.proxy.db.db_span import db_span
        from litellm.repositories.spend_log_insert import spend_log_takeover_statement

        database: Final[LedgerDatabase] = self.prisma_client.writer_db
        async with database.tx(timeout=timedelta(seconds=60)) as transaction:
            existing: Final = _SPEND_LOG_ROWS.validate_python(
                await transaction.query_raw(
                    'SELECT to_jsonb(stored) AS row FROM "LiteLLM_SpendLogs" stored WHERE request_id = $1 FOR UPDATE',
                    row["request_id"],
                )
            )
            if not existing:
                return BatchTakeover(False, "legacy")
            old: Final = _SPEND_LOG_FIELDS.validate_python(existing[0]["row"])
            path: Final = "ledger" if old.get("accounting_path") == "ledger" else "legacy"
            receipts: Final = (
                _SPEND_LOG_ROWS.validate_python(
                    await transaction.query_raw(
                        'SELECT state::text FROM "LiteLLM_AccountingDelivery" WHERE request_id = $1 FOR UPDATE',
                        row["request_id"],
                    )
                )
                if path == "ledger"
                else ()
            )
            async with db_span("update", "LiteLLM_SpendLogs"):
                changed: Final = await transaction.query_raw(*spend_log_takeover_statement(row))
            if not changed:
                return BatchTakeover(False, path)
            if path == "ledger" and (not receipts or receipts[0]["state"] == "applied"):
                await transaction.execute_raw(
                    'INSERT INTO "LiteLLM_AccountingDelivery" (request_id, previous_row) VALUES ($1, $2::jsonb) '
                    "ON CONFLICT (request_id) DO UPDATE SET state = 'pending', previous_row = EXCLUDED.previous_row, "
                    "attempts = 0, next_attempt_at = CURRENT_TIMESTAMP, applied_at = NULL, last_error = NULL",
                    row["request_id"],
                    safe_dumps(old, non_finite_as_null=True),
                )
            return BatchTakeover(True, path)


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

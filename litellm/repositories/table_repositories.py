"""
Passthrough table repositories.

Each repository centralizes access to a single Prisma table behind a ``table``
property, making the repository the one place that names the underlying table.
These are thin wrappers for tables that do not (yet) need domain-specific query
methods; richer repositories live in their own modules.
"""

from collections.abc import Mapping
from typing import Any, Final, Generic, TypeVar

from pydantic import BaseModel, TypeAdapter

from litellm.models.access_group import LiteLLM_AccessGroupTable
from litellm.models.end_user import LiteLLM_EndUserTable
from litellm.models.tag import LiteLLM_TagTable
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.proxy.common_utils.config_sync_pubsub import wrap_table_actions_for_config_sync
from litellm.repositories.prisma_protocols import PrismaTableActions

ModelT = TypeVar("ModelT", bound=BaseModel)

_TOKEN_ADAPTER: Final = TypeAdapter(str)


class PrismaTableRepository:
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
    def table(self) -> Any:  # any-ok: Prisma table actions are reached through the untyped client wrapper
        return wrap_table_actions_for_config_sync(
            actions=getattr(self.prisma_client.db, self.table_name),
            table_name=self.table_name,
        )

    @property
    def typed_table(self) -> PrismaTableActions:  # any-ok: Prisma table actions are untyped at runtime
        """The repository table, narrowed to the action surface the repositories use."""
        return self.table


class ModelBackedTable(Generic[ModelT]):
    """Mixin that validates rows of a prisma table into a pydantic model before handing them out."""

    model_class: type[ModelT]

    @property
    def typed_table(self) -> PrismaTableActions:
        raise NotImplementedError

    async def find_unique_model(
        self, *, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> ModelT | None:
        """Load a single row by a unique constraint and validate it into the domain model."""
        record: Final = await self.typed_table.find_unique(where=where, include=include)
        return None if record is None else self.model_class.model_validate(record.dict())

    async def find_first_model(
        self, *, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> ModelT | None:
        """Load the first row matching the filter and validate it into the domain model."""
        record: Final = await self.typed_table.find_first(where=where, include=include)
        return None if record is None else self.model_class.model_validate(record.dict())

    async def find_many_models(
        self, *, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> tuple[ModelT, ...]:
        """Load every row matching the filter and validate them into domain models."""
        records: Final = await self.typed_table.find_many(where=where, include=include)
        return tuple(self.model_class.model_validate(record.dict()) for record in records)


class PolicyRepository(PrismaTableRepository):
    table_name = "litellm_policytable"


class AgentsRepository(PrismaTableRepository):
    table_name = "litellm_agentstable"


class ObjectPermissionRepository(PrismaTableRepository):
    table_name = "litellm_objectpermissiontable"


class GuardrailsRepository(PrismaTableRepository):
    table_name = "litellm_guardrailstable"


class MCPServerRepository(PrismaTableRepository):
    table_name = "litellm_mcpservertable"


class ManagedObjectRepository(PrismaTableRepository):
    table_name = "litellm_managedobjecttable"


class OrganizationMembershipRepository(PrismaTableRepository):
    table_name = "litellm_organizationmembership"


class SpendLogsRepository(PrismaTableRepository):
    table_name = "litellm_spendlogs"


class ClaudeCodePluginRepository(PrismaTableRepository):
    table_name = "litellm_claudecodeplugintable"


class TeamMembershipRepository(PrismaTableRepository, ModelBackedTable[LiteLLM_TeamMembership]):
    table_name = "litellm_teammembership"
    model_class = LiteLLM_TeamMembership


class EndUserRepository(PrismaTableRepository, ModelBackedTable[LiteLLM_EndUserTable]):
    table_name = "litellm_endusertable"
    model_class = LiteLLM_EndUserTable


class ManagedVectorStoresRepository(PrismaTableRepository):
    table_name = "litellm_managedvectorstorestable"


class MCPUserCredentialsRepository(PrismaTableRepository):
    table_name = "litellm_mcpusercredentials"


class MCPServerOAuthClientRepository(PrismaTableRepository):
    table_name = "litellm_mcpserveroauthclient"


class PromptRepository(PrismaTableRepository):
    table_name = "litellm_prompttable"


class TagRepository(PrismaTableRepository, ModelBackedTable[LiteLLM_TagTable]):
    table_name = "litellm_tagtable"
    model_class = LiteLLM_TagTable


class InvitationLinkRepository(PrismaTableRepository):
    table_name = "litellm_invitationlink"


class JWTKeyMappingRepository(PrismaTableRepository):
    table_name = "litellm_jwtkeymapping"

    async def find_active_token(self, *, jwt_claim_name: str, jwt_claim_value: str) -> str | None:
        """Return the key token mapped to an active JWT claim pair."""
        record: Final = await self.typed_table.find_first(
            where={
                "jwt_claim_name": jwt_claim_name,
                "jwt_claim_value": jwt_claim_value,
                "is_active": True,
            }
        )
        return None if record is None else _TOKEN_ADAPTER.validate_python(record.dict()["token"])


class ManagedFileRepository(PrismaTableRepository):
    table_name = "litellm_managedfiletable"


class MemoryRepository(PrismaTableRepository):
    table_name = "litellm_memorytable"


class SearchToolsRepository(PrismaTableRepository):
    table_name = "litellm_searchtoolstable"


class ConfigOverridesRepository(PrismaTableRepository):
    table_name = "litellm_configoverrides"


class MCPToolsetRepository(PrismaTableRepository):
    table_name = "litellm_mcptoolsettable"


class ToolRepository(PrismaTableRepository):
    table_name = "litellm_tooltable"


class DeletedVerificationTokenRepository(PrismaTableRepository):
    table_name = "litellm_deletedverificationtoken"


class WorkflowRunRepository(PrismaTableRepository):
    table_name = "litellm_workflowrun"


class ModelTableRepository(PrismaTableRepository):
    table_name = "litellm_modeltable"


class AccessGroupRepository(PrismaTableRepository, ModelBackedTable[LiteLLM_AccessGroupTable]):
    table_name = "litellm_accessgrouptable"
    model_class = LiteLLM_AccessGroupTable


class SSOConfigRepository(PrismaTableRepository):
    table_name = "litellm_ssoconfig"


class UISettingsRepository(PrismaTableRepository):
    table_name = "litellm_uisettings"


class DailyGuardrailMetricsRepository(PrismaTableRepository):
    table_name = "litellm_dailyguardrailmetrics"


class PolicyAttachmentRepository(PrismaTableRepository):
    table_name = "litellm_policyattachmenttable"


class DeletedTeamRepository(PrismaTableRepository):
    table_name = "litellm_deletedteamtable"


class SkillsRepository(PrismaTableRepository):
    table_name = "litellm_skillstable"


class CacheConfigRepository(PrismaTableRepository):
    table_name = "litellm_cacheconfig"


class ManagedVectorStoreIndexRepository(PrismaTableRepository):
    table_name = "litellm_managedvectorstoreindextable"


class WorkflowMessageRepository(PrismaTableRepository):
    table_name = "litellm_workflowmessage"


class DailyTagSpendRepository(PrismaTableRepository):
    table_name = "litellm_dailytagspend"


class SpendLogToolIndexRepository(PrismaTableRepository):
    table_name = "litellm_spendlogtoolindex"


class DailyToolSpendRepository(PrismaTableRepository):
    table_name = "litellm_dailytoolspend"


class SpendLogGuardrailIndexRepository(PrismaTableRepository):
    table_name = "litellm_spendlogguardrailindex"


class UserNotificationsRepository(PrismaTableRepository):
    table_name = "litellm_usernotifications"


class HealthCheckRepository(PrismaTableRepository):
    table_name = "litellm_healthchecktable"


class DeprecatedVerificationTokenRepository(PrismaTableRepository):
    table_name = "litellm_deprecatedverificationtoken"


class WorkflowEventRepository(PrismaTableRepository):
    table_name = "litellm_workflowevent"


class DailyPolicyMetricsRepository(PrismaTableRepository):
    table_name = "litellm_dailypolicymetrics"


class AdaptiveRouterStateRepository(PrismaTableRepository):
    table_name = "litellm_adaptiverouterstate"


class AuditLogRepository(PrismaTableRepository):
    table_name = "litellm_auditlog"


class AdaptiveRouterSessionRepository(PrismaTableRepository):
    table_name = "litellm_adaptiveroutersession"

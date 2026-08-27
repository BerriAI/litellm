"""
Team repository for database operations on LiteLLM_TeamTable.
"""

import json
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Final

from pydantic import TypeAdapter

from litellm.models.team import LiteLLM_TeamTable, Member
from litellm.repositories.base_repository import (
    BaseRepository,
    DbRecord,
    record_to_dict,
)
from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models

_MEMBERS_WITH_ROLES_ADAPTER: Final = TypeAdapter(list[Member])
_JSON_ENCODED_TEAM_FIELDS: Final = (
    "metadata",
    "model_spend",
    "model_max_budget",
    "router_settings",
    "budget_limits",
    "members_with_roles",
)


class TeamRepository(BaseRepository[LiteLLM_TeamTable]):
    """Repository for team database operations."""

    @property
    def table(self) -> TableActions["prisma_models.LiteLLM_TeamTable"]:
        return self.prisma_client.db.litellm_teamtable

    @property
    def deleted_table(self) -> TableActions["prisma_models.LiteLLM_DeletedTeamTable"]:
        return self.prisma_client.db.litellm_deletedteamtable

    @property
    def model_class(self) -> type[LiteLLM_TeamTable]:
        return LiteLLM_TeamTable

    def _to_model(self, record: DbRecord | None) -> LiteLLM_TeamTable | None:
        """Convert a database record to a Team model."""
        if record is None:
            return None

        data: Final = {
            field: json.loads(value) if field in _JSON_ENCODED_TEAM_FIELDS and isinstance(value, str) else value
            for field, value in record_to_dict(record).items()
        }

        return LiteLLM_TeamTable.model_validate(data)

    async def get_members_with_roles_locked(self, tx: "Prisma", team_id: str) -> list[Member] | None:
        """Return the team's members_with_roles. The caller must already hold
        ``TEAM_ADVISORY_LOCK_SQL`` for this team_id on ``tx`` before calling this.

        ``None`` when the team row is gone, which is only possible under that lock if
        a delete committed before this read, as opposed to ``[]`` for a team that
        simply has no members.

        A plain read is enough here because the advisory lock, not a row lock, is what
        serializes this against a concurrent writer: ``SELECT ... FOR UPDATE`` would
        additionally take a row lock on ``LiteLLM_TeamTable``, and the access-group
        endpoints lock an access group and then a team row, so a team-row-first lock
        here can deadlock with them. The advisory lock cannot, since those endpoints
        never take it.
        """
        rows: Final = await tx.query_raw(
            'SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = $1',
            team_id,
        )
        if not rows:
            return None
        raw_value: Final = rows[0]["members_with_roles"]
        parsed: Final = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        if not parsed:
            return []
        return _MEMBERS_WITH_ROLES_ADAPTER.validate_python(parsed)

    async def find_by_id(self, team_id: str, id_field: str = "team_id") -> LiteLLM_TeamTable | None:
        return await super().find_by_id(team_id, id_field)

    async def find_by_alias(self, team_alias: str) -> LiteLLM_TeamTable | None:
        """Find a team by alias."""
        records: Final = await self.table.find_many(where={"team_alias": team_alias})
        if records:
            return self._to_model(records[0])
        return None

    async def find_by_organization_id(self, organization_id: str) -> list[LiteLLM_TeamTable]:
        """Find all teams belonging to an organization."""
        records: Final = await self.table.find_many(where={"organization_id": organization_id})
        return self._to_model_list(records)

    async def find_by_member(self, user_id: str) -> list[LiteLLM_TeamTable]:
        """Find all teams where user is a member."""
        records: Final = await self.table.find_many(where={"members": {"has": user_id}})
        return self._to_model_list(records)

    async def find_by_admin(self, user_id: str) -> list[LiteLLM_TeamTable]:
        """Find all teams where user is an admin."""
        records: Final = await self.table.find_many(where={"admins": {"has": user_id}})
        return self._to_model_list(records)

    async def create_team(
        self,
        team_id: str,
        team_alias: str | None = None,
        organization_id: str | None = None,
        admins: list[str] | None = None,
        members: list[str] | None = None,
        members_with_roles: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
        max_budget: float | None = None,
        soft_budget: float | None = None,
        models: list[str] | None = None,
        max_parallel_requests: int | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        budget_duration: str | None = None,
        object_permission_id: str | None = None,
    ) -> LiteLLM_TeamTable:
        """Create a new team."""
        data: Final[dict[str, object]] = {"team_id": team_id}
        if team_alias is not None:
            data["team_alias"] = team_alias
        if organization_id is not None:
            data["organization_id"] = organization_id
        if admins is not None:
            data["admins"] = admins
        if members is not None:
            data["members"] = members
        if members_with_roles is not None:
            data["members_with_roles"] = json.dumps(members_with_roles)
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        if max_budget is not None:
            data["max_budget"] = max_budget
        if soft_budget is not None:
            data["soft_budget"] = soft_budget
        if models is not None:
            data["models"] = models
        if max_parallel_requests is not None:
            data["max_parallel_requests"] = max_parallel_requests
        if tpm_limit is not None:
            data["tpm_limit"] = tpm_limit
        if rpm_limit is not None:
            data["rpm_limit"] = rpm_limit
        if budget_duration is not None:
            data["budget_duration"] = budget_duration
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.create(data)

    async def update_team(
        self,
        team_id: str,
        team_alias: str | None = None,
        organization_id: str | None = None,
        admins: list[str] | None = None,
        members: list[str] | None = None,
        members_with_roles: Mapping[str, object] | None = None,
        metadata: Mapping[str, object] | None = None,
        max_budget: float | None = None,
        soft_budget: float | None = None,
        models: list[str] | None = None,
        max_parallel_requests: int | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        budget_duration: str | None = None,
        blocked: bool | None = None,
        object_permission_id: str | None = None,
    ) -> LiteLLM_TeamTable | None:
        """Update a team."""
        data: Final[dict[str, object]] = {}
        if team_alias is not None:
            data["team_alias"] = team_alias
        if organization_id is not None:
            data["organization_id"] = organization_id
        if admins is not None:
            data["admins"] = admins
        if members is not None:
            data["members"] = members
        if members_with_roles is not None:
            data["members_with_roles"] = json.dumps(members_with_roles)
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        if max_budget is not None:
            data["max_budget"] = max_budget
        if soft_budget is not None:
            data["soft_budget"] = soft_budget
        if models is not None:
            data["models"] = models
        if max_parallel_requests is not None:
            data["max_parallel_requests"] = max_parallel_requests
        if tpm_limit is not None:
            data["tpm_limit"] = tpm_limit
        if rpm_limit is not None:
            data["rpm_limit"] = rpm_limit
        if budget_duration is not None:
            data["budget_duration"] = budget_duration
        if blocked is not None:
            data["blocked"] = blocked
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.update(team_id, data, id_field="team_id")

    async def delete_team(
        self,
        team_id: str,
        deleted_by: str | None = None,
        deleted_by_api_key: str | None = None,
        litellm_changed_by: str | None = None,
    ) -> LiteLLM_TeamTable | None:
        """Delete a team and archive it to the deleted teams table.

        Uses a transaction to ensure atomicity of the archive-then-delete operation.
        """
        team: Final = await self.find_by_id(team_id)
        if team is None:
            return None

        archive_data: Final = self._build_archive_data(team)
        archive_data["deleted_by"] = deleted_by
        archive_data["deleted_by_api_key"] = deleted_by_api_key
        archive_data["litellm_changed_by"] = litellm_changed_by
        archive_data["deleted_at"] = datetime.utcnow()

        async with self.prisma_client.db.tx() as tx:
            await tx.litellm_deletedteamtable.create(data=archive_data)
            await tx.litellm_teamtable.delete(where={"team_id": team_id})

        return team

    def _build_archive_data(self, team: LiteLLM_TeamTable) -> dict[str, object]:
        """Build archive data dict with only columns that exist in LiteLLM_DeletedTeamTable."""
        data: Final[dict[str, object]] = {"team_id": team.team_id}
        if team.team_alias is not None:
            data["team_alias"] = team.team_alias
        if team.organization_id is not None:
            data["organization_id"] = team.organization_id
        if team.object_permission_id is not None:
            data["object_permission_id"] = team.object_permission_id
        data["admins"] = team.admins
        data["members"] = team.members
        if team.members_with_roles:
            data["members_with_roles"] = json.dumps([m.model_dump() for m in team.members_with_roles])
        if team.metadata:
            data["metadata"] = json.dumps(team.metadata)
        if team.max_budget is not None:
            data["max_budget"] = team.max_budget
        if team.soft_budget is not None:
            data["soft_budget"] = team.soft_budget
        data["spend"] = team.spend if team.spend is not None else 0.0
        data["models"] = team.models
        if team.max_parallel_requests is not None:
            data["max_parallel_requests"] = team.max_parallel_requests
        if team.tpm_limit is not None:
            data["tpm_limit"] = team.tpm_limit
        if team.rpm_limit is not None:
            data["rpm_limit"] = team.rpm_limit
        if team.budget_duration is not None:
            data["budget_duration"] = team.budget_duration
        if team.budget_reset_at is not None:
            data["budget_reset_at"] = team.budget_reset_at
        data["blocked"] = team.blocked
        if team.model_spend:
            data["model_spend"] = json.dumps(team.model_spend)
        if team.model_max_budget:
            data["model_max_budget"] = json.dumps(team.model_max_budget)
        if team.router_settings is not None:
            data["router_settings"] = json.dumps(team.router_settings)
        data["team_member_permissions"] = team.team_member_permissions or []
        data["access_group_ids"] = team.access_group_ids or []
        data["policies"] = team.policies or []
        if team.model_id is not None:
            data["model_id"] = team.model_id
        data["allow_team_guardrail_config"] = team.allow_team_guardrail_config
        return data

    async def update_spend(self, team_id: str, spend: float) -> LiteLLM_TeamTable | None:
        """Update team spend."""
        return await self.update(team_id, {"spend": spend}, id_field="team_id")

    async def add_member(self, team_id: str, user_id: str) -> LiteLLM_TeamTable | None:
        """Add a member to a team using atomic array push operation."""
        if not await self.exists(team_id, id_field="team_id"):
            return None

        record: Final = await self.table.update(
            where={"team_id": team_id},
            data={"members": {"push": user_id}},
        )
        return self._to_model(record)

    async def remove_member(self, team_id: str, user_id: str) -> LiteLLM_TeamTable | None:
        """Remove a member from a team.

        Note: Prisma doesn't support atomic array removal, so we use a
        read-modify-write pattern here. For high-concurrency scenarios,
        consider using raw SQL with array_remove().
        """
        team: Final = await self.find_by_id(team_id)
        if team is None:
            return None

        members: Final = [m for m in team.members if m != user_id]
        return await self.update(team_id, {"members": members}, id_field="team_id")

    async def add_admin(self, team_id: str, user_id: str) -> LiteLLM_TeamTable | None:
        """Add an admin to a team using atomic array push operation."""
        if not await self.exists(team_id, id_field="team_id"):
            return None

        record: Final = await self.table.update(
            where={"team_id": team_id},
            data={"admins": {"push": user_id}},
        )
        return self._to_model(record)

    async def remove_admin(self, team_id: str, user_id: str) -> LiteLLM_TeamTable | None:
        """Remove an admin from a team.

        Note: Prisma doesn't support atomic array removal, so we use a
        read-modify-write pattern here. For high-concurrency scenarios,
        consider using raw SQL with array_remove().
        """
        team: Final = await self.find_by_id(team_id)
        if team is None:
            return None

        admins: Final = [a for a in team.admins if a != user_id]
        return await self.update(team_id, {"admins": admins}, id_field="team_id")

    async def add_models(self, team_id: str, models: list[str]) -> LiteLLM_TeamTable | None:
        """Add models to a team's allowed models list using atomic array push."""
        if not await self.exists(team_id, id_field="team_id"):
            return None

        record: Final = await self.table.update(
            where={"team_id": team_id},
            data={"models": {"push": models}},
        )
        return self._to_model(record)

    async def remove_models(self, team_id: str, models: list[str]) -> LiteLLM_TeamTable | None:
        """Remove models from a team's allowed models list.

        Note: Prisma doesn't support atomic array removal, so we use a
        read-modify-write pattern here. For high-concurrency scenarios,
        consider using raw SQL with array_remove().
        """
        team: Final = await self.find_by_id(team_id)
        if team is None:
            return None

        current_models: Final = [m for m in team.models if m not in models]
        return await self.update(team_id, {"models": current_models}, id_field="team_id")

"""
Team repository for database operations on LiteLLM_TeamTable.
"""

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

from pydantic import TypeAdapter

from litellm.models.team import LiteLLM_TeamTable, Member
from litellm.repositories.base_repository import BaseRepository

if TYPE_CHECKING:
    from prisma import Prisma

_MEMBERS_WITH_ROLES_ADAPTER = TypeAdapter(list[Member])


class TeamRepository(BaseRepository[LiteLLM_TeamTable]):
    """Repository for team database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_teamtable

    @property
    def deleted_table(self) -> Any:
        return self.prisma_client.db.litellm_deletedteamtable

    @property
    def model_class(self) -> Type[LiteLLM_TeamTable]:
        return LiteLLM_TeamTable

    def _to_model(self, record: Any) -> Optional[LiteLLM_TeamTable]:
        """Convert a database record to a Team model."""
        if record is None:
            return None

        data = record.dict() if hasattr(record, "dict") else dict(record)

        json_fields = [
            "metadata",
            "model_spend",
            "model_max_budget",
            "router_settings",
            "budget_limits",
            "members_with_roles",
        ]
        for field in json_fields:
            if isinstance(data.get(field), str):
                data[field] = json.loads(data[field])

        return LiteLLM_TeamTable(**data)

    async def get_members_with_roles_locked(self, tx: "Prisma", team_id: str) -> List[Member]:
        """Return the team's members_with_roles, locking the row FOR UPDATE.

        Must be called inside a transaction so the row lock is held until
        commit. This serializes concurrent membership writers on the team row
        so the losing writer appends onto the winner's committed result instead
        of overwriting it from a stale snapshot.
        """
        rows = await tx.query_raw(
            'SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = $1 FOR UPDATE',
            team_id,
        )
        raw_value = rows[0]["members_with_roles"] if rows else None
        parsed = json.loads(raw_value) if isinstance(raw_value, str) else raw_value
        if not parsed:
            return []
        return _MEMBERS_WITH_ROLES_ADAPTER.validate_python(parsed)

    async def find_by_id(self, team_id: str, id_field: str = "team_id") -> Optional[LiteLLM_TeamTable]:
        return await super().find_by_id(team_id, id_field)

    async def find_by_alias(self, team_alias: str) -> Optional[LiteLLM_TeamTable]:
        """Find a team by alias."""
        records = await self.table.find_many(where={"team_alias": team_alias})
        if records:
            return self._to_model(records[0])
        return None

    async def find_by_organization_id(self, organization_id: str) -> List[LiteLLM_TeamTable]:
        """Find all teams belonging to an organization."""
        records = await self.table.find_many(where={"organization_id": organization_id})
        return self._to_model_list(records)

    async def find_by_member(self, user_id: str) -> List[LiteLLM_TeamTable]:
        """Find all teams where user is a member."""
        records = await self.table.find_many(where={"members": {"has": user_id}})
        return self._to_model_list(records)

    async def find_by_admin(self, user_id: str) -> List[LiteLLM_TeamTable]:
        """Find all teams where user is an admin."""
        records = await self.table.find_many(where={"admins": {"has": user_id}})
        return self._to_model_list(records)

    async def create_team(
        self,
        team_id: str,
        team_alias: Optional[str] = None,
        organization_id: Optional[str] = None,
        admins: Optional[List[str]] = None,
        members: Optional[List[str]] = None,
        members_with_roles: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_budget: Optional[float] = None,
        soft_budget: Optional[float] = None,
        models: Optional[List[str]] = None,
        max_parallel_requests: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        budget_duration: Optional[str] = None,
        object_permission_id: Optional[str] = None,
    ) -> LiteLLM_TeamTable:
        """Create a new team."""
        data: Dict[str, Any] = {"team_id": team_id}
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
        team_alias: Optional[str] = None,
        organization_id: Optional[str] = None,
        admins: Optional[List[str]] = None,
        members: Optional[List[str]] = None,
        members_with_roles: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        max_budget: Optional[float] = None,
        soft_budget: Optional[float] = None,
        models: Optional[List[str]] = None,
        max_parallel_requests: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        budget_duration: Optional[str] = None,
        blocked: Optional[bool] = None,
        object_permission_id: Optional[str] = None,
    ) -> Optional[LiteLLM_TeamTable]:
        """Update a team."""
        data: Dict[str, Any] = {}
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
        deleted_by: Optional[str] = None,
        deleted_by_api_key: Optional[str] = None,
        litellm_changed_by: Optional[str] = None,
    ) -> Optional[LiteLLM_TeamTable]:
        """Delete a team and archive it to the deleted teams table.

        Uses a transaction to ensure atomicity of the archive-then-delete operation.
        """
        team = await self.find_by_id(team_id)
        if team is None:
            return None

        archive_data = self._build_archive_data(team)
        archive_data["deleted_by"] = deleted_by
        archive_data["deleted_by_api_key"] = deleted_by_api_key
        archive_data["litellm_changed_by"] = litellm_changed_by
        archive_data["deleted_at"] = datetime.utcnow()

        async with self.prisma_client.db.tx() as tx:
            await tx.litellm_deletedteamtable.create(data=archive_data)
            await tx.litellm_teamtable.delete(where={"team_id": team_id})

        return team

    def _build_archive_data(self, team: LiteLLM_TeamTable) -> Dict[str, Any]:
        """Build archive data dict with only columns that exist in LiteLLM_DeletedTeamTable."""
        data: Dict[str, Any] = {"team_id": team.team_id}
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

    async def update_spend(self, team_id: str, spend: float) -> Optional[LiteLLM_TeamTable]:
        """Update team spend."""
        return await self.update(team_id, {"spend": spend}, id_field="team_id")

    async def add_member(self, team_id: str, user_id: str) -> Optional[LiteLLM_TeamTable]:
        """Add a member to a team using atomic array push operation."""
        if not await self.exists(team_id, id_field="team_id"):
            return None

        record = await self.table.update(
            where={"team_id": team_id},
            data={"members": {"push": user_id}},
        )
        return self._to_model(record)

    async def remove_member(self, team_id: str, user_id: str) -> Optional[LiteLLM_TeamTable]:
        """Remove a member from a team.

        Note: Prisma doesn't support atomic array removal, so we use a
        read-modify-write pattern here. For high-concurrency scenarios,
        consider using raw SQL with array_remove().
        """
        team = await self.find_by_id(team_id)
        if team is None:
            return None

        members = [m for m in team.members if m != user_id]
        return await self.update(team_id, {"members": members}, id_field="team_id")

    async def add_admin(self, team_id: str, user_id: str) -> Optional[LiteLLM_TeamTable]:
        """Add an admin to a team using atomic array push operation."""
        if not await self.exists(team_id, id_field="team_id"):
            return None

        record = await self.table.update(
            where={"team_id": team_id},
            data={"admins": {"push": user_id}},
        )
        return self._to_model(record)

    async def remove_admin(self, team_id: str, user_id: str) -> Optional[LiteLLM_TeamTable]:
        """Remove an admin from a team.

        Note: Prisma doesn't support atomic array removal, so we use a
        read-modify-write pattern here. For high-concurrency scenarios,
        consider using raw SQL with array_remove().
        """
        team = await self.find_by_id(team_id)
        if team is None:
            return None

        admins = [a for a in team.admins if a != user_id]
        return await self.update(team_id, {"admins": admins}, id_field="team_id")

    async def add_models(self, team_id: str, models: List[str]) -> Optional[LiteLLM_TeamTable]:
        """Add models to a team's allowed models list using atomic array push."""
        if not await self.exists(team_id, id_field="team_id"):
            return None

        record = await self.table.update(
            where={"team_id": team_id},
            data={"models": {"push": models}},
        )
        return self._to_model(record)

    async def remove_models(self, team_id: str, models: List[str]) -> Optional[LiteLLM_TeamTable]:
        """Remove models from a team's allowed models list.

        Note: Prisma doesn't support atomic array removal, so we use a
        read-modify-write pattern here. For high-concurrency scenarios,
        consider using raw SQL with array_remove().
        """
        team = await self.find_by_id(team_id)
        if team is None:
            return None

        current_models = [m for m in team.models if m not in models]
        return await self.update(team_id, {"models": current_models}, id_field="team_id")

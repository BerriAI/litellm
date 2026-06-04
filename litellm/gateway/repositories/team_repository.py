"""
Team repository for database operations on LiteLLM_TeamTable.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from litellm.backend.models.team import Team
from litellm.gateway.repositories.base_repository import BaseRepository


class TeamRepository(BaseRepository[Team]):
    """Repository for team database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_teamtable

    @property
    def deleted_table(self) -> Any:
        return self.prisma_client.db.litellm_deletedteamtable

    @property
    def model_class(self) -> Type[Team]:
        return Team

    def _to_model(self, record: Any) -> Optional[Team]:
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

        return Team(**data)

    async def find_by_id(
        self, team_id: str, id_field: str = "team_id"
    ) -> Optional[Team]:
        return await super().find_by_id(team_id, id_field)

    async def find_by_alias(self, team_alias: str) -> Optional[Team]:
        """Find a team by alias."""
        records = await self.table.find_many(where={"team_alias": team_alias})
        if records:
            return self._to_model(records[0])
        return None

    async def find_by_organization_id(self, organization_id: str) -> List[Team]:
        """Find all teams belonging to an organization."""
        records = await self.table.find_many(where={"organization_id": organization_id})
        return self._to_model_list(records)

    async def find_by_member(self, user_id: str) -> List[Team]:
        """Find all teams where user is a member."""
        records = await self.table.find_many(where={"members": {"has": user_id}})
        return self._to_model_list(records)

    async def find_by_admin(self, user_id: str) -> List[Team]:
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
    ) -> Team:
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
    ) -> Optional[Team]:
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
    ) -> Optional[Team]:
        """Delete a team and archive it to the deleted teams table.

        Uses a transaction to ensure atomicity of the archive-then-delete operation.
        """
        team = await self.find_by_id(team_id)
        if team is None:
            return None

        team_data = team.to_db_dict()
        team_data["deleted_by"] = deleted_by
        team_data["deleted_by_api_key"] = deleted_by_api_key
        team_data["litellm_changed_by"] = litellm_changed_by
        team_data["deleted_at"] = datetime.utcnow()

        async with self.prisma_client.db.tx() as tx:
            await tx.litellm_deletedteamtable.create(data=team_data)
            await tx.litellm_teamtable.delete(where={"team_id": team_id})

        return team

    async def update_spend(self, team_id: str, spend: float) -> Optional[Team]:
        """Update team spend."""
        return await self.update(team_id, {"spend": spend}, id_field="team_id")

    async def add_member(self, team_id: str, user_id: str) -> Optional[Team]:
        """Add a member to a team using atomic array push operation."""
        if not await self.exists(team_id, id_field="team_id"):
            return None

        record = await self.table.update(
            where={"team_id": team_id},
            data={"members": {"push": user_id}},
        )
        return self._to_model(record)

    async def remove_member(self, team_id: str, user_id: str) -> Optional[Team]:
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

    async def add_admin(self, team_id: str, user_id: str) -> Optional[Team]:
        """Add an admin to a team using atomic array push operation."""
        if not await self.exists(team_id, id_field="team_id"):
            return None

        record = await self.table.update(
            where={"team_id": team_id},
            data={"admins": {"push": user_id}},
        )
        return self._to_model(record)

    async def remove_admin(self, team_id: str, user_id: str) -> Optional[Team]:
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

    async def add_models(self, team_id: str, models: List[str]) -> Optional[Team]:
        """Add models to a team's allowed models list using atomic array push."""
        if not await self.exists(team_id, id_field="team_id"):
            return None

        record = await self.table.update(
            where={"team_id": team_id},
            data={"models": {"push": models}},
        )
        return self._to_model(record)

    async def remove_models(self, team_id: str, models: List[str]) -> Optional[Team]:
        """Remove models from a team's allowed models list.

        Note: Prisma doesn't support atomic array removal, so we use a
        read-modify-write pattern here. For high-concurrency scenarios,
        consider using raw SQL with array_remove().
        """
        team = await self.find_by_id(team_id)
        if team is None:
            return None

        current_models = [m for m in team.models if m not in models]
        return await self.update(
            team_id, {"models": current_models}, id_field="team_id"
        )

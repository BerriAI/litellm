"""
User repository for database operations on LiteLLM_UserTable.
"""

import json
from typing import Any, Dict, List, Mapping, Optional, Type

from litellm.models.user import LiteLLM_UserTable
from litellm.repositories.base_repository import BaseRepository, DbRecord, record_to_dict

_JSON_ENCODED_COLUMNS = frozenset({"metadata", "model_spend", "model_max_budget"})


class UserRepository(BaseRepository[LiteLLM_UserTable]):
    """Repository for user database operations."""

    @property
    def table(self) -> Any:  # any-ok: Prisma table actions are reached through the untyped client wrapper
        return self.prisma_client.db.litellm_usertable

    @property
    def model_class(self) -> Type[LiteLLM_UserTable]:
        return LiteLLM_UserTable

    def _to_model(self, record: Optional[DbRecord]) -> Optional[LiteLLM_UserTable]:
        """Convert a database record to a User model."""
        if record is None:
            return None

        return LiteLLM_UserTable.model_validate(
            {
                column: json.loads(value) if column in _JSON_ENCODED_COLUMNS and isinstance(value, str) else value
                for column, value in record_to_dict(record).items()
            }
        )

    async def find_by_id(self, id_value: str, id_field: str = "user_id") -> Optional[LiteLLM_UserTable]:
        return await super().find_by_id(id_value, id_field)

    async def find_by_email(self, user_email: str) -> Optional[LiteLLM_UserTable]:
        """Find a user by email."""
        records = await self.find_many(where={"user_email": user_email})
        return records[0] if records else None

    async def find_by_sso_id(self, sso_user_id: str) -> Optional[LiteLLM_UserTable]:
        """Find a user by SSO ID."""
        return await self.find_by_id(sso_user_id, id_field="sso_user_id")

    async def find_by_organization_id(self, organization_id: str) -> List[LiteLLM_UserTable]:
        """Find all users in an organization."""
        return await self.find_many(where={"organization_id": organization_id})

    async def find_by_team_id(self, team_id: str) -> List[LiteLLM_UserTable]:
        """Find all users in a team."""
        return await self.find_many(where={"teams": {"has": team_id}})

    async def count_billable_users(self) -> int:
        """Number of users that count toward the license seat limit.

        Every user is billable except those SCIM-deactivated
        (metadata.scim_active == false). Rows where scim_active is absent,
        null, or true all count, so seats are counted as total users minus
        the deactivated ones.
        """
        from prisma import Json  # pyright: ignore[reportUnknownVariableType]

        total = await self.count()
        deactivated = await self.count(where={"metadata": {"path": ["scim_active"], "equals": Json(False)}})
        return max(0, total - deactivated)

    async def create_user(
        self,
        user_id: str,
        user_alias: Optional[str] = None,
        team_id: Optional[str] = None,
        sso_user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        password: Optional[str] = None,
        teams: Optional[List[str]] = None,
        user_role: Optional[str] = None,
        max_budget: Optional[float] = None,
        user_email: Optional[str] = None,
        models: Optional[List[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
        max_parallel_requests: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        budget_duration: Optional[str] = None,
        allowed_cache_controls: Optional[List[str]] = None,
        policies: Optional[List[str]] = None,
        object_permission_id: Optional[str] = None,
    ) -> LiteLLM_UserTable:
        """Create a new user."""
        data: Dict[str, object] = {"user_id": user_id}
        if user_alias is not None:
            data["user_alias"] = user_alias
        if team_id is not None:
            data["team_id"] = team_id
        if sso_user_id is not None:
            data["sso_user_id"] = sso_user_id
        if organization_id is not None:
            data["organization_id"] = organization_id
        if password is not None:
            data["password"] = password
        if teams is not None:
            data["teams"] = teams
        if user_role is not None:
            data["user_role"] = user_role
        if max_budget is not None:
            data["max_budget"] = max_budget
        if user_email is not None:
            data["user_email"] = user_email
        if models is not None:
            data["models"] = models
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        if max_parallel_requests is not None:
            data["max_parallel_requests"] = max_parallel_requests
        if tpm_limit is not None:
            data["tpm_limit"] = tpm_limit
        if rpm_limit is not None:
            data["rpm_limit"] = rpm_limit
        if budget_duration is not None:
            data["budget_duration"] = budget_duration
        if allowed_cache_controls is not None:
            data["allowed_cache_controls"] = allowed_cache_controls
        if policies is not None:
            data["policies"] = policies
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.create(data)

    async def update_user(
        self,
        user_id: str,
        user_alias: Optional[str] = None,
        team_id: Optional[str] = None,
        sso_user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        password: Optional[str] = None,
        teams: Optional[List[str]] = None,
        user_role: Optional[str] = None,
        max_budget: Optional[float] = None,
        user_email: Optional[str] = None,
        models: Optional[List[str]] = None,
        metadata: Optional[Mapping[str, object]] = None,
        max_parallel_requests: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        budget_duration: Optional[str] = None,
        allowed_cache_controls: Optional[List[str]] = None,
        policies: Optional[List[str]] = None,
        object_permission_id: Optional[str] = None,
    ) -> Optional[LiteLLM_UserTable]:
        """Update a user."""
        data: Dict[str, object] = {}
        if user_alias is not None:
            data["user_alias"] = user_alias
        if team_id is not None:
            data["team_id"] = team_id
        if sso_user_id is not None:
            data["sso_user_id"] = sso_user_id
        if organization_id is not None:
            data["organization_id"] = organization_id
        if password is not None:
            data["password"] = password
        if teams is not None:
            data["teams"] = teams
        if user_role is not None:
            data["user_role"] = user_role
        if max_budget is not None:
            data["max_budget"] = max_budget
        if user_email is not None:
            data["user_email"] = user_email
        if models is not None:
            data["models"] = models
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        if max_parallel_requests is not None:
            data["max_parallel_requests"] = max_parallel_requests
        if tpm_limit is not None:
            data["tpm_limit"] = tpm_limit
        if rpm_limit is not None:
            data["rpm_limit"] = rpm_limit
        if budget_duration is not None:
            data["budget_duration"] = budget_duration
        if allowed_cache_controls is not None:
            data["allowed_cache_controls"] = allowed_cache_controls
        if policies is not None:
            data["policies"] = policies
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.update(user_id, data, id_field="user_id")

    async def delete_user(self, user_id: str) -> Optional[LiteLLM_UserTable]:
        """Delete a user."""
        return await self.delete(user_id, id_field="user_id")

    async def update_spend(self, user_id: str, spend: float) -> Optional[LiteLLM_UserTable]:
        """Update user spend."""
        return await self.update(user_id, {"spend": spend}, id_field="user_id")

    async def add_to_team(self, user_id: str, team_id: str) -> Optional[LiteLLM_UserTable]:
        """Add a user to a team using atomic array push operation."""
        if not await self.exists(user_id, id_field="user_id"):
            return None

        return await self.update(user_id, {"teams": {"push": team_id}}, id_field="user_id")

    async def remove_from_team(self, user_id: str, team_id: str) -> Optional[LiteLLM_UserTable]:
        """Remove a user from a team.

        Note: Prisma doesn't support atomic array removal, so we use a
        read-modify-write pattern here. For high-concurrency scenarios,
        consider using raw SQL with array_remove().
        """
        user = await self.find_by_id(user_id)
        if user is None:
            return None

        teams = [t for t in user.teams if t != team_id]
        return await self.update(user_id, {"teams": teams}, id_field="user_id")

"""
VerificationToken repository for database operations on LiteLLM_VerificationToken.
"""

import json
from collections.abc import Iterator, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from litellm.models.verification_token import (
    LiteLLM_VerificationToken,
)
from litellm.repositories.base_repository import BaseRepository

if TYPE_CHECKING:
    from prisma.models import (
        LiteLLM_VerificationToken as PrismaVerificationToken,
    )

    from litellm.proxy.utils import PrismaClient


class _DictConvertible(Protocol):
    def dict(self) -> dict[str, object]: ...

    def __iter__(self) -> Iterator[tuple[str, object]]: ...


class VerificationTokenRepository(BaseRepository[LiteLLM_VerificationToken]):
    """Repository for verification token (API key) database operations."""

    @property
    def prisma_client(self) -> "PrismaClient":
        prisma_client: PrismaClient = super().prisma_client
        return prisma_client

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_verificationtoken

    @property
    def deleted_table(self) -> Any:
        return self.prisma_client.db.litellm_deletedverificationtoken

    @property
    def model_class(self) -> type[LiteLLM_VerificationToken]:
        return LiteLLM_VerificationToken

    def _to_model(self, record: _DictConvertible | None) -> LiteLLM_VerificationToken | None:
        """Convert a database record to a VerificationToken model."""
        if record is None:
            return None

        data = record.dict() if hasattr(record, "dict") else dict(record)

        json_fields = [
            "aliases",
            "config",
            "permissions",
            "metadata",
            "model_spend",
            "model_max_budget",
            "router_settings",
            "budget_limits",
            "litellm_budget_table",
        ]
        for field in json_fields:
            value = data.get(field)
            if isinstance(value, str):
                data[field] = json.loads(value)

        if data.get("org_id") is None and data.get("organization_id") is not None:
            data["org_id"] = data["organization_id"]

        return LiteLLM_VerificationToken.model_validate(data)

    async def find_by_id(self, token: str, id_field: str = "token") -> LiteLLM_VerificationToken | None:
        return await super().find_by_id(token, id_field)

    async def find_by_alias(self, key_alias: str) -> LiteLLM_VerificationToken | None:
        """Find a token by key alias."""
        records: list[PrismaVerificationToken] = await self.table.find_many(where={"key_alias": key_alias})
        if records:
            return self._to_model(records[0])
        return None

    async def find_by_user_id(self, user_id: str) -> list[LiteLLM_VerificationToken]:
        """Find all tokens belonging to a user."""
        records: list[PrismaVerificationToken] = await self.table.find_many(where={"user_id": user_id})
        return self._to_model_list(records)

    async def find_by_team_id(self, team_id: str) -> list[LiteLLM_VerificationToken]:
        """Find all tokens belonging to a team."""
        records: list[PrismaVerificationToken] = await self.table.find_many(where={"team_id": team_id})
        return self._to_model_list(records)

    async def find_by_project_id(self, project_id: str) -> list[LiteLLM_VerificationToken]:
        """Find all tokens belonging to a project."""
        records: list[PrismaVerificationToken] = await self.table.find_many(where={"project_id": project_id})
        return self._to_model_list(records)

    async def find_active_tokens(self) -> list[LiteLLM_VerificationToken]:
        """Find all active (non-expired, non-blocked) tokens."""
        records: list[PrismaVerificationToken] = await self.table.find_many(
            where={
                "blocked": {"not": True},
                "OR": [{"expires": None}, {"expires": {"gt": datetime.utcnow()}}],
            }
        )
        return self._to_model_list(records)

    def _build_token_data(
        self,
        token: str,
        key_name: str | None = None,
        key_alias: str | None = None,
        max_budget: float | None = None,
        expires: datetime | None = None,
        models: list[str] | None = None,
        aliases: dict[str, str] | None = None,
        config: Mapping[str, object] | None = None,
        user_id: str | None = None,
        team_id: str | None = None,
        agent_id: str | None = None,
        project_id: str | None = None,
        max_parallel_requests: int | None = None,
        metadata: Mapping[str, object] | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        budget_duration: str | None = None,
        allowed_cache_controls: list[str] | None = None,
        allowed_routes: list[str] | None = None,
        permissions: Mapping[str, object] | None = None,
        org_id: str | None = None,
        created_by: str | None = None,
        object_permission_id: str | None = None,
        access_group_ids: list[str] | None = None,
        budget_id: str | None = None,
    ) -> dict[str, object]:
        """Build data dictionary for token creation."""
        json_fields = {
            "aliases": aliases,
            "config": config,
            "metadata": metadata,
            "permissions": permissions,
        }
        simple_fields = {
            "token": token,
            "key_name": key_name,
            "key_alias": key_alias,
            "max_budget": max_budget,
            "expires": expires,
            "models": models,
            "user_id": user_id,
            "team_id": team_id,
            "agent_id": agent_id,
            "project_id": project_id,
            "max_parallel_requests": max_parallel_requests,
            "tpm_limit": tpm_limit,
            "rpm_limit": rpm_limit,
            "budget_duration": budget_duration,
            "allowed_cache_controls": allowed_cache_controls,
            "allowed_routes": allowed_routes,
            "object_permission_id": object_permission_id,
            "access_group_ids": access_group_ids,
            "budget_id": budget_id,
        }
        data: dict[str, object] = {k: v for k, v in simple_fields.items() if v is not None}
        for key, val in json_fields.items():
            if val is not None:
                data[key] = json.dumps(val)
        if org_id is not None:
            data["organization_id"] = org_id
        if created_by is not None:
            data["created_by"] = created_by
            data["updated_by"] = created_by
        return data

    async def create_token(
        self,
        token: str,
        key_name: str | None = None,
        key_alias: str | None = None,
        max_budget: float | None = None,
        expires: datetime | None = None,
        models: list[str] | None = None,
        aliases: dict[str, str] | None = None,
        config: Mapping[str, object] | None = None,
        user_id: str | None = None,
        team_id: str | None = None,
        agent_id: str | None = None,
        project_id: str | None = None,
        max_parallel_requests: int | None = None,
        metadata: Mapping[str, object] | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        budget_duration: str | None = None,
        allowed_cache_controls: list[str] | None = None,
        allowed_routes: list[str] | None = None,
        permissions: Mapping[str, object] | None = None,
        org_id: str | None = None,
        created_by: str | None = None,
        object_permission_id: str | None = None,
        access_group_ids: list[str] | None = None,
        budget_id: str | None = None,
    ) -> LiteLLM_VerificationToken:
        """Create a new verification token."""
        data = self._build_token_data(
            token=token,
            key_name=key_name,
            key_alias=key_alias,
            max_budget=max_budget,
            expires=expires,
            models=models,
            aliases=aliases,
            config=config,
            user_id=user_id,
            team_id=team_id,
            agent_id=agent_id,
            project_id=project_id,
            max_parallel_requests=max_parallel_requests,
            metadata=metadata,
            tpm_limit=tpm_limit,
            rpm_limit=rpm_limit,
            budget_duration=budget_duration,
            allowed_cache_controls=allowed_cache_controls,
            allowed_routes=allowed_routes,
            permissions=permissions,
            org_id=org_id,
            created_by=created_by,
            object_permission_id=object_permission_id,
            access_group_ids=access_group_ids,
            budget_id=budget_id,
        )
        return await self.create(data)

    async def update_token(
        self,
        token: str,
        updated_by: str | None = None,
        key_name: str | None = None,
        key_alias: str | None = None,
        max_budget: float | None = None,
        expires: datetime | None = None,
        models: list[str] | None = None,
        aliases: dict[str, str] | None = None,
        config: Mapping[str, object] | None = None,
        max_parallel_requests: int | None = None,
        metadata: Mapping[str, object] | None = None,
        tpm_limit: int | None = None,
        rpm_limit: int | None = None,
        budget_duration: str | None = None,
        allowed_cache_controls: list[str] | None = None,
        allowed_routes: list[str] | None = None,
        permissions: Mapping[str, object] | None = None,
        blocked: bool | None = None,
        object_permission_id: str | None = None,
        access_group_ids: list[str] | None = None,
    ) -> LiteLLM_VerificationToken | None:
        """Update a verification token."""
        data: dict[str, object] = {}
        if updated_by is not None:
            data["updated_by"] = updated_by
        if key_name is not None:
            data["key_name"] = key_name
        if key_alias is not None:
            data["key_alias"] = key_alias
        if max_budget is not None:
            data["max_budget"] = max_budget
        if expires is not None:
            data["expires"] = expires
        if models is not None:
            data["models"] = models
        if aliases is not None:
            data["aliases"] = json.dumps(aliases)
        if config is not None:
            data["config"] = json.dumps(config)
        if max_parallel_requests is not None:
            data["max_parallel_requests"] = max_parallel_requests
        if metadata is not None:
            data["metadata"] = json.dumps(metadata)
        if tpm_limit is not None:
            data["tpm_limit"] = tpm_limit
        if rpm_limit is not None:
            data["rpm_limit"] = rpm_limit
        if budget_duration is not None:
            data["budget_duration"] = budget_duration
        if allowed_cache_controls is not None:
            data["allowed_cache_controls"] = allowed_cache_controls
        if allowed_routes is not None:
            data["allowed_routes"] = allowed_routes
        if permissions is not None:
            data["permissions"] = json.dumps(permissions)
        if blocked is not None:
            data["blocked"] = blocked
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id
        if access_group_ids is not None:
            data["access_group_ids"] = access_group_ids

        return await self.update(token, data, id_field="token")

    async def delete_token(
        self,
        token: str,
        deleted_by: str | None = None,
        deleted_by_api_key: str | None = None,
        litellm_changed_by: str | None = None,
    ) -> LiteLLM_VerificationToken | None:
        """Delete a token and archive it to the deleted tokens table.

        Uses a transaction to ensure atomicity of the archive-then-delete operation.
        """
        token_record = await self.find_by_id(token)
        if token_record is None:
            return None

        archive_data = self._build_archive_data(token_record)
        archive_data["deleted_by"] = deleted_by
        archive_data["deleted_by_api_key"] = deleted_by_api_key
        archive_data["litellm_changed_by"] = litellm_changed_by
        archive_data["deleted_at"] = datetime.utcnow()

        async with self.prisma_client.db.tx() as tx:
            await tx.litellm_deletedverificationtoken.create(data=archive_data)
            await tx.litellm_verificationtoken.delete(where={"token": token})

        return token_record

    def _build_archive_data(self, token: LiteLLM_VerificationToken) -> dict[str, object]:
        """Build archive data with only columns present in LiteLLM_DeletedVerificationToken.

        Serializes JSON columns to strings (the archive table stores them as JSON
        columns the same way the live table does) and maps ``org_id`` onto the
        ``organization_id`` column so the foreign key is preserved.
        """
        data: dict[str, object] = token.model_dump(exclude_none=True)
        for field in ("object_permission", "litellm_budget_table", "budget_limits"):
            data.pop(field, None)

        org_id = data.pop("org_id", None)
        if org_id is not None:
            data["organization_id"] = org_id

        json_fields = [
            "aliases",
            "config",
            "permissions",
            "metadata",
            "model_spend",
            "model_max_budget",
            "router_settings",
        ]
        for field in json_fields:
            if field in data:
                data[field] = json.dumps(data[field])
        return data

    async def update_spend(self, token: str, spend: float) -> LiteLLM_VerificationToken | None:
        """Update token spend."""
        return await self.update(token, {"spend": spend}, id_field="token")

    async def update_last_active(self, token: str) -> LiteLLM_VerificationToken | None:
        """Update the last_active timestamp."""
        return await self.update(token, {"last_active": datetime.utcnow()}, id_field="token")

    async def block_token(self, token: str, updated_by: str | None = None) -> LiteLLM_VerificationToken | None:
        """Block a token."""
        data: dict[str, object] = {"blocked": True}
        if updated_by is not None:
            data["updated_by"] = updated_by
        return await self.update(token, data, id_field="token")

    async def unblock_token(self, token: str, updated_by: str | None = None) -> LiteLLM_VerificationToken | None:
        """Unblock a token."""
        data: dict[str, object] = {"blocked": False}
        if updated_by is not None:
            data["updated_by"] = updated_by
        return await self.update(token, data, id_field="token")

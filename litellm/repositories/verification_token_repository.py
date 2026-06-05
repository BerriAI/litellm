"""
VerificationToken repository for database operations on LiteLLM_VerificationToken.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from litellm.models.verification_token import (
    LiteLLM_VerificationToken,
)
from litellm.repositories.base_repository import BaseRepository


class VerificationTokenRepository(BaseRepository[LiteLLM_VerificationToken]):
    """Repository for verification token (API key) database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_verificationtoken

    @property
    def deleted_table(self) -> Any:
        return self.prisma_client.db.litellm_deletedverificationtoken

    @property
    def model_class(self) -> Type[LiteLLM_VerificationToken]:
        return LiteLLM_VerificationToken

    def _to_model(self, record: Any) -> Optional[LiteLLM_VerificationToken]:
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
            if isinstance(data.get(field), str):
                data[field] = json.loads(data[field])

        if data.get("org_id") is None and data.get("organization_id") is not None:
            data["org_id"] = data["organization_id"]

        return LiteLLM_VerificationToken(**data)

    async def find_by_id(
        self, token: str, id_field: str = "token"
    ) -> Optional[LiteLLM_VerificationToken]:
        return await super().find_by_id(token, id_field)

    async def find_by_alias(
        self, key_alias: str
    ) -> Optional[LiteLLM_VerificationToken]:
        """Find a token by key alias."""
        records = await self.table.find_many(where={"key_alias": key_alias})
        if records:
            return self._to_model(records[0])
        return None

    async def find_by_user_id(self, user_id: str) -> List[LiteLLM_VerificationToken]:
        """Find all tokens belonging to a user."""
        records = await self.table.find_many(where={"user_id": user_id})
        return self._to_model_list(records)

    async def find_by_team_id(self, team_id: str) -> List[LiteLLM_VerificationToken]:
        """Find all tokens belonging to a team."""
        records = await self.table.find_many(where={"team_id": team_id})
        return self._to_model_list(records)

    async def find_by_project_id(
        self, project_id: str
    ) -> List[LiteLLM_VerificationToken]:
        """Find all tokens belonging to a project."""
        records = await self.table.find_many(where={"project_id": project_id})
        return self._to_model_list(records)

    async def find_active_tokens(self) -> List[LiteLLM_VerificationToken]:
        """Find all active (non-expired, non-blocked) tokens."""
        records = await self.table.find_many(
            where={
                "blocked": {"not": True},
                "OR": [{"expires": None}, {"expires": {"gt": datetime.utcnow()}}],
            }
        )
        return self._to_model_list(records)

    def _build_token_data(
        self,
        token: str,
        key_name: Optional[str] = None,
        key_alias: Optional[str] = None,
        max_budget: Optional[float] = None,
        expires: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        max_parallel_requests: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tpm_limit: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        budget_duration: Optional[str] = None,
        allowed_cache_controls: Optional[List[str]] = None,
        allowed_routes: Optional[List[str]] = None,
        permissions: Optional[Dict[str, Any]] = None,
        org_id: Optional[str] = None,
        created_by: Optional[str] = None,
        object_permission_id: Optional[str] = None,
        access_group_ids: Optional[List[str]] = None,
        budget_id: Optional[str] = None,
    ) -> Dict[str, Any]:
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
        data: Dict[str, Any] = {k: v for k, v in simple_fields.items() if v is not None}
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
        key_name: Optional[str] = None,
        key_alias: Optional[str] = None,
        max_budget: Optional[float] = None,
        expires: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
        team_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        project_id: Optional[str] = None,
        max_parallel_requests: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tpm_limit: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        budget_duration: Optional[str] = None,
        allowed_cache_controls: Optional[List[str]] = None,
        allowed_routes: Optional[List[str]] = None,
        permissions: Optional[Dict[str, Any]] = None,
        org_id: Optional[str] = None,
        created_by: Optional[str] = None,
        object_permission_id: Optional[str] = None,
        access_group_ids: Optional[List[str]] = None,
        budget_id: Optional[str] = None,
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
        updated_by: Optional[str] = None,
        key_name: Optional[str] = None,
        key_alias: Optional[str] = None,
        max_budget: Optional[float] = None,
        expires: Optional[datetime] = None,
        models: Optional[List[str]] = None,
        aliases: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        max_parallel_requests: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tpm_limit: Optional[int] = None,
        rpm_limit: Optional[int] = None,
        budget_duration: Optional[str] = None,
        allowed_cache_controls: Optional[List[str]] = None,
        allowed_routes: Optional[List[str]] = None,
        permissions: Optional[Dict[str, Any]] = None,
        blocked: Optional[bool] = None,
        object_permission_id: Optional[str] = None,
        access_group_ids: Optional[List[str]] = None,
    ) -> Optional[LiteLLM_VerificationToken]:
        """Update a verification token."""
        data: Dict[str, Any] = {}
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
        deleted_by: Optional[str] = None,
        deleted_by_api_key: Optional[str] = None,
        litellm_changed_by: Optional[str] = None,
    ) -> Optional[LiteLLM_VerificationToken]:
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

    def _build_archive_data(self, token: LiteLLM_VerificationToken) -> Dict[str, Any]:
        """Build archive data with only columns present in LiteLLM_DeletedVerificationToken.

        Serializes JSON columns to strings (the archive table stores them as JSON
        columns the same way the live table does) and maps ``org_id`` onto the
        ``organization_id`` column so the foreign key is preserved.
        """
        data = token.model_dump(exclude_none=True)
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

    async def update_spend(
        self, token: str, spend: float
    ) -> Optional[LiteLLM_VerificationToken]:
        """Update token spend."""
        return await self.update(token, {"spend": spend}, id_field="token")

    async def update_last_active(
        self, token: str
    ) -> Optional[LiteLLM_VerificationToken]:
        """Update the last_active timestamp."""
        return await self.update(
            token, {"last_active": datetime.utcnow()}, id_field="token"
        )

    async def block_token(
        self, token: str, updated_by: Optional[str] = None
    ) -> Optional[LiteLLM_VerificationToken]:
        """Block a token."""
        data: Dict[str, Any] = {"blocked": True}
        if updated_by is not None:
            data["updated_by"] = updated_by
        return await self.update(token, data, id_field="token")

    async def unblock_token(
        self, token: str, updated_by: Optional[str] = None
    ) -> Optional[LiteLLM_VerificationToken]:
        """Unblock a token."""
        data: Dict[str, Any] = {"blocked": False}
        if updated_by is not None:
            data["updated_by"] = updated_by
        return await self.update(token, data, id_field="token")

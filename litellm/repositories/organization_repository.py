"""
Organization repository for database operations on LiteLLM_OrganizationTable.
"""

from typing import Any, Dict, List, Optional, Type

from litellm.models.organization import LiteLLM_OrganizationTable
from litellm.repositories.base_repository import BaseRepository


class OrganizationRepository(BaseRepository[LiteLLM_OrganizationTable]):
    """Repository for organization database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_organizationtable

    @property
    def model_class(self) -> Type[LiteLLM_OrganizationTable]:
        return LiteLLM_OrganizationTable

    async def find_by_id(
        self, organization_id: str, id_field: str = "organization_id"
    ) -> Optional[LiteLLM_OrganizationTable]:
        return await super().find_by_id(organization_id, id_field)

    async def find_by_alias(
        self, organization_alias: str
    ) -> Optional[LiteLLM_OrganizationTable]:
        """Find an organization by alias."""
        records = await self.table.find_many(
            where={"organization_alias": organization_alias}
        )
        if records:
            return self._to_model(records[0])
        return None

    async def create_organization(
        self,
        organization_alias: str,
        budget_id: str,
        created_by: str,
        organization_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        models: Optional[List[str]] = None,
        object_permission_id: Optional[str] = None,
    ) -> LiteLLM_OrganizationTable:
        """Create a new organization."""
        data: Dict[str, Any] = {
            "organization_alias": organization_alias,
            "budget_id": budget_id,
            "created_by": created_by,
            "updated_by": created_by,
        }
        if organization_id is not None:
            data["organization_id"] = organization_id
        if metadata is not None:
            data["metadata"] = metadata
        if models is not None:
            data["models"] = models
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.create(data)

    async def update_organization(
        self,
        organization_id: str,
        updated_by: str,
        organization_alias: Optional[str] = None,
        budget_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        models: Optional[List[str]] = None,
        object_permission_id: Optional[str] = None,
    ) -> Optional[LiteLLM_OrganizationTable]:
        """Update an organization."""
        data: Dict[str, Any] = {"updated_by": updated_by}
        if organization_alias is not None:
            data["organization_alias"] = organization_alias
        if budget_id is not None:
            data["budget_id"] = budget_id
        if metadata is not None:
            data["metadata"] = metadata
        if models is not None:
            data["models"] = models
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.update(organization_id, data, id_field="organization_id")

    async def delete_organization(
        self, organization_id: str
    ) -> Optional[LiteLLM_OrganizationTable]:
        """Delete an organization."""
        return await self.delete(organization_id, id_field="organization_id")

    async def update_spend(
        self, organization_id: str, spend: float
    ) -> Optional[LiteLLM_OrganizationTable]:
        """Update organization spend."""
        return await self.update(
            organization_id, {"spend": spend}, id_field="organization_id"
        )

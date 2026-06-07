"""
Project repository for database operations on LiteLLM_ProjectTable.
"""

from typing import Any, Dict, List, Optional, Type

from litellm.models.project import LiteLLM_ProjectTable
from litellm.repositories.base_repository import BaseRepository


class ProjectRepository(BaseRepository[LiteLLM_ProjectTable]):
    """Repository for project database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_projecttable

    @property
    def model_class(self) -> Type[LiteLLM_ProjectTable]:
        return LiteLLM_ProjectTable

    async def find_by_id(
        self, project_id: str, id_field: str = "project_id"
    ) -> Optional[LiteLLM_ProjectTable]:
        return await super().find_by_id(project_id, id_field)

    async def find_by_alias(self, project_alias: str) -> Optional[LiteLLM_ProjectTable]:
        """Find a project by alias."""
        records = await self.table.find_many(where={"project_alias": project_alias})
        if records:
            return self._to_model(records[0])
        return None

    async def find_by_team_id(self, team_id: str) -> List[LiteLLM_ProjectTable]:
        """Find all projects belonging to a team."""
        records = await self.table.find_many(where={"team_id": team_id})
        return self._to_model_list(records)

    async def create_project(
        self,
        created_by: str,
        project_id: Optional[str] = None,
        project_alias: Optional[str] = None,
        description: Optional[str] = None,
        team_id: Optional[str] = None,
        budget_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        models: Optional[List[str]] = None,
        model_rpm_limit: Optional[Dict[str, int]] = None,
        model_tpm_limit: Optional[Dict[str, int]] = None,
        object_permission_id: Optional[str] = None,
    ) -> LiteLLM_ProjectTable:
        """Create a new project."""
        data: Dict[str, Any] = {
            "created_by": created_by,
            "updated_by": created_by,
        }
        if project_id is not None:
            data["project_id"] = project_id
        if project_alias is not None:
            data["project_alias"] = project_alias
        if description is not None:
            data["description"] = description
        if team_id is not None:
            data["team_id"] = team_id
        if budget_id is not None:
            data["budget_id"] = budget_id
        if metadata is not None:
            data["metadata"] = metadata
        if models is not None:
            data["models"] = models
        if model_rpm_limit is not None:
            data["model_rpm_limit"] = model_rpm_limit
        if model_tpm_limit is not None:
            data["model_tpm_limit"] = model_tpm_limit
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.create(data)

    async def update_project(
        self,
        project_id: str,
        updated_by: str,
        project_alias: Optional[str] = None,
        description: Optional[str] = None,
        team_id: Optional[str] = None,
        budget_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        models: Optional[List[str]] = None,
        model_rpm_limit: Optional[Dict[str, int]] = None,
        model_tpm_limit: Optional[Dict[str, int]] = None,
        blocked: Optional[bool] = None,
        object_permission_id: Optional[str] = None,
    ) -> Optional[LiteLLM_ProjectTable]:
        """Update a project."""
        data: Dict[str, Any] = {"updated_by": updated_by}
        if project_alias is not None:
            data["project_alias"] = project_alias
        if description is not None:
            data["description"] = description
        if team_id is not None:
            data["team_id"] = team_id
        if budget_id is not None:
            data["budget_id"] = budget_id
        if metadata is not None:
            data["metadata"] = metadata
        if models is not None:
            data["models"] = models
        if model_rpm_limit is not None:
            data["model_rpm_limit"] = model_rpm_limit
        if model_tpm_limit is not None:
            data["model_tpm_limit"] = model_tpm_limit
        if blocked is not None:
            data["blocked"] = blocked
        if object_permission_id is not None:
            data["object_permission_id"] = object_permission_id

        return await self.update(project_id, data, id_field="project_id")

    async def delete_project(self, project_id: str) -> Optional[LiteLLM_ProjectTable]:
        """Delete a project."""
        return await self.delete(project_id, id_field="project_id")

    async def update_spend(
        self, project_id: str, spend: float
    ) -> Optional[LiteLLM_ProjectTable]:
        """Update project spend."""
        return await self.update(project_id, {"spend": spend}, id_field="project_id")

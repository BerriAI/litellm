"""
Project repository for database operations on LiteLLM_ProjectTable.
"""

from typing import Any, Final

from litellm.models.project import LiteLLM_ProjectTable
from litellm.repositories.base_repository import BaseRepository


class ProjectRepository(BaseRepository[LiteLLM_ProjectTable]):
    """Repository for project database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_projecttable

    @property
    def model_class(self) -> type[LiteLLM_ProjectTable]:
        return LiteLLM_ProjectTable

    async def find_by_id(self, project_id: str, id_field: str = "project_id") -> LiteLLM_ProjectTable | None:
        return await super().find_by_id(project_id, id_field)

    async def find_by_alias(self, project_alias: str) -> LiteLLM_ProjectTable | None:
        """Find a project by alias."""
        projects: Final = await self.find_many(where={"project_alias": project_alias})
        return projects[0] if projects else None

    async def find_by_team_id(self, team_id: str) -> list[LiteLLM_ProjectTable]:
        """Find all projects belonging to a team."""
        return await self.find_many(where={"team_id": team_id})

    async def create_project(
        self,
        created_by: str,
        project_id: str | None = None,
        project_alias: str | None = None,
        description: str | None = None,
        team_id: str | None = None,
        budget_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        models: list[str] | None = None,
        model_rpm_limit: dict[str, int] | None = None,
        model_tpm_limit: dict[str, int] | None = None,
        object_permission_id: str | None = None,
    ) -> LiteLLM_ProjectTable:
        """Create a new project."""
        data: Final[dict[str, Any]] = {
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
        project_alias: str | None = None,
        description: str | None = None,
        team_id: str | None = None,
        budget_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        models: list[str] | None = None,
        model_rpm_limit: dict[str, int] | None = None,
        model_tpm_limit: dict[str, int] | None = None,
        blocked: bool | None = None,
        object_permission_id: str | None = None,
    ) -> LiteLLM_ProjectTable | None:
        """Update a project."""
        data: Final[dict[str, Any]] = {"updated_by": updated_by}
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

    async def delete_project(self, project_id: str) -> LiteLLM_ProjectTable | None:
        """Delete a project."""
        return await self.delete(project_id, id_field="project_id")

    async def update_spend(self, project_id: str, spend: float) -> LiteLLM_ProjectTable | None:
        """Update project spend."""
        return await self.update(project_id, {"spend": spend}, id_field="project_id")

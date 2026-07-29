"""
ObjectPermission repository for database operations on LiteLLM_ObjectPermissionTable.
"""

from typing import Any, Dict, List, Optional, Type

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.repositories.base_repository import BaseRepository


class ObjectPermissionRepository(BaseRepository[LiteLLM_ObjectPermissionTable]):
    """Repository for object permission database operations."""

    @property
    def table(self) -> Any:
        return self.prisma_client.db.litellm_objectpermissiontable

    @property
    def model_class(self) -> Type[LiteLLM_ObjectPermissionTable]:
        return LiteLLM_ObjectPermissionTable

    async def find_by_id(
        self, object_permission_id: str, id_field: str = "object_permission_id"
    ) -> Optional[LiteLLM_ObjectPermissionTable]:
        return await super().find_by_id(object_permission_id, id_field)

    async def create_permission(
        self,
        mcp_servers: Optional[List[str]] = None,
        mcp_access_groups: Optional[List[str]] = None,
        mcp_tool_permissions: Optional[Dict[str, List[str]]] = None,
        vector_stores: Optional[List[str]] = None,
        agents: Optional[List[str]] = None,
        agent_access_groups: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        blocked_tools: Optional[List[str]] = None,
        mcp_toolsets: Optional[List[str]] = None,
        search_tools: Optional[List[str]] = None,
    ) -> LiteLLM_ObjectPermissionTable:
        """Create a new object permission record."""
        data: Dict[str, Any] = {}
        if mcp_servers is not None:
            data["mcp_servers"] = mcp_servers
        if mcp_access_groups is not None:
            data["mcp_access_groups"] = mcp_access_groups
        if mcp_tool_permissions is not None:
            data["mcp_tool_permissions"] = mcp_tool_permissions
        if vector_stores is not None:
            data["vector_stores"] = vector_stores
        if agents is not None:
            data["agents"] = agents
        if agent_access_groups is not None:
            data["agent_access_groups"] = agent_access_groups
        if models is not None:
            data["models"] = models
        if blocked_tools is not None:
            data["blocked_tools"] = blocked_tools
        if mcp_toolsets is not None:
            data["mcp_toolsets"] = mcp_toolsets
        if search_tools is not None:
            data["search_tools"] = search_tools

        return await self.create(data)

    async def update_permission(
        self,
        object_permission_id: str,
        mcp_servers: Optional[List[str]] = None,
        mcp_access_groups: Optional[List[str]] = None,
        mcp_tool_permissions: Optional[Dict[str, List[str]]] = None,
        vector_stores: Optional[List[str]] = None,
        agents: Optional[List[str]] = None,
        agent_access_groups: Optional[List[str]] = None,
        models: Optional[List[str]] = None,
        blocked_tools: Optional[List[str]] = None,
        mcp_toolsets: Optional[List[str]] = None,
        search_tools: Optional[List[str]] = None,
    ) -> Optional[LiteLLM_ObjectPermissionTable]:
        """Update an object permission record."""
        data: Dict[str, Any] = {}
        if mcp_servers is not None:
            data["mcp_servers"] = mcp_servers
        if mcp_access_groups is not None:
            data["mcp_access_groups"] = mcp_access_groups
        if mcp_tool_permissions is not None:
            data["mcp_tool_permissions"] = mcp_tool_permissions
        if vector_stores is not None:
            data["vector_stores"] = vector_stores
        if agents is not None:
            data["agents"] = agents
        if agent_access_groups is not None:
            data["agent_access_groups"] = agent_access_groups
        if models is not None:
            data["models"] = models
        if blocked_tools is not None:
            data["blocked_tools"] = blocked_tools
        if mcp_toolsets is not None:
            data["mcp_toolsets"] = mcp_toolsets
        if search_tools is not None:
            data["search_tools"] = search_tools

        return await self.update(object_permission_id, data, id_field="object_permission_id")

    async def delete_permission(self, object_permission_id: str) -> Optional[LiteLLM_ObjectPermissionTable]:
        """Delete an object permission record."""
        return await self.delete(object_permission_id, id_field="object_permission_id")

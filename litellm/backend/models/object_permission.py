"""
ObjectPermission domain model.
"""

from typing import Dict, List, Optional

from pydantic import Field

from litellm.backend.models.base import DomainModel


class ObjectPermission(DomainModel):
    """Domain model for object-level permissions (MCP, vector stores, agents, etc.)."""

    object_permission_id: Optional[str] = None
    mcp_servers: List[str] = Field(default_factory=list)
    mcp_access_groups: List[str] = Field(default_factory=list)
    mcp_tool_permissions: Optional[Dict[str, List[str]]] = None
    vector_stores: List[str] = Field(default_factory=list)
    agents: List[str] = Field(default_factory=list)
    agent_access_groups: List[str] = Field(default_factory=list)
    models: List[str] = Field(default_factory=list)
    blocked_tools: List[str] = Field(default_factory=list)
    mcp_toolsets: List[str] = Field(default_factory=list)
    search_tools: List[str] = Field(default_factory=list)

    def has_mcp_server_access(self, server_id: str) -> bool:
        """Check if this permission grants access to a specific MCP server."""
        return server_id in self.mcp_servers

    def has_vector_store_access(self, vector_store_id: str) -> bool:
        """Check if this permission grants access to a specific vector store."""
        return vector_store_id in self.vector_stores

    def has_agent_access(self, agent_id: str) -> bool:
        """Check if this permission grants access to a specific agent."""
        return agent_id in self.agents

    def is_tool_blocked(self, tool_name: str) -> bool:
        """Check if a tool is blocked by this permission."""
        return tool_name in self.blocked_tools

    def get_allowed_tools_for_server(self, server_id: str) -> Optional[List[str]]:
        """Get the list of allowed tools for a specific MCP server."""
        if self.mcp_tool_permissions is None:
            return None
        return self.mcp_tool_permissions.get(server_id)

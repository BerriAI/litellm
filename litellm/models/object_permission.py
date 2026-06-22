"""
Object permission table model.

Canonical definition for ``litellm_objectpermissiontable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from typing import Dict, List, Optional

from litellm.types.llms.base import LiteLLMPydanticObjectBase


class LiteLLM_ObjectPermissionTable(LiteLLMPydanticObjectBase):
    """Represents a LiteLLM_ObjectPermissionTable record"""

    object_permission_id: str
    # None = inherit team scope; [] = no servers; ["id", ...] = exactly those. The
    # default must stay None so the inherit/zero distinction survives cache
    # round-trips (serialize drops None via exclude_none, deserialize restores it).
    mcp_servers: Optional[List[str]] = None
    mcp_access_groups: Optional[List[str]] = []
    mcp_tool_permissions: Optional[Dict[str, List[str]]] = None
    vector_stores: Optional[List[str]] = []
    agents: Optional[List[str]] = []
    agent_access_groups: Optional[List[str]] = []
    models: Optional[List[str]] = []
    mcp_toolsets: Optional[List[str]] = None
    blocked_tools: Optional[List[str]] = []
    search_tools: Optional[List[str]] = []

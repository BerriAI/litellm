"""
Object permission table model.

Canonical definition for ``litellm_objectpermissiontable``. Re-exported from
``litellm.proxy._types`` for backwards compatibility.
"""

from collections.abc import Mapping, Sequence
from typing import Final

from pydantic import SerializerFunctionWrapHandler, TypeAdapter, model_serializer

from litellm.types.llms.base import LiteLLMPydanticObjectBase
from litellm.types.object_permission import without_empty_auth_profile_grants

_DUMPED: Final = TypeAdapter(Mapping[str, object])


class LiteLLM_ObjectPermissionTable(LiteLLMPydanticObjectBase):
    """Represents a LiteLLM_ObjectPermissionTable record"""

    object_permission_id: str
    mcp_servers: list[str] | None = []
    mcp_access_groups: list[str] | None = []
    mcp_tool_permissions: dict[str, list[str]] | None = None
    vector_stores: list[str] | None = []
    agents: list[str] | None = []
    agent_access_groups: list[str] | None = []
    models: list[str] | None = []
    mcp_toolsets: list[str] | None = None
    blocked_tools: list[str] | None = []
    search_tools: list[str] | None = []
    mcp_tool_search_enabled: bool | None = None
    skills: list[str] | None = None
    apps: Sequence[str] | None = None
    mcp_auth_profiles: tuple[str, ...] | None = None

    @model_serializer(mode="wrap")
    def _omit_empty_auth_profile_grants(self, handler: SerializerFunctionWrapHandler):  # noqa: ANN202  # a return annotation collapses the OpenAPI response schema
        return without_empty_auth_profile_grants(_DUMPED.validate_python(handler(self)))

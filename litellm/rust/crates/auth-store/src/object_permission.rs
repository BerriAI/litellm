//! An object permission as the auth cache holds it, nested in the key that points at it.

use litellm_redis_store::auth_cache::ObjectPermissionEntry;
use litellm_repositories::ObjectPermission;

use crate::{Error, column::nullable};

/// What `LiteLLM_ObjectPermissionTable.model_dump(mode="json")` holds for `row`, or the column the
/// model would refuse the row for. The table has no `apps` column, so the model's default stands.
pub(crate) fn object_permission_entry(
    row: ObjectPermission,
) -> Result<ObjectPermissionEntry, Error> {
    let ObjectPermission {
        object_permission_id,
        mcp_servers,
        mcp_access_groups,
        mcp_tool_permissions,
        vector_stores,
        agents,
        agent_access_groups,
        models,
        blocked_tools,
        mcp_toolsets,
        search_tools,
        mcp_tool_search_enabled,
        skills,
    } = row;
    Ok(ObjectPermissionEntry {
        object_permission_id,
        mcp_servers: Some(mcp_servers),
        mcp_access_groups: Some(mcp_access_groups),
        mcp_tool_permissions: nullable("mcp_tool_permissions", mcp_tool_permissions.as_ref())?,
        vector_stores: Some(vector_stores),
        agents: Some(agents),
        agent_access_groups: Some(agent_access_groups),
        models: Some(models),
        mcp_toolsets: Some(mcp_toolsets),
        blocked_tools: Some(blocked_tools),
        search_tools: Some(search_tools),
        mcp_tool_search_enabled,
        skills: Some(skills),
        apps: None,
        unread: Vec::new(),
    })
}

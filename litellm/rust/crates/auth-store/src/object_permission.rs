//! An object permission as the auth cache holds it, nested in the key or team that points at it.

use std::sync::Arc;

use litellm_redis_store::auth_cache::ObjectPermissionEntry;
use litellm_repositories::{ObjectPermission, ObjectPermissionRepository, Side};

use crate::{Error, column::nullable, error::within};

/// The object permission an object points at, read as `get_object_permission` nests it.
pub(crate) struct ObjectPermissions(Arc<ObjectPermissionRepository>);

impl ObjectPermissions {
    pub(crate) fn new(repository: Arc<ObjectPermissionRepository>) -> Self {
        Self(repository)
    }

    /// The permission `id` names, or `None` for no id or no row. A row that cannot be read, or
    /// that the model would refuse, leaves a replica read without it, so one bad permission never
    /// takes down the object that points at it, and fails a writer read.
    pub(crate) async fn nested(
        &self,
        id: Option<&str>,
        side: Side,
    ) -> Result<Option<ObjectPermissionEntry>, Error> {
        let Some(id) = id.filter(|id| !id.is_empty()) else {
            return Ok(None);
        };
        match (self.read(id, side).await, side) {
            (Ok(permission), _) => Ok(permission),
            (Err(error), Side::Writer) => Err(error),
            (Err(error), Side::Read) => {
                tracing::debug!(
                    error = %litellm_error::Report::new(&error),
                    "an object permission was not loaded"
                );
                Ok(None)
            }
        }
    }

    async fn read(&self, id: &str, side: Side) -> Result<Option<ObjectPermissionEntry>, Error> {
        self.0
            .find_by_object_permission_id(id, side)
            .await?
            .map(object_permission_entry)
            .transpose()
            .map_err(within("object_permission"))
    }
}

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

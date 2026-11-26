use super::{
    bag::Lists,
    record::{record, required},
};

record! {
    /// `LiteLLM_ObjectPermissionTable.model_dump(mode="json")`, the value under
    /// [`AuthObject::ObjectPermission`](crate::auth_cache::AuthObject::ObjectPermission) and
    /// nested in a key's, a team's, a project's, an end user's or an organization's entry.
    pub struct ObjectPermissionEntry {
        object_permission_id: String = required!("object_permission_id"),
        mcp_servers: Option<Vec<String>> = Some(Vec::new()),
        mcp_access_groups: Option<Vec<String>> = Some(Vec::new()),
        mcp_tool_permissions: Option<Lists>,
        vector_stores: Option<Vec<String>> = Some(Vec::new()),
        agents: Option<Vec<String>> = Some(Vec::new()),
        agent_access_groups: Option<Vec<String>> = Some(Vec::new()),
        models: Option<Vec<String>> = Some(Vec::new()),
        mcp_toolsets: Option<Vec<String>>,
        blocked_tools: Option<Vec<String>> = Some(Vec::new()),
        search_tools: Option<Vec<String>> = Some(Vec::new()),
        mcp_tool_search_enabled: Option<bool>,
        skills: Option<Vec<String>>,
        apps: Option<Vec<String>>,
    }
}

//! Projects an affiliation's stored object permission without applying its grants.

use litellm_gateway_auth_core::record::ObjectPermissionRecord;
use litellm_redis_store::auth_cache::ObjectPermissionEntry;

use super::carried::newer;

pub(super) fn record(
    id: Option<&str>,
    row: Option<&ObjectPermissionEntry>,
) -> Result<Option<Box<ObjectPermissionRecord>>, &'static str> {
    let Some(row) = row else {
        return if id.is_some() {
            Err("object_permission_id")
        } else {
            Ok(None)
        };
    };
    let ObjectPermissionEntry {
        object_permission_id,
        mcp_servers,
        mcp_access_groups,
        mcp_tool_permissions,
        vector_stores,
        agents,
        agent_access_groups,
        models,
        mcp_toolsets,
        blocked_tools,
        search_tools,
        mcp_tool_search_enabled,
        skills,
        apps,
        unread,
    } = row;
    if object_permission_id.is_empty() || id.is_some_and(|id| id != object_permission_id) {
        return Err("object_permission_id");
    }
    if newer(unread) {
        return Err("object_permission");
    }
    Ok(Some(Box::new(ObjectPermissionRecord {
        object_permission_id: object_permission_id.clone(),
        mcp_servers: mcp_servers.clone().unwrap_or_default(),
        mcp_access_groups: mcp_access_groups.clone().unwrap_or_default(),
        mcp_tool_permissions: mcp_tool_permissions
            .as_ref()
            .map(|map| map.0.iter().cloned().collect())
            .unwrap_or_default(),
        vector_stores: vector_stores.clone().unwrap_or_default(),
        agents: agents.clone().unwrap_or_default(),
        agent_access_groups: agent_access_groups.clone().unwrap_or_default(),
        models: models.clone().unwrap_or_default(),
        mcp_toolsets: mcp_toolsets.clone().unwrap_or_default(),
        blocked_tools: blocked_tools.clone().unwrap_or_default(),
        search_tools: search_tools.clone().unwrap_or_default(),
        mcp_tool_search_enabled: *mcp_tool_search_enabled,
        skills: skills.clone().unwrap_or_default(),
        apps: apps.clone().unwrap_or_default(),
    })))
}

#[cfg(test)]
mod tests {
    use litellm_python_compat::json;
    use litellm_redis_store::FromValue;
    use rstest::rstest;

    use super::*;

    #[rstest]
    fn copies_apps_and_tool_permissions() {
        let row = ObjectPermissionEntry::from_value(&json::loads(
            r#"{"object_permission_id":"p","apps":["app"],"mcp_tool_permissions":{"server":["tool"]}}"#,
        ).unwrap()).unwrap();
        let permission = record(Some("p"), Some(&row)).unwrap().unwrap();
        assert_eq!(permission.apps, ["app"]);
        assert_eq!(permission.mcp_tool_permissions["server"], ["tool"]);
    }

    #[rstest]
    #[case::null(None, r#"null"#, Ok(false))]
    #[case::missing(Some("p"), r#"null"#, Err("object_permission_id"))]
    #[case::mismatched(
        Some("p"),
        r#"{"object_permission_id":"q"}"#,
        Err("object_permission_id")
    )]
    #[case::unread(
        Some("p"),
        r#"{"object_permission_id":"p","future_grants":["a"]}"#,
        Err("object_permission")
    )]
    fn projects_only_the_named_permission(
        #[case] id: Option<&str>,
        #[case] input: &str,
        #[case] expected: Result<bool, &str>,
    ) {
        let value = json::loads(input).unwrap();
        let row = ObjectPermissionEntry::from_value(&value).ok();
        assert_eq!(record(id, row.as_ref()).map(|row| row.is_some()), expected);
    }
}

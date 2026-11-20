//! Reads one `LiteLLM_ObjectPermissionTable` row, the grants keys, teams, users, projects and end users point at.

use sqlx::PgExecutor;

use litellm_db::Error;

use crate::RawJson;

/// One `LiteLLM_ObjectPermissionTable` row.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct ObjectPermission {
    object_permission_id: String,
    mcp_servers: Vec<String>,
    mcp_access_groups: Vec<String>,
    mcp_tool_permissions: Option<RawJson>,
    vector_stores: Vec<String>,
    agents: Vec<String>,
    agent_access_groups: Vec<String>,
    models: Vec<String>,
    blocked_tools: Vec<String>,
    mcp_toolsets: Vec<String>,
    search_tools: Vec<String>,
    mcp_tool_search_enabled: Option<bool>,
    skills: Vec<String>,
}

pub(super) async fn find_object_permission(
    executor: impl PgExecutor<'_>,
    object_permission_id: &str,
) -> Result<Option<ObjectPermission>, Error> {
    Ok(sqlx::query_file_as!(
        ObjectPermission,
        "src/object_permission/queries/find_object_permission.sql",
        object_permission_id,
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::find_object_permission;
    use crate::test_support::database_pool;

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_every_column(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_ObjectPermissionTable"
                (object_permission_id, mcp_servers, mcp_access_groups, mcp_tool_permissions, vector_stores,
                 agents, agent_access_groups, models, blocked_tools, mcp_toolsets, search_tools,
                 mcp_tool_search_enabled, skills)
            VALUES ('perm-full', '{s1,s2}', '{g1}', '{"s1": ["t1"]}', '{v1}', '{a1}', '{ag1}', '{m1}', '{bt}',
                    '{ts}', '{st}', true, '{sk}')
            "#,
        )
        .execute(&mut *tx)
        .await?;

        let found = find_object_permission(&mut *tx, "perm-full")
            .await?
            .ok_or("row missing")?;

        assert_eq!(
            serde_json::to_value(&found)?,
            serde_json::json!({
                "object_permission_id": "perm-full",
                "mcp_servers": ["s1", "s2"],
                "mcp_access_groups": ["g1"],
                "mcp_tool_permissions": {"s1": ["t1"]},
                "vector_stores": ["v1"],
                "agents": ["a1"],
                "agent_access_groups": ["ag1"],
                "models": ["m1"],
                "blocked_tools": ["bt"],
                "mcp_toolsets": ["ts"],
                "search_tools": ["st"],
                "mcp_tool_search_enabled": true,
                "skills": ["sk"],
            })
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn null_lists_read_as_empty(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"
            INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id, mcp_servers, models, skills)
            VALUES ('perm-null', NULL, NULL, NULL)
            "#,
        )
        .execute(&mut *tx)
        .await?;

        let found = find_object_permission(&mut *tx, "perm-null")
            .await?
            .ok_or("row missing")?;

        assert_eq!(
            (found.mcp_servers, found.models, found.skills),
            (vec![], vec![], vec![])
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_row_is_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        assert!(
            find_object_permission(&pool, "perm-missing")
                .await?
                .is_none()
        );
        Ok(())
    }
}

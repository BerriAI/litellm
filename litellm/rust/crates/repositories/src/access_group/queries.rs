//! Reads one `LiteLLM_AccessGroupTable` row, the access object that bundles models, MCP servers, agents,
//! skills and vector stores and assigns them to teams and keys.

use chrono::NaiveDateTime;
use sqlx::PgExecutor;

use litellm_db::Error;

use crate::{RawJson, utc_timestamp};

/// One `LiteLLM_AccessGroupTable` row.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct AccessGroup {
    access_group_id: String,
    access_group_name: String,
    description: Option<String>,
    access_model_names: Vec<String>,
    access_mcp_server_ids: Vec<String>,
    mcp_tool_permissions: Option<RawJson>,
    access_agent_ids: Vec<String>,
    access_skill_names: Vec<String>,
    access_vector_store_ids: Vec<String>,
    assigned_team_ids: Vec<String>,
    assigned_key_ids: Vec<String>,
    #[serde(with = "utc_timestamp")]
    created_at: NaiveDateTime,
    created_by: Option<String>,
    #[serde(with = "utc_timestamp")]
    updated_at: NaiveDateTime,
    updated_by: Option<String>,
}

pub(super) async fn find_access_group(
    executor: impl PgExecutor<'_>,
    access_group_id: &str,
) -> Result<Option<AccessGroup>, Error> {
    Ok(sqlx::query_file_as!(
        AccessGroup,
        "src/access_group/queries/find_access_group.sql",
        access_group_id,
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::find_access_group;
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
            INSERT INTO "LiteLLM_AccessGroupTable"
                (access_group_id, access_group_name, description, access_model_names, access_mcp_server_ids,
                 mcp_tool_permissions, access_agent_ids, access_skill_names, access_vector_store_ids,
                 assigned_team_ids, assigned_key_ids, created_at, created_by, updated_at, updated_by)
            VALUES ('ag-full', 'group-full', 'desc', '{m1}', '{s1}', '{"s1": ["t1"]}', '{a1}', '{sk1}', '{v1}',
                    '{team1}', '{key1}', '2026-01-02 03:04:05.678', 'creator', '2026-02-03 04:05:06.789', NULL)
            "#,
        )
        .execute(&mut *tx)
        .await?;

        let found = find_access_group(&mut *tx, "ag-full")
            .await?
            .ok_or("row missing")?;

        assert_eq!(
            serde_json::to_value(&found)?,
            serde_json::json!({
                "access_group_id": "ag-full",
                "access_group_name": "group-full",
                "description": "desc",
                "access_model_names": ["m1"],
                "access_mcp_server_ids": ["s1"],
                "mcp_tool_permissions": {"s1": ["t1"]},
                "access_agent_ids": ["a1"],
                "access_skill_names": ["sk1"],
                "access_vector_store_ids": ["v1"],
                "assigned_team_ids": ["team1"],
                "assigned_key_ids": ["key1"],
                "created_at": "2026-01-02T03:04:05.678+00:00",
                "created_by": "creator",
                "updated_at": "2026-02-03T04:05:06.789+00:00",
                "updated_by": null,
            })
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
        assert!(find_access_group(&pool, "ag-missing").await?.is_none());
        Ok(())
    }
}

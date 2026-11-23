//! `LiteLLM_EndUserTable` reads: one end user with the budget and object permission it links, and the ids of
//! end users that restrict something auth enforces

use serde::{Deserialize, Serialize};
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use crate::{RawJson, budget::Budget};

/// Every scalar column of one `LiteLLM_ObjectPermissionTable` row
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Deserialize, Serialize)]
pub(super) struct ObjectPermission {
    pub object_permission_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_servers: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_access_groups: Option<Vec<String>>,
    pub mcp_tool_permissions: Option<RawJson>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vector_stores: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agents: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent_access_groups: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub models: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked_tools: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mcp_toolsets: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub search_tools: Option<Vec<String>>,
    pub mcp_tool_search_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub skills: Option<Vec<String>>,
}

/// One `LiteLLM_EndUserTable` row with the budget and object permission it links, as the end user repository hands it out
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Serialize)]
pub struct EndUserRecord {
    pub(super) user_id: String,
    pub(super) alias: Option<String>,
    pub(super) spend: f64,
    pub(super) allowed_model_region: Option<String>,
    pub(super) default_model: Option<String>,
    pub(super) models: Vec<String>,
    pub(super) budget_id: Option<String>,
    pub(super) litellm_budget_table: Option<Json<Budget>>,
    pub(super) object_permission_id: Option<String>,
    pub(super) object_permission: Option<Json<ObjectPermission>>,
    pub(super) blocked: bool,
}

/// The end user `end_user_id` with its linked budget and object permission, or `None` when no row exists
pub(super) async fn find_end_user_record(
    executor: impl PgExecutor<'_>,
    end_user_id: &str,
) -> Result<Option<EndUserRecord>, Error> {
    Ok(sqlx::query_file_as!(
        EndUserRecord,
        "src/end_user/queries/find_end_user_record.sql",
        end_user_id
    )
    .fetch_optional(executor)
    .await?)
}

/// Ids of end users whose row restricts something auth enforces, at most `limit` of them
pub(super) async fn find_restricted_end_user_ids(
    executor: impl PgExecutor<'_>,
    limit: i64,
) -> Result<Vec<String>, Error> {
    Ok(sqlx::query_file_scalar!(
        "src/end_user/queries/find_restricted_end_user_ids.sql",
        limit
    )
    .fetch_all(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use crate::test_support::{database_pool, insert_budget};
    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    async fn insert_end_user(
        tx: &mut Transaction<'_, Postgres>,
        user_id: &str,
        spend: f64,
        blocked: bool,
        budget_id: Option<&str>,
    ) -> Result<(), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked, budget_id)
            VALUES ($1, $2, $3, $4)
            "#,
        )
        .bind(user_id)
        .bind(spend)
        .bind(blocked)
        .bind(budget_id)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn end_user_record_carries_budget_and_object_permission(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_budget(&mut tx, "end-user-record-budget").await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let _ = sqlx::query(
            r#"
            WITH permission AS (
                INSERT INTO "LiteLLM_ObjectPermissionTable" (object_permission_id, models, mcp_tool_permissions)
                VALUES ('end-user-record-permission', ARRAY['gpt-4o'], '{"server": ["tool"]}')
            )
            INSERT INTO "LiteLLM_EndUserTable"
                (user_id, alias, spend, allowed_model_region, models, budget_id, object_permission_id, blocked)
            VALUES ('end-user-record', 'alias', 2.5, 'eu', ARRAY['gpt-4o'], 'end-user-record-budget',
                    'end-user-record-permission', true)
            "#,
        )
        .execute(&mut *tx)
        .await?;

        let record = super::find_end_user_record(&mut *tx, "end-user-record")
            .await?
            .ok_or("end user row missing")?;

        let encoded = serde_json::to_value(&record)?;
        assert_eq!(
            encoded["litellm_budget_table"]["budget_id"],
            "end-user-record-budget"
        );
        assert_eq!(
            encoded["litellm_budget_table"]["created_at"],
            "2026-11-01T00:00:00+00:00"
        );
        assert_eq!(
            serde_json::to_value(&record.object_permission)?,
            serde_json::json!({
                "object_permission_id": "end-user-record-permission",
                "mcp_servers": [],
                "mcp_access_groups": [],
                "mcp_tool_permissions": {"server": ["tool"]},
                "vector_stores": [],
                "agents": [],
                "agent_access_groups": [],
                "models": ["gpt-4o"],
                "blocked_tools": [],
                "mcp_toolsets": [],
                "search_tools": [],
                "mcp_tool_search_enabled": null,
                "skills": [],
            })
        );
        assert_eq!(
            (
                encoded["user_id"].clone(),
                encoded["alias"].clone(),
                encoded["spend"].clone(),
                encoded["allowed_model_region"].clone(),
                encoded["models"].clone(),
                encoded["blocked"].clone(),
            ),
            (
                serde_json::json!("end-user-record"),
                serde_json::json!("alias"),
                serde_json::json!(2.5),
                serde_json::json!("eu"),
                serde_json::json!(["gpt-4o"]),
                serde_json::json!(true),
            )
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn end_user_record_without_links(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_end_user(&mut tx, "end-user-record-plain", 0.0, false, None).await?;

        let record = super::find_end_user_record(&mut *tx, "end-user-record-plain")
            .await?
            .ok_or("end user row missing")?;

        assert!(record.litellm_budget_table.is_none() && record.object_permission.is_none());
        assert!(
            super::find_end_user_record(&mut *tx, "end-user-record-missing")
                .await?
                .is_none()
        );
        Ok(())
    }

    #[rstest]
    #[case::unrestricted("restricted-ids-plain", None, false)]
    #[case::blocked(
        "restricted-ids-blocked",
        Some(r#"UPDATE "LiteLLM_EndUserTable" SET blocked = true WHERE user_id = $1"#),
        true
    )]
    #[case::region(
        "restricted-ids-region",
        Some(
            r#"UPDATE "LiteLLM_EndUserTable" SET allowed_model_region = 'eu' WHERE user_id = $1"#
        ),
        true
    )]
    #[case::default_model(
        "restricted-ids-default-model",
        Some(r#"UPDATE "LiteLLM_EndUserTable" SET default_model = 'gpt-4o' WHERE user_id = $1"#),
        true
    )]
    #[case::models(
        "restricted-ids-models",
        Some(r#"UPDATE "LiteLLM_EndUserTable" SET models = ARRAY['gpt-4o'] WHERE user_id = $1"#),
        true
    )]
    #[case::empty_models(
        "restricted-ids-empty-models",
        Some(r#"UPDATE "LiteLLM_EndUserTable" SET models = ARRAY[]::text[] WHERE user_id = $1"#),
        false
    )]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn restricted_end_user_ids_match_the_registry_predicate(
        #[case] user_id: &str,
        #[case] restriction: Option<&'static str>,
        #[case] restricted: bool,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_end_user(&mut tx, user_id, 0.0, false, None).await?;
        if let Some(restriction) = restriction {
            #[expect(
                clippy::disallowed_methods,
                reason = "Fixture updates are outside the compile-checked query set"
            )]
            let _ = sqlx::query(restriction)
                .bind(user_id)
                .execute(&mut *tx)
                .await?;
        }

        let ids = super::find_restricted_end_user_ids(&mut *tx, i64::MAX).await?;

        assert_eq!(ids.contains(&user_id.to_owned()), restricted);
        Ok(())
    }
}

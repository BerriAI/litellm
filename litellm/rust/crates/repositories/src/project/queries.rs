//! Reads one `LiteLLM_ProjectTable` row with its budget, as project auth checks see it.

use chrono::NaiveDateTime;
use sqlx::{PgExecutor, types::Json};

use litellm_db::Error;

use crate::{RawJson, budget::Budget, utc_timestamp};

/// One `LiteLLM_ProjectTable` row and the `LiteLLM_BudgetTable` row its `budget_id` names, if it exists.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct Project {
    project_id: String,
    project_alias: Option<String>,
    description: Option<String>,
    team_id: Option<String>,
    budget_id: Option<String>,
    metadata: RawJson,
    models: Vec<String>,
    spend: f64,
    model_spend: RawJson,
    model_rpm_limit: RawJson,
    model_tpm_limit: RawJson,
    blocked: bool,
    object_permission_id: Option<String>,
    #[serde(with = "utc_timestamp")]
    created_at: NaiveDateTime,
    created_by: String,
    #[serde(with = "utc_timestamp")]
    updated_at: NaiveDateTime,
    updated_by: String,
    litellm_budget_table: Option<Json<Budget>>,
}

impl Project {
    /// The `spend` column, what a budget counter for this row is seeded and floored from.
    pub fn spend(&self) -> f64 {
        self.spend
    }
}

pub(super) async fn find_project(
    executor: impl PgExecutor<'_>,
    project_id: &str,
) -> Result<Option<Project>, Error> {
    Ok(
        sqlx::query_file_as!(Project, "src/project/queries/find_project.sql", project_id)
            .fetch_optional(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::find_project;
    use crate::test_support::{database_pool, insert_budget};

    async fn insert_project(
        tx: &mut Transaction<'_, Postgres>,
        project_id: &str,
        budget_id: Option<&str>,
    ) -> Result<(), sqlx::Error> {
        if let Some(budget_id) = budget_id {
            insert_budget(tx, budget_id).await?;
        }
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_ProjectTable"
                (project_id, project_alias, team_id, budget_id, metadata, models, spend, model_spend, blocked,
                 created_at, created_by, updated_at, updated_by)
            VALUES ($1, 'alias', NULL, $2, '{"k": "v"}', '{m1,m2}', 3.25, '{"m1": 1.0}', true,
                    '2026-01-02 03:04:05.678', 'creator', '2026-02-03 04:05:06.789', 'updater')
            "#,
        )
        .bind(project_id)
        .bind(budget_id)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    #[rstest]
    #[case::with_budget(Some("project-budget"))]
    #[case::without_budget(None)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_project_with_its_budget(
        #[case] budget_id: Option<&str>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_project(&mut tx, "project-1", budget_id).await?;

        let found = find_project(&mut *tx, "project-1")
            .await?
            .ok_or("row missing")?;

        let budget = budget_id.map(|budget_id| {
            serde_json::json!({
                "budget_id": budget_id,
                "name": "team default",
                "max_budget": 10.5,
                "soft_budget": 8.0,
                "max_parallel_requests": 3,
                "tpm_limit": 1000,
                "rpm_limit": 10,
                "tpd_limit": 50000,
                "model_max_budget": {"gpt-4o": {"max_budget": 1.0}},
                "model_tpm_limit": {"gpt-4o": 100},
                "model_rpm_limit": {"gpt-4o": 5},
                "budget_duration": "30d",
                "budget_limits": null,
                "cap_policy": null,
                "budget_reset_at": "2026-12-01T00:00:00+00:00",
                "budget_reset_count": 2,
                "allowed_models": ["gpt-4o"],
                "temp_budget_increase": 1.5,
                "temp_budget_expiry": "2026-11-20T12:30:00.250+00:00",
                "created_at": "2026-11-01T00:00:00+00:00",
                "created_by": "db-query-test",
                "updated_at": "2026-11-02T00:00:00+00:00",
                "updated_by": "db-query-test",
            })
        });
        assert_eq!(
            serde_json::to_value(&found)?,
            serde_json::json!({
                "project_id": "project-1",
                "project_alias": "alias",
                "description": null,
                "team_id": null,
                "budget_id": budget_id,
                "metadata": {"k": "v"},
                "models": ["m1", "m2"],
                "spend": 3.25,
                "model_spend": {"m1": 1.0},
                "model_rpm_limit": {},
                "model_tpm_limit": {},
                "blocked": true,
                "object_permission_id": null,
                "created_at": "2026-01-02T03:04:05.678+00:00",
                "created_by": "creator",
                "updated_at": "2026-02-03T04:05:06.789+00:00",
                "updated_by": "updater",
                "litellm_budget_table": budget,
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
        assert!(find_project(&pool, "project-missing").await?.is_none());
        Ok(())
    }
}

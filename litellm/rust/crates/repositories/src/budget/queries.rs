//! `LiteLLM_BudgetTable` rows: read by id for the budget store, and joined into the rows that
//! link a budget (end users, memberships, organizations, tags, projects)

use serde::{Deserialize, Serialize};
use sqlx::{PgExecutor, types::chrono::NaiveDateTime};

use litellm_db::Error;

use crate::RawJson;

/// Every scalar column of one `LiteLLM_BudgetTable` row
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(Deserialize, Serialize)]
pub struct Budget {
    pub budget_id: String,
    pub name: Option<String>,
    pub max_budget: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rollover_max_budget: Option<f64>,
    pub soft_budget: Option<f64>,
    pub max_parallel_requests: Option<i32>,
    pub tpm_limit: Option<i64>,
    pub rpm_limit: Option<i64>,
    pub tpd_limit: Option<i64>,
    pub model_max_budget: Option<RawJson>,
    pub model_tpm_limit: Option<RawJson>,
    pub model_rpm_limit: Option<RawJson>,
    pub budget_duration: Option<String>,
    pub budget_limits: Option<RawJson>,
    pub cap_policy: Option<String>,
    #[serde(serialize_with = "crate::utc_timestamp::option::serialize")]
    pub budget_reset_at: Option<NaiveDateTime>,
    pub budget_reset_count: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allowed_models: Option<Vec<String>>,
    pub temp_budget_increase: Option<f64>,
    #[serde(serialize_with = "crate::utc_timestamp::option::serialize")]
    pub temp_budget_expiry: Option<NaiveDateTime>,
    #[serde(serialize_with = "crate::utc_timestamp::serialize")]
    pub created_at: NaiveDateTime,
    pub created_by: String,
    #[serde(serialize_with = "crate::utc_timestamp::serialize")]
    pub updated_at: NaiveDateTime,
    pub updated_by: String,
}

pub(super) async fn find_budget(
    executor: impl PgExecutor<'_>,
    budget_id: &str,
) -> Result<Option<Budget>, Error> {
    Ok(
        sqlx::query_file_as!(Budget, "src/budget/queries/find_budget.sql", budget_id)
            .fetch_optional(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::find_budget;
    use crate::test_support::{database_pool, insert_budget};

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_every_column(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_budget(&mut tx, "budget-store-all-columns").await?;

        let budget = find_budget(&mut *tx, "budget-store-all-columns")
            .await?
            .ok_or("budget row missing")?;

        assert_eq!(
            serde_json::to_value(&budget)?,
            serde_json::json!({
                "budget_id": "budget-store-all-columns",
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
        );
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn missing_budget_is_none(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;

        assert!(
            find_budget(&mut *tx, "budget-store-missing")
                .await?
                .is_none()
        );
        Ok(())
    }
}

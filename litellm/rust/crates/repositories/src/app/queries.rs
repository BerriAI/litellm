//! Reads one `LiteLLM_AppTable` row, as the spend counter reseed sees it.

use chrono::NaiveDateTime;
use sqlx::PgExecutor;

use litellm_db::Error;

use crate::utc_timestamp;

/// Every scalar column of one `LiteLLM_AppTable` row
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct App {
    app_id: String,
    app_group_id: Option<String>,
    resource_type: Option<String>,
    resource_id: Option<String>,
    alias: String,
    spend: f64,
    spend_reset_window: Option<String>,
    budget_id: Option<String>,
    default_tier_budget_id: Option<String>,
    default_customer_member_budget_id: Option<String>,
    auto_create_customers: bool,
    customer_margin: Option<f64>,
    #[serde(with = "utc_timestamp")]
    created_at: NaiveDateTime,
    created_by: String,
    #[serde(with = "utc_timestamp")]
    updated_at: NaiveDateTime,
    updated_by: String,
}

pub(super) async fn find_app(
    executor: impl PgExecutor<'_>,
    app_id: &str,
) -> Result<Option<App>, Error> {
    Ok(
        sqlx::query_file_as!(App, "src/app/queries/find_app.sql", app_id)
            .fetch_optional(executor)
            .await?,
    )
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::find_app;
    use crate::test_support::{database_pool, insert_app};

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_every_column(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_app(&mut tx, "app-1").await?;

        let found = find_app(&mut *tx, "app-1").await?.ok_or("row missing")?;

        assert_eq!(
            serde_json::to_value(&found)?,
            serde_json::json!({
                "app_id": "app-1",
                "app_group_id": null,
                "resource_type": "team",
                "resource_id": "app-1",
                "alias": "app-1",
                "spend": 6.5,
                "spend_reset_window": null,
                "budget_id": null,
                "default_tier_budget_id": null,
                "default_customer_member_budget_id": null,
                "auto_create_customers": false,
                "customer_margin": 0.2,
                "created_at": "2026-01-02T03:04:05.678+00:00",
                "created_by": "creator",
                "updated_at": "2026-02-03T04:05:06.789+00:00",
                "updated_by": "updater",
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
        assert!(find_app(&pool, "app-missing").await?.is_none());
        Ok(())
    }
}

//! Reads one `LiteLLM_BudgetWindowSpend` row, the spend the proxy has recorded against one budget window.

use chrono::NaiveDateTime;
use sqlx::PgExecutor;

use litellm_db::Error;

use crate::utc_timestamp;

/// When a budget window started and the spend recorded in it so far.
#[macro_rules_attribute::apply(crate::db_row)]
#[derive(serde::Serialize)]
pub struct BudgetWindowSpend {
    #[serde(with = "utc_timestamp")]
    window_start: NaiveDateTime,
    spend: f64,
}

pub(super) async fn find_window_spend(
    executor: impl PgExecutor<'_>,
    entity_type: &str,
    entity_id: &str,
    window_duration: &str,
) -> Result<Option<BudgetWindowSpend>, Error> {
    Ok(sqlx::query_file_as!(
        BudgetWindowSpend,
        "src/budget_window_spend/queries/find_window_spend.sql",
        entity_type,
        entity_id,
        window_duration,
    )
    .fetch_optional(executor)
    .await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::PgPool;

    use super::find_window_spend;
    use crate::test_support::database_pool;

    #[rstest]
    #[case::exact_key("key", "tok-window", "30d", true)]
    #[case::other_duration("key", "tok-window", "1d", false)]
    #[case::other_entity_type("team", "tok-window", "30d", false)]
    #[case::other_entity_id("key", "tok-other", "30d", false)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn reads_only_the_row_at_its_primary_key(
        #[case] entity_type: &str,
        #[case] entity_id: &str,
        #[case] window_duration: &str,
        #[case] found: bool,
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
            INSERT INTO "LiteLLM_BudgetWindowSpend" (entity_type, entity_id, window_duration, window_start, spend, updated_at)
            VALUES ('key', 'tok-window', '30d', '2026-11-01 09:30:00.250', 4.5, now())
            "#,
        )
        .execute(&mut *tx)
        .await?;

        let window = find_window_spend(&mut *tx, entity_type, entity_id, window_duration).await?;

        assert_eq!(
            window
                .map(|window| serde_json::to_value(&window))
                .transpose()?,
            found.then(|| {
                serde_json::json!({"window_start": "2026-11-01T09:30:00.250+00:00", "spend": 4.5})
            })
        );
        Ok(())
    }
}

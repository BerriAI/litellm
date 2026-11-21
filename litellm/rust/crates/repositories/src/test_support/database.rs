use std::error::Error as StdError;

use litellm_db::Statements;
use rstest::fixture;
use sqlx::{PgConnection, PgPool, Postgres, Transaction};

use crate::{Side, database::Connections};

#[fixture]
pub async fn database_pool() -> Result<PgPool, Box<dyn StdError>> {
    let database_url = std::env::var("DATABASE_URL")?;
    Ok(PgPool::connect(&database_url).await?)
}

/// Every step of an operation on one test transaction
pub(crate) struct OneTransaction<'t, 'c>(pub &'t mut Transaction<'c, Postgres>);

impl Connections for OneTransaction<'_, '_> {
    async fn with<T>(
        &mut self,
        _side: Side,
        step: impl for<'e> AsyncFnOnce(Statements<&'e mut PgConnection>) -> Result<T, litellm_db::Error>,
    ) -> Result<T, litellm_db::Error> {
        step(Statements::new(&mut **self.0, false)).await
    }
}

/// Moves the transaction's session off UTC, so a write that stamps session-local time shows up as hours off
pub(crate) async fn leave_utc(tx: &mut Transaction<'_, Postgres>) -> Result<(), sqlx::Error> {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture statements are outside the compile-checked query set"
    )]
    let result = sqlx::query("SET LOCAL TIME ZONE 'Pacific/Kiritimati'")
        .execute(&mut **tx)
        .await;
    result.map(|_| ())
}

pub(crate) async fn insert_budget(
    tx: &mut Transaction<'_, Postgres>,
    budget_id: &str,
) -> Result<(), sqlx::Error> {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture inserts are outside the compile-checked query set"
    )]
    let result = sqlx::query(
        r#"
        INSERT INTO "LiteLLM_BudgetTable" (
            budget_id, name, max_budget, soft_budget, max_parallel_requests, tpm_limit, rpm_limit,
            tpd_limit, model_max_budget, model_tpm_limit, model_rpm_limit, budget_duration,
            budget_reset_at, budget_reset_count, allowed_models, temp_budget_increase,
            temp_budget_expiry, created_at, created_by, updated_at, updated_by
        )
        VALUES (
            $1, 'team default', 10.5, 8.0, 3, 1000, 10, 50000,
            '{"gpt-4o": {"max_budget": 1.0}}', '{"gpt-4o": 100}', '{"gpt-4o": 5}', '30d',
            '2026-12-01 00:00:00', 2, ARRAY['gpt-4o'], 1.5, '2026-11-20 12:30:00.250',
            '2026-11-01 00:00:00', 'db-query-test', '2026-11-02 00:00:00', 'db-query-test'
        )
        "#,
    )
    .bind(budget_id)
    .execute(&mut **tx)
    .await;
    result.map(|_| ())
}

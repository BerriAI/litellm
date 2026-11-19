use std::error::Error as StdError;

use rstest::fixture;
use sqlx::{PgPool, Postgres, Transaction};

#[fixture]
pub async fn database_pool() -> Result<PgPool, Box<dyn StdError>> {
    let database_url = std::env::var("DATABASE_URL")?;
    Ok(PgPool::connect(&database_url).await?)
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

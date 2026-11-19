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

pub(crate) async fn insert_customer_organization(
    tx: &mut Transaction<'_, Postgres>,
    customer_org_id: &str,
    budget_id: Option<&str>,
) -> Result<(), sqlx::Error> {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture inserts are outside the compile-checked query set"
    )]
    let result = sqlx::query(
        r#"
        INSERT INTO "LiteLLM_CustomerOrganizationTable"
            (customer_org_id, alias, spend, budget_template_ids, budget_id, created_at, created_by, updated_at,
             updated_by)
        VALUES ($1, 'acme', 2.5, '{template-1}', $2, '2026-01-02 03:04:05.678', 'creator',
                '2026-02-03 04:05:06.789', 'updater')
        "#,
    )
    .bind(customer_org_id)
    .bind(budget_id)
    .execute(&mut **tx)
    .await;
    result.map(|_| ())
}

pub(crate) async fn insert_customer_member(
    tx: &mut Transaction<'_, Postgres>,
    customer_member_id: &str,
    parent_org_id: &str,
    budget_id: Option<&str>,
    budget_override_id: Option<&str>,
) -> Result<(), sqlx::Error> {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture inserts are outside the compile-checked query set"
    )]
    let result = sqlx::query(
        r#"
        INSERT INTO "LiteLLM_CustomerMemberTable"
            (customer_member_id, alias, parent_org_id, parent_team_ids, spend, spend_reset_window, budget_id,
             budget_override_id, budget_override_expires_at, created_at, created_by, updated_at, updated_by)
        VALUES ($1, 'jane', $2, '{team-a,team-b}', 1.25, '1mo', $3, $4, '2026-12-24 00:00:00',
                '2026-01-02 03:04:05.678', 'creator', '2026-02-03 04:05:06.789', 'updater')
        "#,
    )
    .bind(customer_member_id)
    .bind(parent_org_id)
    .bind(budget_id)
    .bind(budget_override_id)
    .execute(&mut **tx)
    .await;
    result.map(|_| ())
}

pub(crate) async fn insert_app(
    tx: &mut Transaction<'_, Postgres>,
    app_id: &str,
) -> Result<(), sqlx::Error> {
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture inserts are outside the compile-checked query set"
    )]
    let result = sqlx::query(
        r#"
        INSERT INTO "LiteLLM_AppTable"
            (app_id, resource_type, resource_id, alias, spend, auto_create_customers, customer_margin, created_at,
             created_by, updated_at, updated_by)
        VALUES ($1, 'team', $1, $1, 6.5, false, 0.2, '2026-01-02 03:04:05.678', 'creator',
                '2026-02-03 04:05:06.789', 'updater')
        "#,
    )
    .bind(app_id)
    .execute(&mut **tx)
    .await;
    result.map(|_| ())
}

/// The JSON the `insert_budget` fixture row serializes to, as rows that nest it write it
pub(crate) fn inserted_budget_json(budget_id: &str) -> serde_json::Value {
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
}

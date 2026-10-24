use std::error::Error as StdError;

use litellm_db::{EntitySpend, add_end_user_spend};
use rstest::rstest;
use sqlx::PgPool;

mod common;
use common::database_pool;

#[rstest]
#[case::fractional_costs(0.25, 0.75, 2.5, 1.5)]
#[case::zero_costs(0.0, 0.0, 2.0, 0.0)]
#[tokio::test]
async fn end_user_spend_matches_python_upsert(
    #[case] existing_cost: f64,
    #[case] new_cost: f64,
    #[case] expected_existing_spend: f64,
    #[case] expected_new_spend: f64,
    #[future(awt)] database_pool: Result<Option<PgPool>, sqlx::Error>,
) -> Result<(), Box<dyn StdError>> {
    let Some(pool) = database_pool? else {
        return Ok(());
    };
    let mut tx = pool.begin().await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture writes are outside the compile-checked query set"
    )]
    let _budget = sqlx::query(
        r#"INSERT INTO "LiteLLM_BudgetTable" (budget_id, created_by, updated_by)
           VALUES ('end-user-spend-budget', 'db-query-test', 'db-query-test')"#,
    )
    .execute(&mut *tx)
    .await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture writes are outside the compile-checked query set"
    )]
    let _users = sqlx::query(
        r#"INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked, budget_id)
           VALUES ('end-user-spend-z', 2, true, 'end-user-spend-budget'),
                  ('end-user-spend-untouched', 100, true, NULL)"#,
    )
    .execute(&mut *tx)
    .await?;
    let spend = EntitySpend {
        ids: vec!["end-user-spend-z".into(), "end-user-spend-a".into()],
        costs: vec![existing_cost, new_cost],
    };
    add_end_user_spend(&mut *tx, &spend).await?;
    add_end_user_spend(&mut *tx, &spend).await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Test assertions are outside the compile-checked query set"
    )]
    let rows: Vec<(String, f64, bool, Option<String>)> = sqlx::query_as(
        r#"SELECT user_id, spend, blocked, budget_id FROM "LiteLLM_EndUserTable"
           WHERE user_id = ANY($1) ORDER BY user_id"#,
    )
    .bind(vec![
        "end-user-spend-a",
        "end-user-spend-z",
        "end-user-spend-untouched",
    ])
    .fetch_all(&mut *tx)
    .await?;
    assert_eq!(
        rows,
        vec![
            ("end-user-spend-a".into(), expected_new_spend, false, None),
            ("end-user-spend-untouched".into(), 100.0, true, None),
            (
                "end-user-spend-z".into(),
                expected_existing_spend,
                true,
                Some("end-user-spend-budget".into()),
            ),
        ]
    );
    Ok(())
}

#[rstest]
#[case::empty_batch()]
#[tokio::test]
async fn empty_end_user_spend_is_a_noop(
    #[future(awt)] database_pool: Result<Option<PgPool>, sqlx::Error>,
) -> Result<(), Box<dyn StdError>> {
    let Some(pool) = database_pool? else {
        return Ok(());
    };
    let mut tx = pool.begin().await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Fixture writes are outside the compile-checked query set"
    )]
    let _user = sqlx::query(
        r#"INSERT INTO "LiteLLM_EndUserTable" (user_id, spend, blocked)
           VALUES ('end-user-empty-spend', 8.5, true)"#,
    )
    .execute(&mut *tx)
    .await?;
    add_end_user_spend(&mut *tx, &EntitySpend::default()).await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Test assertions are outside the compile-checked query set"
    )]
    let row: (f64, bool, Option<String>) = sqlx::query_as(
        r#"SELECT spend, blocked, budget_id FROM "LiteLLM_EndUserTable"
           WHERE user_id = 'end-user-empty-spend'"#,
    )
    .fetch_one(&mut *tx)
    .await?;
    assert_eq!(row, (8.5, true, None));
    Ok(())
}

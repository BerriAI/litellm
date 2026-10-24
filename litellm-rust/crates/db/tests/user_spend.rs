use std::error::Error as StdError;

use litellm_db::{EntitySpend, add_user_spend};
use rstest::rstest;
use sqlx::PgPool;

mod common;
use common::database_pool;

#[rstest]
#[case::fractional_costs(0.25, 2.5, 4.0)]
#[case::zero_costs(0.0, 2.0, 3.0)]
#[tokio::test]
async fn user_spend_matches_python_update_many(
    #[case] cost: f64,
    #[case] expected_a_spend: f64,
    #[case] expected_z_spend: f64,
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
    let _inserted = sqlx::query(
        r#"INSERT INTO "LiteLLM_UserTable" (user_id, spend, updated_at)
           VALUES ('user-spend-a', 2, '2000-01-01'),
                  ('user-spend-z', 3, '2000-01-01'),
                  ('user-spend-untouched', 100, '2000-01-01')"#,
    )
    .execute(&mut *tx)
    .await?;
    let spend = EntitySpend {
        ids: vec![
            "user-spend-z".into(),
            "user-spend-a".into(),
            "user-spend-missing".into(),
        ],
        costs: vec![2.0 * cost, cost, 50.0],
    };
    add_user_spend(&mut *tx, &EntitySpend::default()).await?;
    add_user_spend(&mut *tx, &spend).await?;
    add_user_spend(&mut *tx, &spend).await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Test assertions are outside the compile-checked query set"
    )]
    let rows: Vec<(String, f64, bool)> = sqlx::query_as(
        r#"SELECT user_id, spend, updated_at > TIMESTAMP '2000-01-01'
           FROM "LiteLLM_UserTable"
           WHERE user_id = ANY($1) ORDER BY user_id"#,
    )
    .bind(vec![
        "user-spend-a",
        "user-spend-z",
        "user-spend-missing",
        "user-spend-untouched",
    ])
    .fetch_all(&mut *tx)
    .await?;
    assert_eq!(
        rows,
        vec![
            ("user-spend-a".into(), expected_a_spend, true),
            ("user-spend-untouched".into(), 100.0, false),
            ("user-spend-z".into(), expected_z_spend, true),
        ]
    );
    Ok(())
}

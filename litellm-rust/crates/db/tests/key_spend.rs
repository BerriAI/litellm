use std::error::Error as StdError;

use litellm_db::{EntitySpend, add_key_spend};
use rstest::rstest;
use sqlx::PgPool;

mod common;
use common::database_pool;

#[rstest]
#[case::fractional_costs(0.25, 2.5, 7.5, 4.0, 12.0)]
#[case::zero_costs(0.0, 2.0, 7.0, 3.0, 11.0)]
#[tokio::test]
async fn key_spend_matches_python_update_many(
    #[case] cost: f64,
    #[case] expected_a_spend: f64,
    #[case] expected_a_total: f64,
    #[case] expected_z_spend: f64,
    #[case] expected_z_total: f64,
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
        r#"INSERT INTO "LiteLLM_VerificationToken"
            (token, spend, total_spend, last_active, updated_at)
           VALUES ('key-spend-a', 2, 7, NULL, '2000-01-01'),
                  ('key-spend-z', 3, 11, NULL, '2000-01-01'),
                  ('key-spend-untouched', 100, 200, NULL, '2000-01-01')"#,
    )
    .execute(&mut *tx)
    .await?;
    let spend = EntitySpend {
        ids: vec![
            "key-spend-z".into(),
            "key-spend-a".into(),
            "key-spend-missing".into(),
        ],
        costs: vec![2.0 * cost, cost, 50.0],
    };
    add_key_spend(&mut *tx, &EntitySpend::default()).await?;
    add_key_spend(&mut *tx, &spend).await?;
    add_key_spend(&mut *tx, &spend).await?;
    #[expect(
        clippy::disallowed_methods,
        reason = "Test assertions are outside the compile-checked query set"
    )]
    let rows: Vec<(String, f64, f64, bool, bool)> = sqlx::query_as(
        r#"SELECT token, spend, total_spend, last_active IS NOT NULL,
                  updated_at > TIMESTAMP '2000-01-01'
           FROM "LiteLLM_VerificationToken"
           WHERE token = ANY($1) ORDER BY token"#,
    )
    .bind(vec![
        "key-spend-a",
        "key-spend-z",
        "key-spend-missing",
        "key-spend-untouched",
    ])
    .fetch_all(&mut *tx)
    .await?;
    assert_eq!(
        rows,
        vec![
            (
                "key-spend-a".into(),
                expected_a_spend,
                expected_a_total,
                true,
                true
            ),
            ("key-spend-untouched".into(), 100.0, 200.0, false, false),
            (
                "key-spend-z".into(),
                expected_z_spend,
                expected_z_total,
                true,
                true
            ),
        ]
    );
    Ok(())
}

use sqlx::PgExecutor;

use super::EntitySpend;
use crate::Error;

pub async fn add_team_spend(
    executor: impl PgExecutor<'_>,
    spend: &EntitySpend,
) -> Result<(), Error> {
    spend.validate()?;
    if spend.is_empty() {
        return Ok(());
    }
    sqlx::query_file!("queries/spend/add_team_spend.sql", &spend.ids, &spend.costs)
        .execute(executor)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{EntitySpend, add_team_spend};
    use crate::Error;
    use crate::test_support::database_pool;
    use rstest::rstest;
    use sqlx::PgPool;

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn duplicate_team_ids_add_every_cost(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _inserted = sqlx::query(
            r#"INSERT INTO "LiteLLM_TeamTable" (team_id, spend, created_at, updated_at)
               VALUES ('team-spend-duplicate-a', 2, '2000-01-01', '2000-01-01'),
                      ('team-spend-duplicate-z', 3, '2000-01-01', '2000-01-01')"#,
        )
        .execute(&mut *tx)
        .await?;
        let spend = EntitySpend {
            ids: vec![
                "team-spend-duplicate-z".into(),
                "team-spend-duplicate-a".into(),
                "team-spend-duplicate-z".into(),
                "team-spend-duplicate-a".into(),
            ],
            costs: vec![0.5, 0.25, 1.0, 0.5],
        };
        add_team_spend(&mut *tx, &spend).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let rows: Vec<(String, f64)> = sqlx::query_as(
            r#"SELECT team_id, spend FROM "LiteLLM_TeamTable"
               WHERE team_id = ANY($1) ORDER BY team_id"#,
        )
        .bind(&spend.ids)
        .fetch_all(&mut *tx)
        .await?;
        assert_eq!(
            rows,
            vec![
                ("team-spend-duplicate-a".into(), 2.75),
                ("team-spend-duplicate-z".into(), 4.5),
            ]
        );
        Ok(())
    }

    #[rstest]
    #[case::missing_cost(vec!["team-spend-malformed".into()], vec![])]
    #[case::extra_cost(vec!["team-spend-malformed".into()], vec![0.25, 0.5])]
    #[case::empty_ids(vec![], vec![0.25])]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn malformed_team_batch_does_not_write(
        #[case] ids: Vec<String>,
        #[case] costs: Vec<f64>,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _inserted = sqlx::query(
            r#"INSERT INTO "LiteLLM_TeamTable" (team_id, spend, created_at, updated_at)
               VALUES ('team-spend-malformed', 2, '2000-01-01', '2000-01-01')"#,
        )
        .execute(&mut *tx)
        .await?;
        let result = add_team_spend(&mut *tx, &EntitySpend { ids, costs }).await;
        assert!(matches!(result, Err(Error::InvalidSpendBatch { .. })));
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let row: (f64, bool) = sqlx::query_as(
            r#"SELECT spend, updated_at = TIMESTAMP '2000-01-01'
               FROM "LiteLLM_TeamTable" WHERE team_id = 'team-spend-malformed'"#,
        )
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(row, (2.0, true));
        Ok(())
    }

    #[rstest]
    #[case::fractional_costs(0.25, 2.5, 4.0)]
    #[case::zero_costs(0.0, 2.0, 3.0)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn team_spend_matches_python_update_many(
        #[case] cost: f64,
        #[case] expected_a_spend: f64,
        #[case] expected_z_spend: f64,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        let _inserted = sqlx::query(
            r#"INSERT INTO "LiteLLM_TeamTable" (team_id, spend, created_at, updated_at)
               VALUES ('team-spend-a', 2, '2000-01-01', '2000-01-01'),
                      ('team-spend-z', 3, '2000-01-01', '2000-01-01'),
                      ('team-spend-untouched', 100, '2000-01-01', '2000-01-01')"#,
        )
        .execute(&mut *tx)
        .await?;
        let spend = EntitySpend {
            ids: vec![
                "team-spend-z".into(),
                "team-spend-a".into(),
                "team-spend-missing".into(),
            ],
            costs: vec![2.0 * cost, cost, 50.0],
        };
        add_team_spend(&mut *tx, &EntitySpend::default()).await?;
        add_team_spend(&mut *tx, &spend).await?;
        add_team_spend(&mut *tx, &spend).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let rows: Vec<(String, f64, bool)> = sqlx::query_as(
            r#"SELECT team_id, spend, updated_at > TIMESTAMP '2000-01-01'
               FROM "LiteLLM_TeamTable"
               WHERE team_id = ANY($1) ORDER BY team_id"#,
        )
        .bind(vec![
            "team-spend-a",
            "team-spend-z",
            "team-spend-missing",
            "team-spend-untouched",
        ])
        .fetch_all(&mut *tx)
        .await?;
        assert_eq!(
            rows,
            vec![
                ("team-spend-a".into(), expected_a_spend, true),
                ("team-spend-untouched".into(), 100.0, false),
                ("team-spend-z".into(), expected_z_spend, true),
            ]
        );
        Ok(())
    }
}

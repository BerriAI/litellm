use sqlx::PgExecutor;
use std::collections::BTreeMap;

use crate::Error;

#[macro_rules_attribute::apply(crate::db_input)]
#[derive(Default)]
pub struct EntitySpend {
    pub ids: Vec<String>,
    pub costs: Vec<f64>,
}

impl EntitySpend {
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.ids.len() != self.costs.len() {
            return Err(Error::InvalidSpendBatch {
                ids: self.ids.len(),
                costs: self.costs.len(),
            });
        }
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

impl FromIterator<(String, f64)> for EntitySpend {
    fn from_iter<I: IntoIterator<Item = (String, f64)>>(entries: I) -> Self {
        let entries = entries
            .into_iter()
            .fold(BTreeMap::new(), |mut sums, (id, cost)| {
                *sums.entry(id).or_insert(0.0) += cost;
                sums
            });
        let (ids, costs) = entries.into_iter().unzip();
        Self { ids, costs }
    }
}

pub async fn add_key_spend(
    executor: impl PgExecutor<'_>,
    spend: &EntitySpend,
) -> Result<(), Error> {
    spend.validate()?;
    if spend.is_empty() {
        return Ok(());
    }
    sqlx::query_file!("queries/spend/add_key_spend.sql", &spend.ids, &spend.costs)
        .execute(executor)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use super::{EntitySpend, add_key_spend};
    use crate::{Error, test_support::database_pool};
    use rstest::rstest;
    use sqlx::PgPool;

    #[rstest]
    #[case::direct(false)]
    #[case::collected(true)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn duplicate_key_spend_accumulates(
        #[case] collected: bool,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"INSERT INTO "LiteLLM_VerificationToken" (token, spend, total_spend, updated_at)
               VALUES ('key-spend-duplicate', 2, 7, '2000-01-01')"#,
        )
        .execute(&mut *tx)
        .await?;
        let spend = if collected {
            [
                ("key-spend-duplicate".into(), 0.25),
                ("key-spend-duplicate".into(), 0.75),
            ]
            .into_iter()
            .collect()
        } else {
            EntitySpend {
                ids: vec!["key-spend-duplicate".into(), "key-spend-duplicate".into()],
                costs: vec![0.25, 0.75],
            }
        };
        add_key_spend(&mut *tx, &spend).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let row: (f64, f64) = sqlx::query_as(
            r#"SELECT spend, total_spend FROM "LiteLLM_VerificationToken"
               WHERE token = 'key-spend-duplicate'"#,
        )
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(row, (3.0, 8.0));
        Ok(())
    }

    #[rstest]
    #[case::extra_id(vec!["key-spend-invalid".into(), "key-spend-invalid".into()], vec![0.25])]
    #[case::extra_cost(vec!["key-spend-invalid".into()], vec![0.25, 0.75])]
    #[case::empty_ids(vec![], vec![0.25])]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn malformed_key_spend_leaves_transaction_usable(
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
        sqlx::query(
            r#"INSERT INTO "LiteLLM_VerificationToken"
                   (token, spend, total_spend, last_active, updated_at)
               VALUES ('key-spend-invalid', 2, 7, NULL, '2000-01-01')"#,
        )
        .execute(&mut *tx)
        .await?;
        let spend = EntitySpend { ids, costs };
        let result = add_key_spend(&mut *tx, &spend).await;
        assert!(matches!(
            result,
            Err(Error::InvalidSpendBatch { ids, costs })
                if ids == spend.ids.len() && costs == spend.costs.len()
        ));
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let unchanged: (f64, f64, bool, bool) = sqlx::query_as(
            r#"SELECT spend, total_spend, last_active IS NULL,
                      updated_at = TIMESTAMP '2000-01-01'
               FROM "LiteLLM_VerificationToken" WHERE token = 'key-spend-invalid'"#,
        )
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(unchanged, (2.0, 7.0, true, true));
        let valid = EntitySpend {
            ids: vec!["key-spend-invalid".into()],
            costs: vec![0.5],
        };
        add_key_spend(&mut *tx, &valid).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let updated: (f64, f64) = sqlx::query_as(
            r#"SELECT spend, total_spend FROM "LiteLLM_VerificationToken"
               WHERE token = 'key-spend-invalid'"#,
        )
        .fetch_one(&mut *tx)
        .await?;
        assert_eq!(updated, (2.5, 7.5));
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn key_activity_uses_update_time_in_old_transaction(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture writes are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"INSERT INTO "LiteLLM_VerificationToken" (token, spend, total_spend, updated_at)
               VALUES ('key-spend-old-transaction', 2, 7, '2000-01-01')"#,
        )
        .execute(&mut *tx)
        .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Transaction timing setup is outside the compile-checked query set"
        )]
        sqlx::query("SELECT pg_sleep(0.02)")
            .execute(&mut *tx)
            .await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let boundary: chrono::NaiveDateTime = sqlx::query_scalar(
            "SELECT date_trunc('milliseconds', clock_timestamp() AT TIME ZONE 'UTC')",
        )
        .fetch_one(&mut *tx)
        .await?;
        let spend = EntitySpend {
            ids: vec!["key-spend-old-transaction".into()],
            costs: vec![0.25],
        };
        add_key_spend(&mut *tx, &spend).await?;
        #[expect(
            clippy::disallowed_methods,
            reason = "Test assertions are outside the compile-checked query set"
        )]
        let row: (chrono::NaiveDateTime, chrono::NaiveDateTime) = sqlx::query_as(
            r#"SELECT last_active, updated_at FROM "LiteLLM_VerificationToken"
               WHERE token = 'key-spend-old-transaction'"#,
        )
        .fetch_one(&mut *tx)
        .await?;
        assert!(row.0 >= boundary, "last_active predates update boundary");
        assert!(row.1 >= boundary, "updated_at predates update boundary");
        Ok(())
    }

    #[rstest]
    #[case::fractional_costs(0.25, 2.5, 7.5, 4.0, 12.0)]
    #[case::zero_costs(0.0, 2.0, 7.0, 3.0, 11.0)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn key_spend_matches_python_update_many(
        #[case] cost: f64,
        #[case] expected_a_spend: f64,
        #[case] expected_a_total: f64,
        #[case] expected_z_spend: f64,
        #[case] expected_z_total: f64,
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
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
}

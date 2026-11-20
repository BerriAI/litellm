//! Sums of `LiteLLM_SpendLogs.spend` for one key, team or user since a window start

use chrono::NaiveDateTime;
use sqlx::PgExecutor;

use litellm_db::Error;

use super::SpendLogOwner;

macro_rules! window_spend_sql {
    ($column:literal) => {
        concat!(
            r#"SELECT COALESCE(SUM(spend), 0) FROM "LiteLLM_SpendLogs" WHERE "#,
            $column,
            r#" = $1 AND "startTime" >= $2"#
        )
    };
}

pub(super) async fn window_spend(
    executor: impl PgExecutor<'_>,
    owner: SpendLogOwner,
    owner_id: &str,
    since: NaiveDateTime,
) -> Result<f64, Error> {
    let sql = match owner {
        SpendLogOwner::Key => window_spend_sql!("api_key"),
        SpendLogOwner::Team => window_spend_sql!("team_id"),
        SpendLogOwner::User => window_spend_sql!(r#""user""#),
    };
    #[expect(
        clippy::disallowed_methods,
        reason = "The column comes from matching SpendLogOwner, so the SQL is one of three static strings"
    )]
    let query = sqlx::query_scalar::<_, f64>(sql);
    Ok(query.bind(owner_id).bind(since).fetch_one(executor).await?)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use chrono::NaiveDateTime;
    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::{SpendLogOwner, window_spend};
    use crate::test_support::database_pool;

    /// One log owned by `owner` in the column under test, with `other` in the other two owner columns. The column
    /// goes into the request id so the cases, which run at once, never wait on each other's primary key.
    async fn insert_spend_log(
        tx: &mut Transaction<'_, Postgres>,
        column: SpendLogOwner,
        request_id: &str,
        owner: &str,
        other: &str,
        spend: f64,
        start_time: &str,
    ) -> Result<(), sqlx::Error> {
        let (api_key, team_id, user) = match column {
            SpendLogOwner::Key => (owner, other, other),
            SpendLogOwner::Team => (other, owner, other),
            SpendLogOwner::User => (other, other, owner),
        };
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        let result = sqlx::query(
            r#"
            INSERT INTO "LiteLLM_SpendLogs"
                (request_id, call_type, api_key, team_id, "user", spend, "startTime", "endTime")
            VALUES ($1, 'acompletion', $2, $3, $4, $5, $6::timestamp, $6::timestamp)
            "#,
        )
        .bind(format!("{request_id}-{column:?}"))
        .bind(api_key)
        .bind(team_id)
        .bind(user)
        .bind(spend)
        .bind(start_time)
        .execute(&mut **tx)
        .await;
        result.map(|_| ())
    }

    fn window_start() -> Result<NaiveDateTime, chrono::ParseError> {
        NaiveDateTime::parse_from_str("2026-08-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    }

    #[rstest]
    #[case::key(SpendLogOwner::Key)]
    #[case::team(SpendLogOwner::Team)]
    #[case::user(SpendLogOwner::User)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn sums_only_the_owners_column_from_the_window_start_on(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
        #[case] column: SpendLogOwner,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        for (request_id, owner, other, spend, start_time) in [
            (
                "window-before",
                "window-owner",
                "window-x",
                100.0,
                "2026-07-31 23:59:59.999",
            ),
            (
                "window-at-start",
                "window-owner",
                "window-x",
                1.25,
                "2026-08-01 00:00:00",
            ),
            (
                "window-inside",
                "window-owner",
                "window-x",
                2.5,
                "2026-08-03 12:00:00",
            ),
            (
                "window-other-column",
                "window-x",
                "window-owner",
                50.0,
                "2026-08-02 00:00:00",
            ),
        ] {
            insert_spend_log(&mut tx, column, request_id, owner, other, spend, start_time).await?;
        }

        assert_eq!(
            window_spend(&mut *tx, column, "window-owner", window_start()?).await?,
            3.75
        );
        Ok(())
    }

    #[rstest]
    #[case::key(SpendLogOwner::Key)]
    #[case::team(SpendLogOwner::Team)]
    #[case::user(SpendLogOwner::User)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn no_rows_in_the_window_is_zero(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
        #[case] column: SpendLogOwner,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        insert_spend_log(
            &mut tx,
            column,
            "window-old",
            "window-quiet-owner",
            "window-x",
            9.0,
            "2026-07-01 00:00:00",
        )
        .await?;
        let since = window_start()?;

        assert_eq!(
            window_spend(&mut *tx, column, "window-quiet-owner", since).await?,
            0.0
        );
        assert_eq!(
            window_spend(&mut *tx, column, "window-missing-owner", since).await?,
            0.0
        );
        Ok(())
    }
}

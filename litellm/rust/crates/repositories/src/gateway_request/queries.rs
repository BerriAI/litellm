//! The one statement that adds a flush's gateway request counts onto `LiteLLM_DailyGatewayRequests`
//! and `LiteLLM_DailyGatewayFailedRequests`, mirroring `build_gateway_requests_upsert`

use sqlx::PgExecutor;

use litellm_db::Error;

/// The failure table's rows, one column per field, in the order they lock in.
#[derive(Debug, Default, PartialEq)]
pub(super) struct FailedColumns {
    pub(super) dates: Vec<String>,
    pub(super) categories: Vec<String>,
    pub(super) routes: Vec<String>,
    pub(super) status_codes: Vec<i32>,
    pub(super) failed_requests: Vec<i64>,
}

/// The totals table's rows, one column per field, in the order they lock in.
#[derive(Debug, Default, PartialEq)]
pub(super) struct TotalColumns {
    pub(super) dates: Vec<String>,
    pub(super) categories: Vec<String>,
    pub(super) routes: Vec<String>,
    pub(super) successful_requests: Vec<i64>,
    pub(super) failed_requests: Vec<i64>,
}

/// Adds `failed` onto the failure table by status code and `totals` onto the totals table, in one
/// statement that locks the rows in the order given.
pub(super) async fn increment_gateway_requests(
    executor: impl PgExecutor<'_>,
    failed: &FailedColumns,
    totals: &TotalColumns,
) -> Result<(), Error> {
    sqlx::query_file!(
        "src/gateway_request/queries/increment_gateway_requests.sql",
        &failed.dates,
        &failed.categories,
        &failed.routes,
        &failed.status_codes,
        &failed.failed_requests,
        &totals.dates,
        &totals.categories,
        &totals.routes,
        &totals.successful_requests,
        &totals.failed_requests,
    )
    .execute(executor)
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{PgPool, Postgres, Transaction};

    use super::{FailedColumns, TotalColumns, increment_gateway_requests};
    use crate::test_support::{database_pool, leave_utc};

    const DATE: &str = "2099-01-01";

    fn totals(successful: i64, failed: i64) -> TotalColumns {
        TotalColumns {
            dates: vec![DATE.to_owned()],
            categories: vec!["llm".to_owned()],
            routes: vec!["/v1/messages".to_owned()],
            successful_requests: vec![successful],
            failed_requests: vec![failed],
        }
    }

    fn failed(status_code: i32, count: i64) -> FailedColumns {
        FailedColumns {
            dates: vec![DATE.to_owned()],
            categories: vec!["llm".to_owned()],
            routes: vec!["/v1/messages".to_owned()],
            status_codes: vec![status_code],
            failed_requests: vec![count],
        }
    }

    async fn stored_totals(
        tx: &mut Transaction<'_, Postgres>,
    ) -> Result<(i64, i64, bool), sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture reads are outside the compile-checked query set"
        )]
        let row = sqlx::query_as::<_, (i64, i64, bool)>(
            r#"SELECT successful_requests, failed_requests,
                      ABS(EXTRACT(EPOCH FROM (updated_at - (NOW() AT TIME ZONE 'UTC')))) < 3600
               FROM "LiteLLM_DailyGatewayRequests"
               WHERE date = $1 AND category = 'llm' AND route = '/v1/messages'"#,
        )
        .bind(DATE)
        .fetch_one(&mut **tx)
        .await;
        row
    }

    async fn stored_failures(
        tx: &mut Transaction<'_, Postgres>,
    ) -> Result<Vec<(i32, i64)>, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture reads are outside the compile-checked query set"
        )]
        let rows = sqlx::query_as::<_, (i32, i64)>(
            r#"SELECT status_code, failed_requests FROM "LiteLLM_DailyGatewayFailedRequests"
               WHERE date = $1 AND category = 'llm' AND route = '/v1/messages'
               ORDER BY status_code"#,
        )
        .bind(DATE)
        .fetch_all(&mut **tx)
        .await;
        rows
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn counts_add_onto_the_rows_already_stored_in_utc(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;
        leave_utc(&mut tx).await?;

        increment_gateway_requests(&mut *tx, &failed(500, 1), &totals(2, 1)).await?;
        increment_gateway_requests(&mut *tx, &failed(429, 2), &totals(3, 2)).await?;
        increment_gateway_requests(&mut *tx, &failed(500, 4), &totals(0, 4)).await?;

        assert_eq!(stored_totals(&mut tx).await?, (5, 7, true));
        assert_eq!(stored_failures(&mut tx).await?, vec![(429, 2), (500, 5)]);
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn totals_with_no_failures_leave_the_failure_table_alone(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut tx = pool.begin().await?;

        increment_gateway_requests(&mut *tx, &FailedColumns::default(), &totals(4, 0)).await?;

        assert_eq!(stored_totals(&mut tx).await?, (4, 0, true));
        assert_eq!(stored_failures(&mut tx).await?, vec![]);
        Ok(())
    }
}

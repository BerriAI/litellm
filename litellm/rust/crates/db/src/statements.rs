use futures_util::{future::BoxFuture, stream::BoxStream};
use sqlx::{
    Describe, Either, Execute, Executor, Postgres, SqlStr,
    error::BoxDynError,
    postgres::{PgArguments, PgQueryResult, PgRow, PgStatement, PgTypeInfo},
};

// TODO(transact-rs/sqlx#4337): sqlx-postgres 0.9.0 still names every statement when
// statement_cache_capacity is 0, leaving one open per query on the server and colliding
// as "sqlx_s_1 already exists" behind transaction poolers. #4337 makes capacity 0 send
// unnamed statements on its own; delete this wrapper once a sqlx release carries it.
#[derive(Debug)]
pub struct Statements<E> {
    executor: E,
    named: bool,
}

impl<E> Statements<E> {
    pub fn new(executor: E, named: bool) -> Self {
        Self { executor, named }
    }
}

struct Naming<Q> {
    query: Q,
    named: bool,
}

impl<'q, Q: Execute<'q, Postgres>> Execute<'q, Postgres> for Naming<Q> {
    fn sql(self) -> SqlStr {
        self.query.sql()
    }

    fn statement(&self) -> Option<&PgStatement> {
        self.query.statement()
    }

    fn take_arguments(&mut self) -> Result<Option<PgArguments>, BoxDynError> {
        self.query.take_arguments()
    }

    fn persistent(&self) -> bool {
        self.named && self.query.persistent()
    }
}

impl<'c, E: Executor<'c, Database = Postgres>> Executor<'c> for Statements<E> {
    type Database = Postgres;

    fn fetch_many<'e, 'q: 'e, Q>(
        self,
        query: Q,
    ) -> BoxStream<'e, Result<Either<PgQueryResult, PgRow>, sqlx::Error>>
    where
        'c: 'e,
        Q: 'q + Execute<'q, Postgres>,
    {
        self.executor.fetch_many(Naming {
            query,
            named: self.named,
        })
    }

    fn fetch_optional<'e, 'q: 'e, Q>(
        self,
        query: Q,
    ) -> BoxFuture<'e, Result<Option<PgRow>, sqlx::Error>>
    where
        'c: 'e,
        Q: 'q + Execute<'q, Postgres>,
    {
        self.executor.fetch_optional(Naming {
            query,
            named: self.named,
        })
    }

    fn prepare_with<'e>(
        self,
        sql: SqlStr,
        parameters: &'e [PgTypeInfo],
    ) -> BoxFuture<'e, Result<PgStatement, sqlx::Error>>
    where
        'c: 'e,
    {
        self.executor.prepare_with(sql, parameters)
    }

    fn describe<'e>(self, sql: SqlStr) -> BoxFuture<'e, Result<Describe<Postgres>, sqlx::Error>>
    where
        'c: 'e,
    {
        self.executor.describe(sql)
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use rstest::rstest;
    use sqlx::{AssertSqlSafe, PgConnection};

    use super::Statements;
    use crate::{Endpoint, test_support::database_endpoint};

    async fn named_statements(
        connection: &mut PgConnection,
        named: bool,
    ) -> Result<i64, sqlx::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "pg_prepared_statements is not part of the compile-checked schema"
        )]
        sqlx::query_scalar("SELECT count(*) FROM pg_prepared_statements WHERE NOT from_sql")
            .fetch_one(Statements::new(connection, named))
            .await
    }

    #[rstest]
    #[case::cache_off(0, 0)]
    #[case::cache_on(100, 4)]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn the_endpoint_cache_decides_whether_statements_stay_named(
        #[case] statement_cache_capacity: usize,
        #[case] expected: i64,
    ) -> Result<(), Box<dyn StdError>> {
        let endpoint = Endpoint {
            statement_cache_capacity,
            ..database_endpoint()?
        };
        let pool = endpoint.pool()?;
        let mut connection = pool.acquire().await?;

        for query in ["SELECT 1", "SELECT 2", "SELECT 3"] {
            #[expect(clippy::disallowed_methods, reason = "probe queries need no schema")]
            sqlx::query(AssertSqlSafe(query))
                .execute(Statements::new(
                    &mut *connection,
                    endpoint.names_statements(),
                ))
                .await?;
        }

        assert_eq!(
            named_statements(&mut connection, endpoint.names_statements()).await?,
            expected
        );
        Ok(())
    }
}

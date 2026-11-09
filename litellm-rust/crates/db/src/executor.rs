use futures_util::{future::BoxFuture, stream::BoxStream};
use sqlx::{
    Describe, Either, Execute, Executor, Postgres, SqlStr,
    postgres::{PgArguments, PgQueryResult, PgRow, PgStatement, PgTypeInfo},
};

#[derive(Debug)]
pub(crate) struct StatementPersistence<E> {
    pub executor: E,
    pub persistent: bool,
}

struct QueryPersistence<Q> {
    query: Q,
    persistent: bool,
}

impl<'q, Q: Execute<'q, Postgres>> Execute<'q, Postgres> for QueryPersistence<Q> {
    fn sql(self) -> SqlStr {
        self.query.sql()
    }

    fn statement(&self) -> Option<&PgStatement> {
        self.query.statement()
    }

    fn take_arguments(&mut self) -> Result<Option<PgArguments>, sqlx::error::BoxDynError> {
        self.query.take_arguments()
    }

    fn persistent(&self) -> bool {
        self.persistent
    }
}

impl<'c, E: Executor<'c, Database = Postgres>> Executor<'c> for StatementPersistence<E> {
    type Database = Postgres;

    fn fetch_many<'e, 'q: 'e, Q>(
        self,
        query: Q,
    ) -> BoxStream<'e, Result<Either<PgQueryResult, PgRow>, sqlx::Error>>
    where
        'c: 'e,
        Q: 'q + Execute<'q, Postgres>,
    {
        self.executor.fetch_many(QueryPersistence {
            query,
            persistent: self.persistent,
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
        self.executor.fetch_optional(QueryPersistence {
            query,
            persistent: self.persistent,
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

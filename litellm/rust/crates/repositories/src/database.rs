//! The pools, metrics sink and per-operation deadline every repository runs its queries through.

use std::{sync::Arc, time::Duration};

use litellm_db::{Call, ErrorKind, Observer, Pools, Statements};
use sqlx::{PgConnection, Postgres, pool::PoolConnection};

/// Which side of the database a read goes to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    /// The read replica when one is configured, else the writer.
    Read,
    /// The writer alone, for reads that must see the latest commit. Fails while the writer is down rather
    /// than reading a replica that may lag.
    Writer,
}

/// What every repository needs to run a query: the pools, where to report each operation, the per-operation
/// deadline, and whether statements are prepared under a name.
#[derive(Clone)]
pub struct Database {
    pools: Pools,
    events: Arc<dyn Observer>,
    socket_timeout: Option<Duration>,
    names_statements: bool,
}

impl Database {
    pub fn new(
        pools: Pools,
        events: Arc<dyn Observer>,
        socket_timeout: Option<Duration>,
        names_statements: bool,
    ) -> Self {
        Self {
            pools,
            events,
            socket_timeout,
            names_statements,
        }
    }

    /// Runs `operation` as one reported operation under the socket deadline. An error that calls for fresh
    /// connections, such as a writer that became read-only after a failover, recycles the pools; after a stale
    /// prepared plan the operation runs once more, on the fresh connections.
    pub(crate) async fn run<T, F, Fut>(
        &self,
        call: Call,
        operation: F,
    ) -> Result<T, litellm_db::Error>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, litellm_db::Error>>,
    {
        litellm_db::observe(
            self.events.as_ref(),
            call,
            litellm_db::within(self.socket_timeout, async {
                match operation().await {
                    Err(error) if self.pools.recycle_after(&error) => {
                        if error.kind() == ErrorKind::StalePlan {
                            operation().await
                        } else {
                            Err(error)
                        }
                    }
                    result => result,
                }
            }),
        )
        .await
    }

    /// Runs `steps` in one transaction on the writer. It commits when they return `Ok` and rolls back when they
    /// fail or the operation is dropped, so the second attempt `run` makes on a stale plan starts from nothing.
    pub(crate) async fn transaction<T>(
        &self,
        steps: impl for<'t, 'c> AsyncFnOnce(&'t mut Transaction<'c>) -> Result<T, litellm_db::Error>,
    ) -> Result<T, litellm_db::Error> {
        let mut session = self.session(Side::Writer).await?;
        in_transaction(&mut session.connection, session.named, steps).await
    }

    /// A connection from `side` for the queries of one operation.
    pub(crate) async fn session(&self, side: Side) -> Result<Session, litellm_db::Error> {
        let connection = match side {
            Side::Read => self.pools.read().await?,
            Side::Writer => self.pools.writer_only().await?,
        };
        Ok(self.session_on(connection))
    }

    /// A connection from the writer, or from the read replica while the writer is marked down, for a read that must
    /// see the latest commit yet keep working through a writer outage, as Python's `WriterPinnedClient` does.
    pub(crate) async fn writer_preferred_session(&self) -> Result<Session, litellm_db::Error> {
        Ok(self.session_on(self.pools.read_after_write().await?))
    }

    fn session_on(&self, connection: PoolConnection<Postgres>) -> Session {
        Session {
            connection,
            named: self.names_statements,
        }
    }
}

/// One checked-out connection for the queries of a single repository operation.
pub(crate) struct Session {
    connection: PoolConnection<Postgres>,
    named: bool,
}

impl Session {
    /// An executor for the next query on this connection.
    pub(crate) fn executor(&mut self) -> Statements<&mut PgConnection> {
        Statements::new(&mut *self.connection, self.named)
    }

    /// The connection itself, for a repository that opens a transaction on it. Its queries pass
    /// [`Session::names_statements`] to `Statements::new`.
    pub(crate) fn connection(&mut self) -> &mut PgConnection {
        &mut self.connection
    }

    /// Whether statements on this connection are prepared under a name.
    pub(crate) fn names_statements(&self) -> bool {
        self.named
    }
}

/// Where the steps of one operation take their connection: a session per step from the database, one after the
/// other so an operation never holds two, or a single transaction in tests.
pub(crate) trait Connections {
    async fn with<T>(
        &mut self,
        side: Side,
        step: impl for<'e> AsyncFnOnce(Statements<&'e mut PgConnection>) -> Result<T, litellm_db::Error>,
    ) -> Result<T, litellm_db::Error>;
}

impl Connections for &Database {
    async fn with<T>(
        &mut self,
        side: Side,
        step: impl for<'e> AsyncFnOnce(Statements<&'e mut PgConnection>) -> Result<T, litellm_db::Error>,
    ) -> Result<T, litellm_db::Error> {
        let mut session = self.session(side).await?;
        step(session.executor()).await
    }
}

/// One open transaction. Every step of the operation runs on it, whichever side the step asks for.
pub(crate) struct Transaction<'c> {
    inner: sqlx::Transaction<'c, Postgres>,
    named: bool,
}

impl Connections for Transaction<'_> {
    async fn with<T>(
        &mut self,
        _side: Side,
        step: impl for<'e> AsyncFnOnce(Statements<&'e mut PgConnection>) -> Result<T, litellm_db::Error>,
    ) -> Result<T, litellm_db::Error> {
        step(Statements::new(&mut *self.inner, self.named)).await
    }
}

/// Runs `steps` in a transaction opened on `connection`, committed only when they return `Ok`.
async fn in_transaction<T>(
    connection: &mut PgConnection,
    named: bool,
    steps: impl for<'t, 'c> AsyncFnOnce(&'t mut Transaction<'c>) -> Result<T, litellm_db::Error>,
) -> Result<T, litellm_db::Error> {
    let mut transaction = Transaction {
        inner: sqlx::Connection::begin(connection).await?,
        named,
    };
    let outcome = steps(&mut transaction).await?;
    transaction.inner.commit().await?;
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use std::error::Error as StdError;

    use litellm_db::Statements;
    use rstest::rstest;
    use sqlx::{PgConnection, PgPool};

    use super::{Connections, Side, in_transaction};
    use crate::test_support::database_pool;

    async fn insert_marker(
        db: Statements<&mut PgConnection>,
        budget_id: &str,
    ) -> Result<(), litellm_db::Error> {
        #[expect(
            clippy::disallowed_methods,
            reason = "Fixture inserts are outside the compile-checked query set"
        )]
        sqlx::query(
            r#"INSERT INTO "LiteLLM_BudgetTable" (budget_id, created_by, updated_by) VALUES ($1, 'db-query-test', 'db-query-test')"#,
        )
        .bind(budget_id)
        .execute(db)
        .await?;
        Ok(())
    }

    #[rstest]
    #[tokio::test]
    #[ignore = "requires DATABASE_URL pointing to a Prisma-migrated PostgreSQL database"]
    async fn a_transaction_keeps_the_writes_of_steps_that_succeed_and_drops_those_of_steps_that_fail(
        #[future(awt)] database_pool: Result<PgPool, Box<dyn StdError>>,
    ) -> Result<(), Box<dyn StdError>> {
        let pool = database_pool?;
        let mut outer = pool.begin().await?;

        in_transaction(&mut outer, false, async |transaction| {
            transaction
                .with(Side::Writer, async |db| {
                    insert_marker(db, "transaction-kept").await
                })
                .await
        })
        .await?;
        let failed = in_transaction(&mut outer, false, async |transaction| {
            transaction
                .with(Side::Writer, async |db| {
                    insert_marker(db, "transaction-dropped").await
                })
                .await?;
            Err::<(), _>(sqlx::Error::RowNotFound.into())
        })
        .await;

        #[expect(
            clippy::disallowed_methods,
            reason = "Assertion reads are outside the compile-checked query set"
        )]
        let stored: Vec<String> = sqlx::query_scalar(
            r#"SELECT budget_id FROM "LiteLLM_BudgetTable" WHERE budget_id LIKE 'transaction-%' ORDER BY budget_id"#,
        )
        .fetch_all(&mut *outer)
        .await?;
        assert!(failed.is_err());
        assert_eq!(stored, ["transaction-kept"]);
        Ok(())
    }
}

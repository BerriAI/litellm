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

    /// Runs `operation` as one reported operation under the socket deadline. A stale prepared plan recycles
    /// the pools and the operation runs once more, on fresh connections.
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
                    Err(error)
                        if error.kind() == ErrorKind::StalePlan
                            && self.pools.recycle_after(&error) =>
                    {
                        operation().await
                    }
                    result => result,
                }
            }),
        )
        .await
    }

    /// A connection from `side` for the queries of one operation.
    pub(crate) async fn session(&self, side: Side) -> Result<Session, litellm_db::Error> {
        let connection = match side {
            Side::Read => self.pools.read().await?,
            Side::Writer => self.pools.writer_only().await?,
        };
        Ok(Session {
            connection,
            named: self.names_statements,
        })
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
}

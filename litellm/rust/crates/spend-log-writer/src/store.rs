//! The Postgres `LedgerStore`, over the spend log repository.

use std::time::Duration;

use litellm_db::Error;
use litellm_repositories::{SpendLogRepository, SpendLogWrite};

use crate::LedgerStore;

/// Commits statements to Postgres through the spend log repository, which reports each
/// statement and applies the deadlines; a ledger transaction gives up after `transaction_timeout`.
pub struct PgLedgerStore {
    repository: SpendLogRepository,
    transaction_timeout: Duration,
}

impl PgLedgerStore {
    pub fn new(repository: SpendLogRepository, transaction_timeout: Duration) -> Self {
        Self {
            repository,
            transaction_timeout,
        }
    }
}

impl LedgerStore for PgLedgerStore {
    async fn insert(&self, writes: &[SpendLogWrite]) -> Result<usize, Error> {
        self.repository
            .insert_with_deliveries(writes, self.transaction_timeout)
            .await
    }
}

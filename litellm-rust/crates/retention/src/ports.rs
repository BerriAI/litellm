use std::{future::Future, time::Duration};

use time::OffsetDateTime;
use tokio::time::Instant;

use crate::{PartitionInterval, Target};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StatementBudget {
    deadline: Instant,
    per_statement: Duration,
}

impl StatementBudget {
    pub fn new(deadline: Instant, per_statement: Duration) -> Self {
        Self {
            deadline,
            per_statement,
        }
    }

    pub fn deadline(&self) -> Instant {
        self.deadline
    }

    pub fn timeout(&self) -> Option<Duration> {
        let left = self.deadline.saturating_duration_since(Instant::now());
        (!left.is_zero()).then(|| left.min(self.per_statement))
    }
}

pub trait RetentionStore: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn delete_batch(
        &self,
        target: Target,
        cutoff: OffsetDateTime,
        limit: u32,
        timeout: Duration,
    ) -> impl Future<Output = Result<u64, Self::Error>> + Send;
}

pub trait PartitionStore: Send + Sync + 'static {
    type Error: std::error::Error + Send + Sync + 'static;

    fn ensure_ahead(
        &self,
        now: OffsetDateTime,
        interval: PartitionInterval,
        periods_ahead: u32,
        budget: StatementBudget,
    ) -> impl Future<Output = Result<(), Self::Error>> + Send;

    fn drop_before(
        &self,
        cutoff: OffsetDateTime,
        budget: StatementBudget,
    ) -> impl Future<Output = Result<Vec<String>, Self::Error>> + Send;
}

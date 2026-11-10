mod keys;

use std::{sync::Arc, time::Duration};

pub use self::keys::{Counter, Kind};
use crate::{
    Error, Number,
    dual_cache::{BatchReads, DualCache, PipelineIncrement, TierPolicy},
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
};

/// `RoutingArgs.ttl`, the window of a per-minute counter.
pub const COUNTER_TTL: Duration = Duration::from_secs(60);

/// One router's global TPM/RPM counters: memory per router, Redis shared by every pod.
pub struct RouterUsage {
    usage: DualCache<Number, BatchReads>,
}

impl RouterUsage {
    pub fn new(remote: Option<Remote>, policy: TierPolicy) -> Self {
        Self::with_clock(remote, policy, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, policy: TierPolicy, now: Clock) -> Self {
        Self {
            usage: DualCache::batch_reading(Arc::new(RemoteSlot::new(remote)), policy, now),
        }
    }

    /// `DualCache.attach_redis_cache`: memory stays, and `false` means a Redis was already attached.
    pub fn attach(&self, remote: Remote) -> bool {
        self.usage.redis().attach(remote)
    }

    pub fn has_redis(&self) -> bool {
        self.usage.redis().get().is_some()
    }

    /// `DualCache.async_increment_cache`. A Redis failure answers with the memory count, and an
    /// entry memory cannot add to with `None`.
    pub async fn increment(
        &self,
        counter: &Counter,
        value: Number,
        ttl: Duration,
    ) -> Option<Number> {
        self.usage.increment(&counter.key(), value, Some(ttl)).await
    }

    /// One deployment's tokens and requests, `async_increment_pipeline` on its own round trip, with
    /// the totals mirrored back into memory as `int`s.
    pub async fn increment_deployment(
        &self,
        deployment_id: &str,
        model: Option<&str>,
        minute: &str,
        tokens: Number,
        requests: i64,
    ) -> Result<(), Error> {
        let increment = |kind, amount| PipelineIncrement {
            key: Counter {
                deployment_id: deployment_id.to_owned(),
                model: model.map(str::to_owned),
                minute: minute.to_owned(),
                kind,
            }
            .key(),
            amount,
            ttl: Some(COUNTER_TTL),
        };
        let increments = [
            increment(Kind::Tpm, tokens),
            increment(Kind::Rpm, Number::Int(requests)),
        ];
        let Some(totals) = self.usage.increment_pipeline(&increments).await else {
            return Ok(());
        };
        if self.usage.redis().get().is_none() {
            return Ok(());
        }
        increments
            .iter()
            .zip(totals)
            .try_for_each(|(increment, total)| {
                self.usage.memory().set(
                    &increment.key,
                    Number::Int(total.truncate()),
                    Some(COUNTER_TTL),
                )
            })?;
        Ok(())
    }

    pub async fn get_many(&self, counters: &[Counter]) -> Result<Vec<Option<Number>>, Error> {
        let keys = counters.iter().map(Counter::key).collect::<Vec<_>>();
        self.usage.batch_get(&keys).await
    }

    pub fn flush_local_counters(&self) -> Result<(), Error> {
        Ok(self.usage.memory().flush()?)
    }

    /// The memory tier keyed as Python's `InMemoryCache` was, for the fuzz probe that reads
    /// in-memory router state.
    pub fn snapshot(&self) -> Result<Vec<(String, Number)>, Error> {
        Ok(self.usage.memory().entries()?)
    }
}

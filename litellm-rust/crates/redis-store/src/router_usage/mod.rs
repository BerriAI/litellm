mod keys;
mod types;

use std::time::Duration;

use litellm_redis_client::{Declared, Expiry, RedisBatch, commands};

pub use self::keys::{Counter, Kind};
use crate::{
    Error, Number,
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
    tier::{MemoryTier, OnBreakerOpen, TierPolicy},
};

/// `main`'s usage tier: `DualCache`'s 10 s default refresh, `InMemoryCache`'s 600 s default for
/// backfilled entries and `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`.
pub const POLICY: TierPolicy = TierPolicy {
    refresh_after: Duration::from_secs(10),
    backfill_ttl: Duration::from_secs(600),
    on_breaker_open: OnBreakerOpen::ServeMemory,
    tracked_reads: 1000,
};
/// `RoutingArgs.ttl`, the window of a per-minute counter.
pub const COUNTER_TTL: Duration = Duration::from_secs(60);

/// One router's global TPM/RPM counters: memory per router, Redis shared by every pod.
pub struct RouterUsage {
    usage: MemoryTier<Number>,
    remote: RemoteSlot,
}

impl RouterUsage {
    pub fn new(remote: Option<Remote>, policy: TierPolicy) -> Self {
        Self::with_clock(remote, policy, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, policy: TierPolicy, now: Clock) -> Self {
        Self {
            usage: MemoryTier::read_through(policy, now),
            remote: RemoteSlot::new(remote),
        }
    }

    /// `DualCache.attach_redis_cache`: memory stays, and `false` means a Redis was already attached.
    pub fn attach(&self, remote: Remote) -> bool {
        self.remote.attach(remote)
    }

    /// `DualCache.async_increment_cache`: memory, then `INCRBYFLOAT` keeping an existing TTL.
    /// A Redis failure answers with the memory count.
    pub async fn increment(&self, counter: &Counter, value: Number, ttl: Duration) -> Result<Number, Error> {
        let key = counter.key();
        let local = self.add_local(&key, value, ttl)?;
        let Some(client) = self.remote.client() else {
            return Ok(local);
        };
        let increment = commands::incr_by_float(&key, value.into(), Some(Expiry::KeepExisting(ttl)));
        match client.send(increment).await {
            Ok(total) => Ok(Number::Float(total)),
            Err(error) => {
                tracing::warn!(%error, "Redis usage increment failed, falling back to the in-memory count");
                Ok(local)
            }
        }
    }

    /// One deployment's tokens and requests, `async_increment_pipeline` on its own round trip:
    /// memory, then `INCRBYFLOAT` and `EXPIRE 60` for each counter, mirrored back into memory as
    /// `int` totals.
    pub async fn increment_deployment(
        &self,
        deployment_id: &str,
        model: Option<&str>,
        minute: &str,
        tokens: Number,
        requests: i64,
    ) -> Result<(), Error> {
        let counter = |kind| Counter {
            deployment_id: deployment_id.to_owned(),
            model: model.map(str::to_owned),
            minute: minute.to_owned(),
            kind,
        };
        let counts = [
            (counter(Kind::Tpm).key(), tokens),
            (counter(Kind::Rpm).key(), Number::Int(requests)),
        ];
        let local = counts
            .iter()
            .map(|(key, value)| self.add_local(key, *value, COUNTER_TTL))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(client) = self.remote.client() else {
            return Ok(());
        };
        let pipeline = RedisBatch::pipelined(client.clone());
        let totals = counts.iter().map(|(key, value)| {
            pipeline.add(commands::incr_by_float(key, (*value).into(), Some(Expiry::Refresh(COUNTER_TTL))))
        });
        let totals = Declared::all(totals.collect::<Vec<_>>())
            .await
            .into_iter()
            .collect::<Result<Vec<_>, _>>();
        let totals = match totals {
            Ok(totals) => totals.into_iter().map(Number::Float).collect(),
            Err(error) => {
                tracing::warn!(%error, "Redis usage pipeline failed, keeping the in-memory counts");
                local
            }
        };
        counts.iter().zip(totals).try_for_each(|((key, _), total)| {
            self.usage
                .memory()
                .set(key, Number::Int(total.truncate()), Some(COUNTER_TTL))
        })?;
        Ok(())
    }

    pub async fn get_many(&self, counters: &[Counter]) -> Result<Vec<Option<Number>>, Error> {
        let keys = counters.iter().map(Counter::key).collect::<Vec<_>>();
        Ok(self.usage.get_many(&self.remote, &keys, types::decode).await?)
    }

    pub fn flush_local_counters(&self) -> Result<(), Error> {
        Ok(self.usage.memory().flush()?)
    }

    /// The memory tier keyed as Python's `InMemoryCache` was, for the fuzz probe that reads
    /// in-memory router state.
    pub fn snapshot(&self) -> Result<Vec<(String, Number)>, Error> {
        Ok(self.usage.memory().entries()?)
    }

    fn add_local(&self, key: &str, value: Number, ttl: Duration) -> Result<Number, Error> {
        Ok(self.usage.memory().update(key, Some(ttl), |current| {
            Ok(current.unwrap_or(Number::Int(0)) + value)
        })?)
    }
}

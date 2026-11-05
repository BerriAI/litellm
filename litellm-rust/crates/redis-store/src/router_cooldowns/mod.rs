mod keys;
mod types;

use std::{sync::Arc, time::Duration};

use litellm_redis_client::commands;

pub use self::types::Cooldown;
use crate::{
    Error, Number, Pending, Send,
    memory::{Clock, Memory, system_clock},
    remote::{Remote, RemoteSlot},
    tier::{MemoryTier, OnBreakerOpen, TierPolicy},
};

/// `main`'s cooldown tier: `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS` (1 s), `InMemoryCache`'s
/// 600 s default for backfilled entries and `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`.
pub const POLICY: TierPolicy = TierPolicy {
    refresh_after: Duration::from_secs(1),
    backfill_ttl: Duration::from_secs(600),
    on_breaker_open: OnBreakerOpen::ServeMemory,
    tracked_reads: 1000,
};
const MAX_CORRECTED_TTL: Duration = Duration::from_secs(60);
const CORRECTION_SLACK: Duration = Duration::from_secs(5);

/// One router's cooldowns and allowed-fails counters: memory per router, Redis shared by every pod.
pub struct RouterCooldowns {
    cooldowns: MemoryTier<Cooldown>,
    fails: Memory<i64>,
    remote: RemoteSlot,
}

impl RouterCooldowns {
    pub fn new(remote: Option<Remote>, policy: TierPolicy) -> Self {
        Self::with_clock(remote, policy, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, policy: TierPolicy, now: Clock) -> Self {
        Self {
            fails: Memory::new(None, &now),
            cooldowns: MemoryTier::read_through(policy, now),
            remote: RemoteSlot::new(remote),
        }
    }

    /// `DualCache.attach_redis_cache`: memory stays, and `false` means a Redis was already attached.
    pub fn attach(&self, remote: Remote) -> bool {
        self.remote.attach(remote)
    }

    /// Memory first, then a `SET ... EX int(ttl)` whose failure only logs, like `DualCache.set_cache`.
    pub async fn put(&self, model_id: &str, cooldown: &Cooldown, ttl: Duration) -> Result<(), Error> {
        let key = keys::cooldown(model_id);
        self.cooldowns.memory().set(&key, cooldown.clone(), Some(ttl))?;
        if let Some(client) = self.remote.client()
            && let Err(error) = client.send(commands::set(&key, cooldown.encode(), ttl)).await
        {
            tracing::warn!(%error, "writing a cooldown to Redis failed");
        }
        Ok(())
    }

    /// The fleet-wide fail count. A Redis failure falls back to this router's own count.
    pub async fn increment_allowed_fails(
        &self,
        model_id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> Result<i64, Error> {
        let key = keys::allowed_fails(model_id, suffix);
        let local = self
            .fails
            .update(&key, Some(ttl), |count| Ok(count.unwrap_or(0) + 1))?;
        let Some(client) = self.remote.client() else {
            return Ok(local);
        };
        match client.send(commands::incr_by(&key, 1, ttl)).await {
            Ok(count) => Ok(count),
            Err(error) => {
                tracing::warn!(%error, "allowed_fails counter fell back to this worker's in-memory count");
                Ok(local)
            }
        }
    }

    /// The deployments still cooling down, read on the batch `send` picks. An entry whose cooldown
    /// already ended is dropped from memory, and one memory would keep past its end is shortened, at
    /// most to 60 s.
    pub fn active(
        self: &Arc<Self>,
        send: Send<'_>,
        model_ids: &[String],
    ) -> Result<Pending<Vec<(String, Cooldown)>>, Error> {
        let read = self
            .cooldowns
            .declare_many(&self.remote, send, &keys::cooldowns(model_ids))?;
        let model_ids = model_ids.to_vec();
        let store = Arc::clone(self);
        Ok(Box::pin(async move {
            let read = store.cooldowns.resolve_many(read, Cooldown::decode).await?;
            let now = store.cooldowns.now();
            let active = model_ids
                .into_iter()
                .zip(read)
                .filter_map(|(model_id, cooldown)| cooldown.map(|cooldown| (model_id, cooldown)))
                .map(|(model_id, cooldown)| {
                    Ok(store
                        .correct(&model_id, &cooldown, now)?
                        .then_some((model_id, cooldown)))
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(active.into_iter().flatten().collect())
        }))
    }

    /// The shortest stored cooldown, ended ones included, as `main`'s `get_min_cooldown` reads it.
    pub async fn minimum(&self, model_ids: &[String]) -> Result<Option<Number>, Error> {
        let read = self
            .cooldowns
            .get_many(&self.remote, &keys::cooldowns(model_ids), Cooldown::decode)
            .await?;
        Ok(Number::min(
            read.into_iter().flatten().map(|cooldown| cooldown.cooldown_time),
        ))
    }

    pub fn delete_local(&self, model_id: &str) -> Result<(), Error> {
        Ok(self.cooldowns.memory().delete(&keys::cooldown(model_id))?)
    }

    /// Drops this router's allowed-fails counts, leaving its cooldowns in place.
    pub fn flush_local_counters(&self) -> Result<(), Error> {
        Ok(self.fails.flush()?)
    }

    /// The memory tier keyed as Python's `InMemoryCache` was, for the fuzz probe that reads
    /// in-memory router state.
    pub fn snapshot(&self) -> Result<Vec<(String, Cooldown)>, Error> {
        Ok(self.cooldowns.memory().entries()?)
    }

    fn correct(&self, model_id: &str, cooldown: &Cooldown, now: Duration) -> Result<bool, Error> {
        let key = keys::cooldown(model_id);
        let memory = self.cooldowns.memory();
        let remaining = cooldown.remaining(now);
        if remaining <= 0.0 {
            memory.delete(&key)?;
            return Ok(false);
        }
        let remaining = Duration::from_secs_f64(remaining);
        if memory.expires_at(&key)? > Some(now + remaining + CORRECTION_SLACK) {
            memory.delete(&key)?;
            memory.set(&key, cooldown.clone(), Some(remaining.min(MAX_CORRECTED_TTL)))?;
        }
        Ok(true)
    }
}

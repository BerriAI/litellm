mod keys;
mod types;

use std::{sync::Arc, time::Duration};

use litellm_redis_client::{Amount, Command, Expiry, commands};

pub use self::types::Cooldown;
use crate::{
    Error, Number, Pending, Send,
    dual_cache::{BatchReads, DualCache, TierPolicy},
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
};

const MAX_CORRECTED_TTL: Duration = Duration::from_secs(60);
const CORRECTION_SLACK: Duration = Duration::from_secs(5);

/// One router's cooldowns and allowed-fails counters: memory per router, Redis shared by every pod.
pub struct RouterCooldowns {
    cooldowns: DualCache<Cooldown, BatchReads>,
    fails: DualCache<i64>,
    redis: Arc<RemoteSlot>,
}

impl RouterCooldowns {
    pub fn new(remote: Option<Remote>, policy: TierPolicy) -> Self {
        Self::with_clock(remote, policy, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, policy: TierPolicy, now: Clock) -> Self {
        let redis = Arc::new(RemoteSlot::new(remote));
        Self {
            fails: DualCache::new(redis.clone(), policy.memory(), now.clone()),
            cooldowns: DualCache::batch_reading(redis.clone(), policy, now),
            redis,
        }
    }

    /// `DualCache.attach_redis_cache`: memory stays, and `false` means a Redis was already attached.
    pub fn attach(&self, remote: Remote) -> bool {
        self.redis.attach(remote)
    }

    pub fn has_redis(&self) -> bool {
        self.redis.get().is_some()
    }

    /// `DualCache.async_set_cache`: memory first, then a `SET` of `json.dumps(value)` with
    /// `EX int(ttl)` whose failure only logs.
    pub async fn put(
        &self,
        model_id: &str,
        cooldown: &Cooldown,
        ttl: Duration,
    ) -> Result<(), Error> {
        self.cooldowns
            .set(&keys::cooldown(model_id), cooldown.clone(), Some(ttl))
            .await
    }

    /// `DualCache.set_cache`: memory first, then a `SET` of `str(value)` with `EX int(ttl)` whose
    /// failure only logs.
    pub async fn put_str(
        &self,
        model_id: &str,
        cooldown: &Cooldown,
        ttl: Duration,
    ) -> Result<(), Error> {
        self.cooldowns
            .set_str(&keys::cooldown(model_id), cooldown.clone(), Some(ttl))
            .await
    }

    /// `DualCache.async_increment_cache`: the fleet-wide fail count through `INCRBYFLOAT`. A Redis
    /// failure falls back to this router's own count.
    pub async fn increment_allowed_fails(
        &self,
        model_id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> Result<i64, Error> {
        let key = keys::allowed_fails(model_id, suffix);
        let command =
            commands::incr_by_float(&key, Amount::Int(1), Some(Expiry::KeepExisting(ttl)));
        self.count_fail(&key, ttl, command, Number::Float).await
    }

    /// `DualCache.increment_cache`: the fleet-wide fail count through `INCRBY`. A Redis failure
    /// falls back to this router's own count.
    pub async fn increment_allowed_fails_int(
        &self,
        model_id: &str,
        suffix: Option<&str>,
        ttl: Duration,
    ) -> Result<i64, Error> {
        let key = keys::allowed_fails(model_id, suffix);
        let command = commands::incr_by(&key, 1, ttl);
        self.count_fail(&key, ttl, command, Number::Int).await
    }

    async fn count_fail<T>(
        &self,
        key: &str,
        ttl: Duration,
        command: Command<T>,
        total: fn(T) -> Number,
    ) -> Result<i64, Error> {
        let local = self
            .fails
            .memory()
            .update(key, Some(ttl), |count| Ok(count.unwrap_or(0) + 1))?;
        let Some(client) = self.redis.client() else {
            return Ok(local);
        };
        match client.send(command).await {
            Ok(count) => Ok(total(count).truncate()),
            Err(error) => {
                crate::failure_log::redis_failure!(
                    &error,
                    "allowed_fails counter fell back to this worker's in-memory count"
                );
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
            .declare_batch_get(send, &keys::cooldowns(model_ids))?;
        let model_ids = model_ids.to_vec();
        let store = Arc::clone(self);
        Ok(Box::pin(async move {
            let read = store.cooldowns.resolve_batch_get(read).await?;
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
            .batch_get(&keys::cooldowns(model_ids))
            .await?;
        Ok(Number::min(
            read.into_iter()
                .flatten()
                .map(|cooldown| cooldown.cooldown_time),
        ))
    }

    pub fn delete_local(&self, model_id: &str) -> Result<(), Error> {
        Ok(self.cooldowns.memory().delete(&keys::cooldown(model_id))?)
    }

    /// Drops this router's allowed-fails counts, leaving its cooldowns in place.
    pub fn flush_local_counters(&self) -> Result<(), Error> {
        Ok(self.fails.memory().flush()?)
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
        let remaining = Duration::try_from_secs_f64(remaining).unwrap_or(Duration::MAX);
        let ends = now.saturating_add(remaining);
        if memory.expires_at(&key)? > Some(ends.saturating_add(CORRECTION_SLACK)) {
            memory.refresh(&key, cooldown.clone(), remaining.min(MAX_CORRECTED_TTL))?;
        }
        Ok(true)
    }
}

use std::time::Duration;

use litellm_cache::{
    Clock, DualCache, Error as MemoryError, OnBreakerOpen, TierPolicy, system_clock,
};
use litellm_cache_memory::InMemoryCache;
use litellm_python_compat::{Value, repr::repr};
use litellm_redis_client::{Expiry, RedisClient};

use crate::{DecodeError, Number, codec::python_value};

/// `main`'s cooldown tier: `DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS` (1 s), `InMemoryCache`'s
/// 600 s default for backfilled entries and `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`.
pub const POLICY: TierPolicy = TierPolicy {
    refresh_after: Duration::from_secs(1),
    backfill_ttl: Duration::from_secs(600),
    on_breaker_open: OnBreakerOpen::ServeMemory,
    tracked_reads: 1000,
};
const MEMORY_ENTRIES: usize = 200;
const MAX_CORRECTED_TTL: Duration = Duration::from_secs(60);
const CORRECTION_SLACK: Duration = Duration::from_secs(5);

#[derive(Clone, Debug, PartialEq)]
pub struct Cooldown {
    pub exception_received: String,
    pub status_code: String,
    pub timestamp: Number,
    pub cooldown_time: Number,
}

impl Cooldown {
    fn remaining(&self, now: Duration) -> f64 {
        self.timestamp.as_f64() + self.cooldown_time.as_f64() - now.as_secs_f64()
    }

    /// `str(dict)`, the bytes `RedisCache.set_cache` writes on `main`.
    pub fn encode(&self) -> String {
        let field = |name: &str, value: Value| (Value::Str(name.to_owned()), value);
        repr(&Value::Dict(vec![
            field(
                "exception_received",
                Value::Str(self.exception_received.clone()),
            ),
            field("status_code", Value::Str(self.status_code.clone())),
            field("timestamp", self.timestamp.to_value()),
            field("cooldown_time", self.cooldown_time.to_value()),
        ]))
    }

    /// Reads both `main`'s `str(dict)` and a JSON object, as `RedisCache._get_cache_logic` does.
    pub fn decode(raw: &[u8]) -> Result<Self, DecodeError> {
        let Value::Dict(fields) = python_value(raw)? else {
            return Err(DecodeError::Shape("a cooldown is a dict"));
        };
        let field = |name: &str| {
            fields
                .iter()
                .find(|(key, _)| matches!(key, Value::Str(key) if key == name))
                .map(|(_, value)| value)
        };
        let text = |name: &'static str| match field(name) {
            Some(Value::Str(text)) => Ok(text.clone()),
            _ => Err(DecodeError::Shape(name)),
        };
        let number = |name: &'static str| {
            field(name)
                .and_then(Number::from_value)
                .ok_or(DecodeError::Shape(name))
        };
        Ok(Self {
            exception_received: text("exception_received")?,
            status_code: text("status_code")?,
            timestamp: number("timestamp")?,
            cooldown_time: number("cooldown_time")?,
        })
    }
}

fn key(model_id: &str) -> String {
    format!("deployment:{model_id}:cooldown")
}

fn allowed_fails_key(model_id: &str, suffix: Option<&str>) -> String {
    match suffix.filter(|suffix| !suffix.is_empty()) {
        Some(suffix) => format!("deployment:{model_id}:allowed_fails:{suffix}"),
        None => format!("deployment:{model_id}:allowed_fails"),
    }
}

fn memory<V: Clone>(policy: TierPolicy, now: &Clock) -> InMemoryCache<V> {
    let now = now.clone();
    InMemoryCache::with_clock(Some(MEMORY_ENTRIES), Some(policy.backfill_ttl), move || {
        now()
    })
}

/// `BaseCache.get_ttl` sends `int(ttl)`, so Redis sees whole seconds.
fn whole_seconds(ttl: Duration) -> Duration {
    Duration::from_secs(ttl.as_secs())
}

/// One router's cooldowns and allowed-fails counters: memory per router, Redis shared by every pod.
pub struct RouterCooldowns {
    cooldowns: DualCache<InMemoryCache<Cooldown>, RedisClient>,
    fails: DualCache<InMemoryCache<i64>, RedisClient>,
}

impl RouterCooldowns {
    pub fn new(client: Option<RedisClient>, policy: TierPolicy) -> Self {
        Self::with_clock(client, policy, system_clock())
    }

    pub fn with_clock(client: Option<RedisClient>, policy: TierPolicy, now: Clock) -> Self {
        Self {
            cooldowns: DualCache::read_through(
                memory(policy, &now),
                client.clone(),
                policy,
                now.clone(),
            ),
            fails: DualCache::read_through(memory(policy, &now), client, policy, now),
        }
    }

    /// Memory first, then a `SET ... EX int(ttl)` whose failure only logs, like `DualCache.set_cache`.
    pub async fn put(
        &self,
        model_id: &str,
        cooldown: &Cooldown,
        ttl: Duration,
    ) -> Result<(), MemoryError> {
        let key = key(model_id);
        self.cooldowns.set_memory(&key, cooldown.clone(), ttl)?;
        if let Some(remote) = self.cooldowns.remote()
            && let Err(error) = remote
                .set(&key, cooldown.encode(), whole_seconds(ttl))
                .await
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
    ) -> Result<i64, MemoryError> {
        let key = allowed_fails_key(model_id, suffix);
        let local = self
            .fails
            .update_memory(&key, ttl, |count| count.unwrap_or(0) + 1)?;
        let Some(remote) = self.fails.remote() else {
            return Ok(local);
        };
        match remote
            .incr_by(&key, 1, Expiry::KeepExisting(whole_seconds(ttl)))
            .await
        {
            Ok(count) => Ok(count),
            Err(error) => {
                tracing::warn!(%error, "allowed_fails counter fell back to this worker's in-memory count");
                Ok(local)
            }
        }
    }

    /// The deployments still cooling down. An entry whose cooldown already ended is dropped
    /// from memory, and one memory would keep past its end is shortened, at most to 60 s.
    pub async fn active(
        &self,
        model_ids: &[String],
    ) -> Result<Vec<(String, Cooldown)>, MemoryError> {
        let read = self.read(model_ids).await?;
        let now = self.cooldowns.now();
        let corrected = model_ids
            .iter()
            .zip(read)
            .filter_map(|(model_id, cooldown)| cooldown.map(|cooldown| (model_id, cooldown)))
            .map(|(model_id, cooldown)| {
                Ok(self
                    .correct(model_id, &cooldown, now)?
                    .then(|| (model_id.clone(), cooldown)))
            })
            .collect::<Result<Vec<_>, MemoryError>>()?;
        Ok(corrected.into_iter().flatten().collect())
    }

    /// The shortest stored cooldown, ended ones included, as `main`'s `get_min_cooldown` reads it.
    pub async fn minimum(&self, model_ids: &[String]) -> Result<Option<Number>, MemoryError> {
        let read = self.read(model_ids).await?;
        Ok(Number::min(
            read.into_iter()
                .flatten()
                .map(|cooldown| cooldown.cooldown_time),
        ))
    }

    pub fn evict_local(&self, model_id: &str) -> Result<(), MemoryError> {
        self.cooldowns.delete_memory(&key(model_id))
    }

    pub fn flush_local_counters(&self) -> Result<(), MemoryError> {
        self.fails.flush_memory()
    }

    /// The memory tier keyed as Python's `InMemoryCache` was, for the fuzz probe that reads
    /// in-memory router state.
    pub fn snapshot(&self) -> Result<Vec<(String, Cooldown)>, MemoryError> {
        self.cooldowns.memory().entries()
    }

    async fn read(&self, model_ids: &[String]) -> Result<Vec<Option<Cooldown>>, MemoryError> {
        let keys = model_ids
            .iter()
            .map(|model_id| key(model_id))
            .collect::<Vec<_>>();
        self.cooldowns.get_many(&keys, Cooldown::decode).await
    }

    fn correct(
        &self,
        model_id: &str,
        cooldown: &Cooldown,
        now: Duration,
    ) -> Result<bool, MemoryError> {
        let key = key(model_id);
        let remaining = cooldown.remaining(now);
        if remaining <= 0.0 {
            self.cooldowns.delete_memory(&key)?;
            return Ok(false);
        }
        let remaining = Duration::from_secs_f64(remaining);
        let memory = self.cooldowns.memory();
        if memory.expires_at(&key)? > Some(now + remaining + CORRECTION_SLACK) {
            self.cooldowns.delete_memory(&key)?;
            self.cooldowns
                .set_memory(&key, cooldown.clone(), remaining.min(MAX_CORRECTED_TTL))?;
        }
        Ok(true)
    }
}

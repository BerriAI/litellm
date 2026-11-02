use std::time::Duration;

use litellm_cache::{
    Clock, DualCache, Error as MemoryError, OnBreakerOpen, TierPolicy, system_clock,
};
use litellm_cache_memory::InMemoryCache;
use litellm_redis_client::{Expiry, RedisClient, Value};

use crate::{DecodeError, Number, codec::python_value};

/// `main`'s usage tier: `DualCache`'s 10 s default refresh, `InMemoryCache`'s 600 s default for
/// backfilled entries and `DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE`.
pub const POLICY: TierPolicy = TierPolicy {
    refresh_after: Duration::from_secs(10),
    backfill_ttl: Duration::from_secs(600),
    on_breaker_open: OnBreakerOpen::ServeMemory,
    tracked_reads: 1000,
};
const MEMORY_ENTRIES: usize = 200;
/// `RoutingArgs.ttl`, the window of a per-minute counter.
pub const COUNTER_TTL: Duration = Duration::from_secs(60);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Tpm,
    Rpm,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counter {
    pub deployment_id: String,
    pub model: Option<String>,
    pub minute: String,
    pub kind: Kind,
}

impl Counter {
    /// `global_router:{id}:{model}:{tpm|rpm}:{HH-MM}`, where Python formats a missing model as `None`.
    fn key(&self) -> String {
        let kind = match self.kind {
            Kind::Tpm => "tpm",
            Kind::Rpm => "rpm",
        };
        let model = self.model.as_deref().unwrap_or("None");
        format!(
            "global_router:{}:{model}:{kind}:{}",
            self.deployment_id, self.minute
        )
    }
}

fn decode(raw: &[u8]) -> Result<Number, DecodeError> {
    Number::from_value(&python_value(raw)?).ok_or(DecodeError::Shape("a usage counter is a number"))
}

fn whole_seconds(ttl: Duration) -> Duration {
    Duration::from_secs(ttl.as_secs())
}

/// One router's global TPM/RPM counters: memory per router, Redis shared by every pod.
pub struct RouterUsage(DualCache<InMemoryCache<Number>, RedisClient>);

impl RouterUsage {
    pub fn new(client: Option<RedisClient>, policy: TierPolicy) -> Self {
        Self::with_clock(client, policy, system_clock())
    }

    pub fn with_clock(client: Option<RedisClient>, policy: TierPolicy, now: Clock) -> Self {
        let clock = now.clone();
        let memory =
            InMemoryCache::with_clock(Some(MEMORY_ENTRIES), Some(policy.backfill_ttl), move || {
                clock()
            });
        Self(DualCache::read_through(memory, client, policy, now))
    }

    /// `DualCache.async_increment_cache`: memory, then `INCRBYFLOAT` keeping an existing TTL.
    /// A Redis failure answers with the memory count.
    pub async fn increment(
        &self,
        counter: &Counter,
        value: Number,
        ttl: Duration,
    ) -> Result<Number, MemoryError> {
        let key = counter.key();
        let local = self.add_local(&key, value, ttl)?;
        let Some(remote) = self.0.remote() else {
            return Ok(local);
        };
        match remote
            .incr_by_float(
                &key,
                value.as_f64(),
                Expiry::KeepExisting(whole_seconds(ttl)),
            )
            .await
        {
            Ok(total) => Ok(Number::Float(total)),
            Err(error) => {
                tracing::warn!(%error, "Redis usage increment failed, falling back to the in-memory count");
                Ok(local)
            }
        }
    }

    /// One deployment's tokens and requests: memory, then one pipeline that resets each
    /// counter's 60 s window, mirrored back into memory as `int` totals.
    pub async fn increment_deployment(
        &self,
        deployment_id: &str,
        model: Option<&str>,
        minute: &str,
        tokens: Number,
        requests: i64,
    ) -> Result<(), MemoryError> {
        let counter = |kind| Counter {
            deployment_id: deployment_id.to_owned(),
            model: model.map(str::to_owned),
            minute: minute.to_owned(),
            kind,
        };
        let increments = [
            (counter(Kind::Tpm).key(), tokens),
            (counter(Kind::Rpm).key(), Number::Int(requests)),
        ];
        let local = increments
            .iter()
            .map(|(key, value)| self.add_local(key, *value, COUNTER_TTL))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(remote) = self.0.remote() else {
            return Ok(());
        };
        let pipeline = increments
            .iter()
            .fold(remote.pipeline(), |pipeline, (key, value)| {
                pipeline.incr_by_float(key, value.as_f64(), Expiry::Refresh(COUNTER_TTL))
            });
        let totals = match remote
            .run(&pipeline)
            .await
            .into_iter()
            .map(total)
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(totals) => totals,
            Err(error) => {
                tracing::warn!(%error, "Redis usage pipeline failed, keeping the in-memory counts");
                local
            }
        };
        for ((key, _), total) in increments.iter().zip(totals) {
            self.0
                .set_memory(key, Number::Int(total.truncate()), COUNTER_TTL)?;
        }
        Ok(())
    }

    pub async fn get_many(&self, counters: &[Counter]) -> Result<Vec<Option<Number>>, MemoryError> {
        let keys = counters.iter().map(Counter::key).collect::<Vec<_>>();
        self.0.get_many(&keys, decode).await
    }

    pub fn flush_local_counters(&self) -> Result<(), MemoryError> {
        self.0.flush_memory()
    }

    /// The memory tier keyed as Python's `InMemoryCache` was, for the fuzz probe that reads
    /// in-memory router state.
    pub fn snapshot(&self) -> Result<Vec<(String, Number)>, MemoryError> {
        self.0.memory().entries()
    }

    fn add_local(&self, key: &str, value: Number, ttl: Duration) -> Result<Number, MemoryError> {
        self.0.update_memory(key, ttl, |current| {
            current.unwrap_or(Number::Int(0)) + value
        })
    }
}

#[derive(Debug, thiserror::Error)]
enum TotalError {
    #[error(transparent)]
    Redis(#[from] litellm_redis_client::Error),
    #[error("INCRBYFLOAT answered {0:?}")]
    Unexpected(Value),
}

fn total(reply: Result<Value, litellm_redis_client::Error>) -> Result<Number, TotalError> {
    let reply = reply?;
    let text = match &reply {
        Value::BulkString(bytes) => std::str::from_utf8(bytes).ok(),
        Value::SimpleString(text) => Some(text.as_str()),
        _ => None,
    };
    text.and_then(|text| text.parse::<f64>().ok())
        .map(Number::Float)
        .ok_or(TotalError::Unexpected(reply))
}

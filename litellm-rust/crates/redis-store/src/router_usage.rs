use std::time::Duration;

use litellm_cache::Error as MemoryError;
use litellm_cache_memory::InMemoryCache;
use litellm_redis_client::{Error, Expiry, RedisClient, RedisError, RedisErrorKind, Value};

use crate::{
    DecodeError, Number,
    codec::python_value,
    tier::{Clock, MemoryTier, OnBreakerOpen, TierPolicy, system_clock},
};

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
pub struct RouterUsage(MemoryTier<InMemoryCache<Number>, RedisClient>);

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
        Self(MemoryTier::read_through(memory, client, policy, now))
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
            .incr_by_float(&key, value.into(), Expiry::KeepExisting(whole_seconds(ttl)))
            .await
        {
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
    ) -> Result<(), MemoryError> {
        let increment = Increment::new(deployment_id, model, minute, tokens, requests);
        let local = increment
            .counts
            .iter()
            .map(|(key, value)| self.add_local(key, *value, COUNTER_TTL))
            .collect::<Result<Vec<_>, _>>()?;
        let Some(remote) = self.0.remote() else {
            return Ok(());
        };
        let pipeline = increment
            .counts
            .iter()
            .fold(remote.pipeline(), |pipeline, (key, value)| {
                pipeline.incr_by_float(key, (*value).into(), Some(COUNTER_TTL))
            });
        let totals = remote
            .run(&pipeline)
            .await
            .into_iter()
            .map(|total| total.and_then(float_reply))
            .collect();
        let keys: Vec<String> = increment.counts.into_iter().map(|(key, _)| key).collect();
        self.mirror(&keys, local, totals)
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

    fn mirror(
        &self,
        keys: &[String],
        local: Vec<Number>,
        totals: Result<Vec<f64>, Error>,
    ) -> Result<(), MemoryError> {
        let totals = match totals {
            Ok(totals) => totals.into_iter().map(Number::Float).collect(),
            Err(error) => {
                tracing::warn!(%error, "Redis usage pipeline failed, keeping the in-memory counts");
                local
            }
        };
        keys.iter().zip(totals).try_for_each(|(key, total)| {
            self.0
                .set_memory(key, Number::Int(total.truncate()), COUNTER_TTL)
        })
    }

    fn add_local(&self, key: &str, value: Number, ttl: Duration) -> Result<Number, MemoryError> {
        self.0.update_memory(key, ttl, |current| {
            current.unwrap_or(Number::Int(0)) + value
        })
    }
}

struct Increment {
    counts: [(String, Number); 2],
}

impl Increment {
    fn new(
        deployment_id: &str,
        model: Option<&str>,
        minute: &str,
        tokens: Number,
        requests: i64,
    ) -> Self {
        let counter = |kind| Counter {
            deployment_id: deployment_id.to_owned(),
            model: model.map(str::to_owned),
            minute: minute.to_owned(),
            kind,
        };
        Self {
            counts: [
                (counter(Kind::Tpm).key(), tokens),
                (counter(Kind::Rpm).key(), Number::Int(requests)),
            ],
        }
    }
}

fn float_reply(value: Value) -> Result<f64, Error> {
    let total = match &value {
        Value::BulkString(bytes) => std::str::from_utf8(bytes)
            .ok()
            .and_then(|text| text.parse().ok()),
        Value::Double(total) => Some(*total),
        _ => None,
    };
    total.ok_or_else(|| {
        Error::Decode(RedisError::from((
            RedisErrorKind::UnexpectedReturnType,
            "INCRBYFLOAT did not answer a number",
        )))
    })
}

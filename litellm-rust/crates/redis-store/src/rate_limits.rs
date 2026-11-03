mod lua;
mod scripts;

use std::{
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
};

use futures_util::future::join_all;
use litellm_cache::Error as MemoryError;
use litellm_cache_memory::InMemoryCache;
use litellm_python_compat::Value;
use litellm_redis_client::{Declared, Expiry, RedisBatch, RedisClient, Slot};

pub use self::scripts::{
    AcquireReply, Applied, AtomicCounter, AtomicReply, Gauge, GuardedIncrement, GuardedReply,
    Pending, WindowCount,
};
use crate::{Number, RequestRedis, codec::python_value};

/// A rate-limit descriptor, `{descriptor:value}`, the hash tag every key of its window shares.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Scope {
    pub descriptor: String,
    pub value: String,
}

impl Scope {
    fn key(&self, suffix: &str) -> String {
        format!("{{{}:{}}}:{suffix}", self.descriptor, self.value)
    }

    fn window_key(&self) -> String {
        self.key("window")
    }

    fn gauge_key(&self) -> String {
        self.key("max_parallel_requests")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Requests,
    Tokens,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Counter {
    pub scope: Scope,
    pub kind: Kind,
}

impl Counter {
    fn key(&self) -> String {
        self.scope.key(match self.kind {
            Kind::Requests => "requests",
            Kind::Tokens => "tokens",
        })
    }
}

/// One value of the limiter's keyspace, as both tiers address it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Entry {
    Window(Scope),
    Counter(Counter),
    Gauge(Scope),
}

impl Entry {
    fn key(&self) -> String {
        match self {
            Self::Window(scope) => scope.window_key(),
            Self::Counter(counter) => counter.key(),
            Self::Gauge(scope) => scope.gauge_key(),
        }
    }
}

/// What the memory tier holds: counters and mirrored gauge counts, window starts, and the per-slot
/// registry a gauge keeps while Redis is unreachable.
#[derive(Clone, Debug, PartialEq)]
pub enum Local {
    Number(Number),
    Text(String),
    Slots(Vec<(String, f64)>),
}

/// A `RedisPipelineIncrementOperation`.
#[derive(Clone, Debug, PartialEq)]
pub struct Increment {
    pub counter: Counter,
    pub amount: Number,
    pub ttl: Option<Duration>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Redis(#[from] litellm_redis_client::Error),
    #[error("rate limiter script reply {0}")]
    Reply(&'static str),
    #[error("the rate limiter has no Redis to run its scripts on")]
    NoRedis,
}

/// The v3 rate limiter's keyspace: its memory tier, its Lua scripts and the plain counter writes
/// `DualCache` made for it on `main`. The admission logic stays with the caller.
pub struct RateLimits {
    redis: Option<RedisClient>,
    memory: InMemoryCache<Local>,
    increments: Mutex<()>,
    backfill_ttl: Option<Duration>,
    redis_ttl: Duration,
}

impl RateLimits {
    /// `backfill_ttl` is the `default_in_memory_ttl` of the `DualCache` this tier replaces, and
    /// `redis_ttl` its `RedisCache.default_ttl`, which an increment without a TTL takes.
    pub fn new(
        redis: Option<RedisClient>,
        backfill_ttl: Option<Duration>,
        redis_ttl: Duration,
    ) -> Self {
        Self::with_memory(
            redis,
            backfill_ttl,
            redis_ttl,
            InMemoryCache::new(None, None),
        )
    }

    pub fn with_memory(
        redis: Option<RedisClient>,
        backfill_ttl: Option<Duration>,
        redis_ttl: Duration,
        memory: InMemoryCache<Local>,
    ) -> Self {
        Self {
            redis,
            memory,
            increments: Mutex::new(()),
            backfill_ttl,
            redis_ttl,
        }
    }

    pub fn has_redis(&self) -> bool {
        self.redis.is_some()
    }

    /// `_group_keys_by_hash_tag`: indices of `scopes` sharing a cluster slot, in first-seen order,
    /// or one group when the client is not a cluster and grouping is not forced.
    pub fn slot_groups(&self, scopes: &[Scope], force: bool) -> Vec<Vec<usize>> {
        let by_slot = force || self.redis.as_ref().is_some_and(RedisClient::is_cluster);
        if !by_slot {
            return vec![(0..scopes.len()).collect()];
        }
        scopes
            .iter()
            .enumerate()
            .fold(
                Vec::<(Slot, Vec<usize>)>::new(),
                |mut groups, (index, scope)| {
                    let slot = Slot::for_key(scope.window_key());
                    match groups.iter_mut().find(|(seen, _)| *seen == slot) {
                        Some((_, members)) => members.push(index),
                        None => groups.push((slot, vec![index])),
                    }
                    groups
                },
            )
            .into_iter()
            .map(|(_, members)| members)
            .collect()
    }

    pub fn local(&self, entry: &Entry) -> Result<Option<Local>, MemoryError> {
        self.memory.get_cache(&entry.key())
    }

    pub fn set_local(
        &self,
        entry: &Entry,
        value: Local,
        ttl: Option<Duration>,
    ) -> Result<(), MemoryError> {
        self.memory.set_cache(entry.key(), value, ttl).map(drop)
    }

    /// `DualCache.async_get_cache`: memory, else Redis written back to memory for `backfill_ttl`,
    /// or the store's default. A Redis failure reads as a miss, as `DualCache` logs and swallows it.
    pub async fn get(
        &self,
        entry: &Entry,
        backfill_ttl: Option<Duration>,
    ) -> Result<Option<Local>, MemoryError> {
        let key = entry.key();
        if let Some(local) = self.memory.get_cache(&key)? {
            return Ok(Some(local));
        }
        let Some(redis) = &self.redis else {
            return Ok(None);
        };
        let raw = match redis.get::<Vec<u8>>(&key).await {
            Ok(raw) => raw,
            Err(error) => {
                tracing::error!(%error, "LiteLLM Cache: exception in async_get_cache");
                return Ok(None);
            }
        };
        let Some(value) = raw
            .map(|raw| decode(&raw))
            .transpose()
            .unwrap_or_else(|error| {
                tracing::error!(%error, "LiteLLM Cache: exception in async_get_cache");
                None
            })
        else {
            return Ok(None);
        };
        self.memory
            .set_cache(key, value.clone(), backfill_ttl.or(self.backfill_ttl))?;
        Ok(Some(value))
    }

    pub fn sliding_window(
        &self,
        request: Option<&RequestRedis>,
        counters: &[Counter],
        now: i64,
        window_size: i64,
    ) -> Result<Pending<Vec<WindowCount>>, Error> {
        Ok(scripts::sliding_window(
            &self.pre(request)?,
            counters,
            now,
            window_size,
        ))
    }

    pub fn read_counters(
        &self,
        request: Option<&RequestRedis>,
        counters: &[Counter],
    ) -> Result<Pending<Vec<Option<String>>>, Error> {
        Ok(scripts::read_counters(&self.pre(request)?, counters))
    }

    pub fn check_and_increment(
        &self,
        request: Option<&RequestRedis>,
        counters: &[AtomicCounter],
    ) -> Result<Pending<AtomicReply>, Error> {
        Ok(scripts::check_and_increment(&self.pre(request)?, counters))
    }

    pub fn guarded_increment(
        &self,
        request: Option<&RequestRedis>,
        increment: &GuardedIncrement,
    ) -> Result<Pending<GuardedReply>, Error> {
        Ok(scripts::guarded_increment(&self.pre(request)?, increment))
    }

    pub fn parallel_acquire(
        &self,
        request: Option<&RequestRedis>,
        gauges: &[Gauge],
        slot_ttl: Duration,
        slot_id: &str,
    ) -> Result<Pending<AcquireReply>, Error> {
        Ok(scripts::parallel_acquire(
            &self.pre(request)?,
            gauges,
            slot_ttl,
            slot_id,
        ))
    }

    pub fn parallel_release(
        &self,
        request: Option<&RequestRedis>,
        gauges: &[Scope],
        slot_id: &str,
    ) -> Result<Pending<Vec<i64>>, Error> {
        Ok(scripts::parallel_release(
            &self.pre(request)?,
            gauges,
            slot_id,
        ))
    }

    pub fn parallel_count(
        &self,
        request: Option<&RequestRedis>,
        gauges: &[Scope],
        slot_ttl: Duration,
    ) -> Result<Pending<Vec<i64>>, Error> {
        Ok(scripts::parallel_count(
            &self.pre(request)?,
            gauges,
            slot_ttl,
        ))
    }

    pub fn token_increment(
        &self,
        request: Option<&RequestRedis>,
        increments: &[Increment],
    ) -> Result<Pending<Vec<f64>>, Error> {
        Ok(scripts::token_increment(&self.pre(request)?, increments))
    }

    /// `TOKEN_INCREMENT_SCRIPT` on the request's post-call batch. A failed script falls back to
    /// `increment_many` for these increments once the batch has gone out.
    pub fn token_increment_post_call(
        self: &Arc<Self>,
        request: &RequestRedis,
        increments: Vec<Increment>,
    ) -> Result<(), Error> {
        let pending = scripts::token_increment(&self.post(request)?, &increments);
        let store = Arc::clone(self);
        request.spawn(async move {
            let Err(error) = pending.settled().await else {
                return;
            };
            tracing::warn!(%error, "TTL preservation failed, falling back to regular pipeline");
            if let Err(error) = store.increment_many(&increments).await {
                tracing::warn!(%error, "in-memory increment failed");
            }
        });
        Ok(())
    }

    /// `PARALLEL_RELEASE_SCRIPT` on the request's post-call batch; the caller has already freed
    /// the slot in memory.
    pub fn parallel_release_post_call(
        &self,
        request: &RequestRedis,
        gauges: &[Scope],
        slot_id: &str,
    ) -> Result<(), Error> {
        let pending = scripts::parallel_release(&self.post(request)?, gauges, slot_id);
        request.spawn(async move {
            if let Err(error) = pending.settled().await {
                tracing::warn!(
                    %error,
                    "parallel_release_script failed, the slot stays released in memory only"
                );
            }
        });
        Ok(())
    }

    /// `DualCache.async_increment_cache`: memory, then `INCRBYFLOAT` keeping an existing TTL. A
    /// Redis failure answers with the memory count.
    pub async fn increment(&self, increment: &Increment) -> Result<Number, MemoryError> {
        let local = self.add_local(increment)?;
        let Some(redis) = &self.redis else {
            return Ok(local);
        };
        let key = increment.counter.key();
        let ttl = increment.ttl.unwrap_or(self.redis_ttl);
        let total = redis
            .incr_by_float(&key, increment.amount.into(), Expiry::KeepExisting(ttl))
            .await
            .map_err(Error::from);
        Ok(total.map(Number::Float).unwrap_or_else(|error| {
            tracing::warn!(
                %error,
                "Redis async_increment_cache failed, falling back to in-memory result"
            );
            local
        }))
    }

    /// `DualCache.async_increment_cache_pipeline`: memory, then one pipeline of `INCRBYFLOAT`
    /// with `EXPIRE` per increment.
    pub async fn increment_many(&self, increments: &[Increment]) -> Result<(), MemoryError> {
        increments
            .iter()
            .try_for_each(|increment| self.add_local(increment).map(drop))?;
        let Some(redis) = &self.redis else {
            return Ok(());
        };
        let pipeline = increments
            .iter()
            .fold(redis.pipeline(), |pipeline, increment| {
                pipeline.incr_by_float(
                    &increment.counter.key(),
                    increment.amount.into(),
                    increment.ttl,
                )
            });
        if let Some(error) = redis.run(&pipeline).await.into_iter().find_map(Result::err) {
            tracing::warn!(
                %error,
                "Redis async_increment_cache_pipeline failed, falling back to in-memory result"
            );
        }
        Ok(())
    }

    /// `DualCache.async_increment_cache_pipeline_post_call`: memory now, the Redis increments on
    /// the request's post-call batch.
    pub fn increment_many_post_call(
        &self,
        request: &RequestRedis,
        increments: &[Increment],
    ) -> Result<(), MemoryError> {
        let Some(declared) = self.declare_increments(increments, |client| request.post(client))?
        else {
            return Ok(());
        };
        request.spawn(async move {
            if let Some(error) = settle(declared).await {
                tracing::warn!(%error, "deferred Redis increment failed");
            }
        });
        Ok(())
    }

    fn declare_increments(
        &self,
        increments: &[Increment],
        batch_for: impl FnOnce(&RedisClient) -> RedisBatch,
    ) -> Result<Option<Vec<Declared<f64>>>, MemoryError> {
        increments
            .iter()
            .try_for_each(|increment| self.add_local(increment).map(drop))?;
        let Some(redis) = &self.redis else {
            return Ok(None);
        };
        let batch = batch_for(redis);
        Ok(Some(
            increments
                .iter()
                .map(|increment| scripts::counter_increment(&batch, increment))
                .collect(),
        ))
    }

    /// `InMemoryCache.increment_cache`: `(current or 0) + amount` under one lock.
    fn add_local(&self, increment: &Increment) -> Result<Number, MemoryError> {
        let _guard = self
            .increments
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        let key = increment.counter.key();
        let current = match self.memory.get_cache(&key)? {
            None => Number::Int(0),
            Some(Local::Number(current)) => current,
            Some(_) => return Err(MemoryError::InvalidEntry(None)),
        };
        let total = current + increment.amount;
        self.memory
            .set_cache(key, Local::Number(total), increment.ttl)?;
        Ok(total)
    }

    fn pre(&self, request: Option<&RequestRedis>) -> Result<RedisBatch, Error> {
        let redis = self.redis.as_ref().ok_or(Error::NoRedis)?;
        Ok(request.map_or_else(|| redis.batch(), |request| request.pre(redis)))
    }

    fn post(&self, request: &RequestRedis) -> Result<RedisBatch, Error> {
        let redis = self.redis.as_ref().ok_or(Error::NoRedis)?;
        Ok(request.post(redis))
    }
}

/// Every increment's outcome; the first failure when there is one. `flush` sends the batch first.
async fn settle(declared: Vec<Declared<f64>>) -> Option<Error> {
    let outcomes = join_all(declared.into_iter().map(Declared::settled)).await;
    outcomes.into_iter().find_map(Result::err).map(Error::from)
}

fn decode(raw: &[u8]) -> Result<Local, crate::DecodeError> {
    match python_value(raw)? {
        Value::Str(text) => Ok(Local::Text(text)),
        value => Number::from_value(&value)
            .map(Local::Number)
            .ok_or(crate::DecodeError::Shape(
                "a rate-limit value is a number or a string",
            )),
    }
}

mod keys;
mod types;

use std::{sync::Arc, time::Duration};

use litellm_python_compat::repr::to_str;
use litellm_redis_client::{Declared, Expiry, RedisBatch, commands};

pub use self::keys::{NotASpendCounter, SpendCounter};
pub use self::types::{LostSeed, Reading};
use crate::{
    Error, Number, Pending, RequestRedis, Send,
    dual_cache::{DualCache, MemorySettings},
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
};

/// `RedisCache.async_set_max`, byte for byte, sent as `EVAL` like it: a counter only ever rises to
/// the authoritative spend.
const SET_MAX: &str = concat!(
    "local cur = redis.call('GET', KEYS[1]) ",
    "if cur == false or tonumber(cur) < tonumber(ARGV[1]) then ",
    "redis.call('SET', KEYS[1], ARGV[1]) ",
    "if tonumber(ARGV[2]) > 0 then redis.call('EXPIRE', KEYS[1], ARGV[2]) end ",
    "return ARGV[1] end ",
    "return cur"
);

/// The proxy's spend counters: Redis shared by every pod, and this pod's memory copy that answers
/// only when Redis cannot. Like `spend_counter_cache`'s callers on `main`, most ops reach past
/// `DualCache` to its Redis and its memory.
pub struct SpendCounters {
    cache: DualCache<Number>,
}

impl SpendCounters {
    pub fn new(remote: Option<Remote>) -> Self {
        Self::with_clock(remote, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, now: Clock) -> Self {
        Self {
            cache: DualCache::new(
                Arc::new(RemoteSlot::new(remote)),
                MemorySettings::default(),
                now,
            ),
        }
    }

    /// `DualCache.attach_redis_cache`: memory stays, and `false` means a Redis was already attached.
    pub fn attach(&self, remote: Remote) -> bool {
        self.cache.redis().attach(remote)
    }

    pub fn has_redis(&self) -> bool {
        self.cache.redis().get().is_some()
    }

    /// The TTL every Redis write gives a counter, `None` without Redis.
    pub fn redis_ttl(&self) -> Option<Duration> {
        self.cache.redis().get().map(|remote| remote.default_ttl)
    }

    /// `read_spend_counter_cache_value`: Redis first. `RedisCache.async_get_cache` swallows an unreachable
    /// `GET` as a miss, so an open breaker, a transport error, or a value `float()` refuses reads memory instead.
    pub async fn read(&self, counter: &SpendCounter) -> Result<Reading, Error> {
        let Some(client) = self.cache.redis().client() else {
            return Ok(Reading::Memory(self.get_local(counter)?));
        };
        match client.send(commands::get::<Vec<u8>>(&counter.key())).await {
            Ok(raw) => match types::decode(raw) {
                Ok(value) => Ok(Reading::Redis(value)),
                Err(error) => {
                    tracing::debug!(%error, "spend counter Redis value did not decode, falling back to memory");
                    Ok(Reading::Memory(self.get_local(counter)?))
                }
            },
            Err(litellm_redis_client::Error::BreakerOpen) => {
                Ok(Reading::Memory(self.get_local(counter)?))
            }
            Err(error) => {
                crate::failure_log::redis_failure!(&error, "spend counter Redis read failed");
                Ok(Reading::Memory(self.get_local(counter)?))
            }
        }
    }

    /// One `MGET` for many counters on the batch `send` picks; `None` without Redis. The read
    /// resolves to `None` when it failed, so each caller reads on its own.
    pub fn read_many(
        &self,
        send: Send<'_>,
        counters: &[SpendCounter],
    ) -> Option<Pending<Option<Vec<Option<Number>>>>> {
        let batch = self.cache.redis().batch(send)?;
        let keys: Vec<String> = counters.iter().map(SpendCounter::key).collect();
        let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
        let read = batch.add(commands::mget::<Vec<u8>>(&keys));
        Some(Box::pin(async move {
            let decoded = read
                .resolve()
                .await
                .map_err(Error::from)
                .and_then(|replies| {
                    replies
                        .into_iter()
                        .map(|raw| Ok(types::decode(raw)?))
                        .collect::<Result<Vec<_>, Error>>()
                });
            Ok(decoded
                .inspect_err(|error| {
                    tracing::debug!(%error, "spend counter batch read failed, falling back to per-key reads");
                })
                .ok())
        }))
    }

    /// `increment_spend_counter_cache`: `INCRBYFLOAT` refreshing the TTL, mirrored into memory. A
    /// failed increment may or may not have landed, so the counter is dropped everywhere before the
    /// error returns. Without Redis it is a memory increment.
    pub async fn increment(&self, counter: &SpendCounter, amount: Number) -> Result<Number, Error> {
        let Some(remote) = self.cache.redis().get() else {
            return self.increment_local(counter, amount);
        };
        let key = counter.key();
        let increment = commands::incr_by_float(
            &key,
            amount.into(),
            Some(Expiry::Refresh(remote.default_ttl)),
        );
        match remote.client.send(increment).await {
            Ok(total) => {
                let total = Number::Float(total);
                self.cache.memory().set(&key, total, None)?;
                Ok(total)
            }
            Err(error) => {
                self.invalidate(std::slice::from_ref(counter)).await;
                Err(error.into())
            }
        }
    }

    /// `run_spend_counter_pipeline`: every increment in one round trip, totals in order. The caller
    /// decides what a failure means for counters whose increment may or may not have landed.
    pub async fn increment_many(
        &self,
        increments: &[(SpendCounter, Number)],
    ) -> Result<Vec<Number>, Error> {
        let Some(remote) = self.cache.redis().get() else {
            return increments
                .iter()
                .map(|(counter, amount)| self.increment_local(counter, *amount))
                .collect();
        };
        let pipeline = RedisBatch::pipelined(remote.client.clone());
        let totals = increments.iter().map(|(counter, amount)| {
            pipeline.add(commands::incr_by_float(
                &counter.key(),
                (*amount).into(),
                Some(Expiry::Refresh(remote.default_ttl)),
            ))
        });
        let totals = Declared::all(totals.collect::<Vec<_>>()).await;
        increments
            .iter()
            .zip(totals)
            .map(|((counter, _), total)| {
                let total = Number::Float(total?);
                self.cache.memory().set(&counter.key(), total, None)?;
                Ok(total)
            })
            .collect()
    }

    /// `increment_many` on the request's post-call batch. Each total reaches memory once the batch
    /// goes out, and a counter whose increment failed is dropped everywhere. `false` without Redis.
    pub fn increment_many_post_call(
        self: &Arc<Self>,
        request: &RequestRedis,
        increments: &[(SpendCounter, Number)],
    ) -> bool {
        let Some(remote) = self.cache.redis().get() else {
            return false;
        };
        let batch = Send::PostCall(request).batch(&remote.client);
        let declared = increments
            .iter()
            .map(|(counter, amount)| {
                let total = batch.add(commands::incr_by_float(
                    &counter.key(),
                    (*amount).into(),
                    Some(Expiry::KeepExisting(remote.default_ttl)),
                ));
                (counter.clone(), total)
            })
            .collect::<Vec<_>>();
        let store = Arc::clone(self);
        request.spawn(async move {
            for (counter, total) in declared {
                match total.settled().await {
                    Ok(total) => {
                        if let Err(error) = store.cache.memory().set(&counter.key(), Number::Float(total), None) {
                            tracing::warn!(%error, "mirroring a spend counter total into memory failed");
                        }
                    }
                    Err(error) => {
                        let key = counter.key();
                        crate::failure_log::redis_failure!(
                            &error,
                            &format!("spend counter {key} increment did not land in the post-call pipeline; invalidating it"),
                        );
                        store.invalidate(std::slice::from_ref(&counter)).await;
                    }
                }
            }
        });
        true
    }

    /// `_repair_stale_spend_counter`: raise memory and Redis to `spend`, never lowering either.
    /// `None` without Redis, else what Redis holds afterwards.
    pub async fn set_max(
        &self,
        counter: &SpendCounter,
        spend: Number,
    ) -> Result<Option<Number>, Error> {
        let key = counter.key();
        {
            let _guard = self.cache.memory().guard();
            if self
                .cache
                .memory()
                .get(&key)?
                .is_none_or(|cached| cached.as_f64() < spend.as_f64())
            {
                self.cache.memory().set(&key, spend, None)?;
            }
        }
        let Some(remote) = self.cache.redis().get() else {
            return Ok(None);
        };
        let args = (
            to_str(&spend.to_value()),
            remote.default_ttl.as_secs().to_string(),
        );
        let stored: Option<f64> = remote
            .client
            .send(commands::eval_source(SET_MAX, &[&key], args))
            .await?;
        Ok(stored.map(Number::Float))
    }

    /// `SpendCounterReseed.coalesced(_window)`: seed a cold counter with `SET NX` so one pod wins,
    /// answer with whatever the counter holds, and mirror it into memory. Without Redis, memory
    /// keeps the larger of the seed and what it held.
    pub async fn seed(
        &self,
        counter: &SpendCounter,
        spend: Number,
        lost: LostSeed,
    ) -> Result<Number, Error> {
        let key = counter.key();
        let Some(Remote {
            client,
            default_ttl,
        }) = self.cache.redis().get()
        else {
            return Ok(self.cache.memory().update(&key, None, |cached| {
                Ok(match cached {
                    Some(cached) if cached.as_f64() > spend.as_f64() => {
                        Number::Float(cached.as_f64())
                    }
                    _ => spend,
                })
            })?);
        };
        let current = if client
            .send(commands::set_nx(&key, spend.to_json(), *default_ttl))
            .await?
        {
            spend
        } else {
            match types::decode(client.send(commands::get::<Vec<u8>>(&key)).await?)? {
                Some(winner) => Number::Float(winner.as_f64()),
                None => match lost {
                    LostSeed::Answer => spend,
                    LostSeed::Increment => {
                        let increment = commands::incr_by_float(
                            &key,
                            spend.into(),
                            Some(Expiry::KeepExisting(*default_ttl)),
                        );
                        Number::Float(client.send(increment).await?)
                    }
                },
            }
        };
        self.cache.memory().set(&key, current, None)?;
        Ok(current)
    }

    /// A spend reset: memory, then Redis, each for `ttl` or its own default.
    pub async fn set(
        &self,
        counter: &SpendCounter,
        spend: Number,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        let key = counter.key();
        self.cache.memory().set(&key, spend, ttl)?;
        let Some(remote) = self.cache.redis().get() else {
            return Ok(());
        };
        let ttl = ttl.unwrap_or(remote.default_ttl);
        remote
            .client
            .send(commands::set(&key, spend.to_json(), ttl))
            .await?;
        Ok(())
    }

    /// Drop counters from memory, then from Redis in one round trip.
    pub async fn delete(&self, counters: &[SpendCounter]) -> Result<(), Error> {
        let keys: Vec<String> = counters.iter().map(SpendCounter::key).collect();
        self.cache.delete_keys(&keys).await
    }

    /// `refresh_spend_counter_ttl`: `false` without Redis or for a counter Redis no longer holds.
    pub async fn refresh_ttl(&self, counter: &SpendCounter) -> Result<bool, Error> {
        let Some(remote) = self.cache.redis().get() else {
            return Ok(false);
        };
        Ok(remote
            .client
            .send(commands::expire(&counter.key(), remote.default_ttl))
            .await?)
    }

    pub fn get_local(&self, counter: &SpendCounter) -> Result<Option<Number>, Error> {
        Ok(self.cache.memory().get(&counter.key())?)
    }

    pub fn set_local(
        &self,
        counter: &SpendCounter,
        spend: Number,
        ttl: Option<Duration>,
    ) -> Result<(), Error> {
        Ok(self.cache.memory().set(&counter.key(), spend, ttl)?)
    }

    pub fn delete_local(&self, counter: &SpendCounter) -> Result<(), Error> {
        Ok(self.cache.memory().delete(&counter.key())?)
    }

    /// `DualCache.async_increment_cache(local_only=True)`, which passes `InMemoryCache` no TTL.
    pub fn increment_local(&self, counter: &SpendCounter, amount: Number) -> Result<Number, Error> {
        Ok(self
            .cache
            .memory()
            .increment(&counter.key(), amount, None)?)
    }

    pub fn flush_local(&self) -> Result<(), Error> {
        Ok(self.cache.memory().flush()?)
    }

    pub fn snapshot(&self) -> Result<Vec<(String, Number)>, Error> {
        Ok(self.cache.memory().entries()?)
    }

    /// `invalidate_spend_counter`: memory always, Redis best effort.
    pub async fn invalidate(&self, counters: &[SpendCounter]) {
        if let Err(error) = self.delete(counters).await {
            tracing::debug!(%error, "dropping spend counters from Redis failed");
        }
    }
}

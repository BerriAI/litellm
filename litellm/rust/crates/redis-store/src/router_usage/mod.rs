mod keys;

use std::{sync::Arc, time::Duration};

use litellm_redis_client::{Error as RedisError, commands};
use tokio::sync::oneshot;

pub use self::keys::{Counter, Kind, Scope};
use crate::{
    Error, Number, Pending, RequestRedis, Send, codec,
    dual_cache::{BatchReads, DualCache, PipelineIncrement, Reservation, TierPolicy},
    failure_log::redis_failure,
    memory::{Clock, system_clock},
    remote::{Remote, RemoteSlot},
};

/// `RoutingArgs.ttl`, the window of a per-minute counter.
pub const COUNTER_TTL: Duration = Duration::from_secs(60);

/// A `RedisPipelineIncrementOperation` on one counter.
#[derive(Clone, Debug, PartialEq)]
pub struct Increment {
    pub counter: Counter,
    pub amount: Number,
    pub ttl: Option<Duration>,
}

impl From<&Increment> for PipelineIncrement {
    fn from(increment: &Increment) -> Self {
        Self {
            key: increment.counter.key(),
            amount: increment.amount,
            ttl: increment.ttl,
        }
    }
}

fn pipeline(increments: &[Increment]) -> Vec<PipelineIncrement> {
    increments.iter().map(PipelineIncrement::from).collect()
}

/// The throttle refreshes a declared usage read reserved, which `RouterUsage::release` gives back.
pub struct UsageReservation(Option<Reservation>);

/// A usage read `RouterUsage::declare_get_many` declared, with the refreshes it reserved.
pub struct DeclaredUsage {
    pub read: Pending<Vec<Option<Number>>>,
    pub reservation: UsageReservation,
}

/// One router's usage counters: memory per router, Redis shared by every pod. The router core's
/// `global_router:` counters and the per-deployment ones its strategies and pre-call checks count
/// share one `DualCache`, as they shared the router's on `main`.
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
                scope: Scope::GlobalRouter,
                deployment_id: deployment_id.to_owned(),
                model: model.map(str::to_owned),
                window: minute.to_owned(),
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
        self.usage.batch_get(&keys(counters)).await
    }

    /// `get_many` declared on the batch `send` picks, so a pick's usage read can ride the request's
    /// pre-call pipeline, with the refreshes it reserved. On a request it settles when its batch
    /// answers, whoever awaits it, so a prefetch the pick never takes still backfills memory or gives
    /// its reservation back, as the settled hook of `main`'s prefetch does.
    pub fn declare_get_many(
        self: &Arc<Self>,
        send: Send<'_>,
        counters: &[Counter],
    ) -> Result<DeclaredUsage, Error> {
        let read = self.usage.declare_batch_get(send, &keys(counters))?;
        let reservation = UsageReservation(read.reservation());
        let store = Arc::clone(self);
        let (Some(request), Some(batch)) = (send.request(), read.batch()) else {
            return Ok(DeclaredUsage {
                read: Box::pin(async move { store.usage.resolve_batch_get(read).await }),
                reservation,
            });
        };
        let (settled, answer) = oneshot::channel();
        let settling = Arc::clone(self);
        request.detach(async move {
            drop(settled.send(settling.usage.settle_batch_get(read).await));
        });
        let counters = counters.to_vec();
        Ok(DeclaredUsage {
            read: Box::pin(async move {
                batch.flush().await;
                match answer.await {
                    Ok(read) => read,
                    Err(_) => store.get_local_many(&counters),
                }
            }),
            reservation,
        })
    }

    /// `ArmedRead.release`: a prefetch a pick took but could not use gives its refreshes back, so the
    /// pick's own read goes to Redis.
    pub fn release(&self, reservation: UsageReservation) {
        if let Some(reservation) = reservation.0 {
            self.usage.release(reservation);
        }
    }

    /// `DualCache.async_increment_cache_pipeline`: memory, then one pipeline on its own round trip.
    /// `None` when memory refuses an increment, and the memory totals when Redis fails.
    pub async fn increment_pipeline(&self, increments: &[Increment]) -> Option<Vec<Number>> {
        self.usage.increment_pipeline(&pipeline(increments)).await
    }

    /// `DualCache.async_increment_cache_pipeline_post_call`: memory now, Redis on the request's
    /// post-call batch.
    pub fn increment_post_call(&self, request: &RequestRedis, increments: &[Increment]) {
        self.usage
            .increment_pipeline_post_call(request, &pipeline(increments));
    }

    /// `RedisCache.async_increment_pipeline`, which a strategy's periodic sync sends with the
    /// increments it already counted in memory. `None` without Redis or increments, and a Redis failure is the
    /// caller's to handle, as there.
    pub async fn increment_remote(
        &self,
        increments: &[Increment],
    ) -> Result<Option<Vec<Number>>, Error> {
        self.usage
            .increment_pipeline_remote(&pipeline(increments))
            .await
            .transpose()
            .map_err(Error::from)
    }

    /// `in_memory_cache.async_increment`.
    pub fn increment_local(&self, increment: &Increment) -> Result<Number, Error> {
        Ok(self.usage.memory().increment(
            &increment.counter.key(),
            increment.amount,
            increment.ttl,
        )?)
    }

    /// `get_cache(local_only=True)`.
    pub fn get_local(&self, counter: &Counter) -> Result<Option<Number>, Error> {
        Ok(self.usage.memory().get(&counter.key())?)
    }

    /// `in_memory_cache.async_batch_get_cache`.
    pub fn get_local_many(&self, counters: &[Counter]) -> Result<Vec<Option<Number>>, Error> {
        counters
            .iter()
            .map(|counter| self.get_local(counter))
            .collect()
    }

    /// `in_memory_cache.async_set_cache` without a TTL, which keeps `InMemoryCache`'s own.
    pub fn set_local(&self, counter: &Counter, value: Number) -> Result<(), Error> {
        Ok(self.usage.memory().set(&counter.key(), value, None)?)
    }

    /// The model rate limit check's current usage: memory alone when it already reaches `limit`,
    /// else `redis_cache.get_cache`. No Redis and an open breaker serve memory, and any other
    /// failure reads as `None`, as `RedisCache` swallows it.
    pub async fn current(&self, counter: &Counter, limit: Number) -> Result<Option<Number>, Error> {
        let local = self.get_local(counter)?;
        if local.is_some_and(|local| local.as_f64() >= limit.as_f64()) {
            return Ok(local);
        }
        Ok(match self.shared(counter).await {
            Shared::Unasked => local,
            Shared::Failed => None,
            Shared::Read(count) => count,
        })
    }

    /// The io-token check's usage: what every pod counted in Redis, else this router's memory
    /// count when Redis has none, is absent, refuses or fails.
    pub async fn shared_or_local(&self, counter: &Counter) -> Result<Option<Number>, Error> {
        let local = self.get_local(counter)?;
        Ok(match self.shared(counter).await {
            Shared::Read(Some(count)) => Some(count),
            Shared::Unasked | Shared::Failed | Shared::Read(None) => local,
        })
    }

    pub fn flush_local_counters(&self) -> Result<(), Error> {
        Ok(self.usage.memory().flush()?)
    }

    /// The memory tier keyed as Python's `InMemoryCache` was, for the fuzz probe that reads
    /// in-memory router state.
    pub fn snapshot(&self) -> Result<Vec<(String, Number)>, Error> {
        Ok(self.usage.memory().entries()?)
    }

    /// `redis_cache.get_cache` of one counter, which `RedisCache` answers with `None` on a failure
    /// and its breaker refuses.
    async fn shared(&self, counter: &Counter) -> Shared {
        let Some(client) = self.usage.redis().client() else {
            return Shared::Unasked;
        };
        let key = counter.key();
        match client.send(commands::get::<Vec<u8>>(&key)).await {
            Err(RedisError::BreakerOpen) => Shared::Unasked,
            Err(error) => {
                redis_failure!(
                    &error,
                    "litellm.caching.caching: get() - Got exception from REDIS"
                );
                Shared::Failed
            }
            Ok(raw) => match raw.as_deref().map(codec::decode).transpose() {
                Ok(count) => Shared::Read(count),
                Err(error) => {
                    tracing::warn!(%error, key, "a usage counter did not decode");
                    Shared::Failed
                }
            },
        }
    }
}

/// What one counter's Redis read came back with.
enum Shared {
    /// No Redis, or its breaker refused the read.
    Unasked,
    Failed,
    Read(Option<Number>),
}

fn keys(counters: &[Counter]) -> Vec<String> {
    counters.iter().map(Counter::key).collect()
}

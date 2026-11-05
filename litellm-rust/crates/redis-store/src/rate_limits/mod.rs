mod keys;
mod lua;
mod scripts;
mod types;

use std::{sync::Arc, time::Duration};

use futures_util::future::join_all;
use litellm_redis_client::{Declared, Expiry, RedisBatch, RedisClient, Slot, commands};

pub use self::keys::{Counter, Entry, Kind, Scope};
pub use self::types::{
    AcquireReply, Applied, AtomicCounter, AtomicReply, Gauge, GuardedIncrement, GuardedReply,
    Increment, Local, WindowCount,
};
use crate::{
    Error, Number, Pending, RequestRedis, Send,
    memory::{Clock, Memory, system_clock},
    remote::{Remote, RemoteSlot},
    request::pending,
};

/// The v3 rate limiter's keyspace: its memory tier, its Lua scripts and the plain counter writes
/// `DualCache` made for it on `main`. The admission logic stays with the caller. A script needs
/// Redis, so without one it declares nothing and answers `None`.
pub struct RateLimits {
    memory: Memory<Local>,
    remote: RemoteSlot,
    backfill_ttl: Option<Duration>,
}

impl RateLimits {
    /// `backfill_ttl` is the `default_in_memory_ttl` of the `DualCache` this tier replaces. An
    /// increment without a TTL takes the remote's default.
    pub fn new(remote: Option<Remote>, backfill_ttl: Option<Duration>) -> Self {
        Self::with_clock(remote, backfill_ttl, system_clock())
    }

    pub fn with_clock(remote: Option<Remote>, backfill_ttl: Option<Duration>, now: Clock) -> Self {
        Self {
            memory: Memory::new(None, &now),
            remote: RemoteSlot::new(remote),
            backfill_ttl,
        }
    }

    pub fn has_redis(&self) -> bool {
        self.remote.get().is_some()
    }

    /// `_group_keys_by_hash_tag`: indices of `scopes` sharing a cluster slot, in first-seen order,
    /// or one group when the client is not a cluster and grouping is not forced.
    pub fn slot_groups(&self, scopes: &[Scope], force: bool) -> Vec<Vec<usize>> {
        let by_slot = force || self.remote.client().is_some_and(RedisClient::is_cluster);
        if !by_slot {
            return vec![(0..scopes.len()).collect()];
        }
        scopes
            .iter()
            .enumerate()
            .fold(Vec::<(Slot, Vec<usize>)>::new(), |mut groups, (index, scope)| {
                let slot = Slot::for_key(scope.window_key());
                match groups.iter_mut().find(|(seen, _)| *seen == slot) {
                    Some((_, members)) => members.push(index),
                    None => groups.push((slot, vec![index])),
                }
                groups
            })
            .into_iter()
            .map(|(_, members)| members)
            .collect()
    }

    pub fn get_local(&self, entry: &Entry) -> Result<Option<Local>, Error> {
        Ok(self.memory.get(&entry.key())?)
    }

    pub fn set_local(&self, entry: &Entry, value: Local, ttl: Option<Duration>) -> Result<(), Error> {
        Ok(self.memory.set(&entry.key(), value, ttl)?)
    }

    /// `DualCache.async_get_cache`, backfilling memory for `backfill_ttl` or the store's default.
    pub async fn get(&self, entry: &Entry, backfill_ttl: Option<Duration>) -> Result<Option<Local>, Error> {
        let backfill_ttl = backfill_ttl.or(self.backfill_ttl);
        self.remote
            .read_through(&self.memory, &entry.key(), types::decode, backfill_ttl)
            .await
    }

    pub fn sliding_window(
        &self,
        send: Send<'_>,
        counters: &[Counter],
        now: i64,
        window_size: i64,
    ) -> Option<Pending<Vec<WindowCount>>> {
        Some(pending(scripts::sliding_window(&self.batch(send)?, counters, now, window_size)))
    }

    pub fn read_counters(&self, send: Send<'_>, counters: &[Counter]) -> Option<Pending<Vec<Option<String>>>> {
        Some(pending(scripts::read_counters(&self.batch(send)?, counters)))
    }

    pub fn check_and_increment(&self, send: Send<'_>, counters: &[AtomicCounter]) -> Option<Pending<AtomicReply>> {
        Some(pending(scripts::check_and_increment(&self.batch(send)?, counters)))
    }

    pub fn guarded_increment(&self, send: Send<'_>, increment: &GuardedIncrement) -> Option<Pending<GuardedReply>> {
        Some(pending(scripts::guarded_increment(&self.batch(send)?, increment)))
    }

    pub fn parallel_acquire(
        &self,
        send: Send<'_>,
        gauges: &[Gauge],
        slot_ttl: Duration,
        slot_id: &str,
    ) -> Option<Pending<AcquireReply>> {
        Some(pending(scripts::parallel_acquire(&self.batch(send)?, gauges, slot_ttl, slot_id)))
    }

    pub fn parallel_release(&self, send: Send<'_>, gauges: &[Scope], slot_id: &str) -> Option<Pending<Vec<i64>>> {
        Some(pending(self.declare_release(send, gauges, slot_id)?))
    }

    pub fn parallel_count(&self, send: Send<'_>, gauges: &[Scope], slot_ttl: Duration) -> Option<Pending<Vec<i64>>> {
        Some(pending(scripts::parallel_count(&self.batch(send)?, gauges, slot_ttl)))
    }

    pub fn token_increment(&self, send: Send<'_>, increments: &[Increment]) -> Option<Pending<Vec<f64>>> {
        Some(pending(self.declare_token_increment(send, increments)?))
    }

    /// `TOKEN_INCREMENT_SCRIPT` on the request's post-call batch. A failed script falls back to
    /// `increment_many` for these increments once the batch has gone out. `false` without Redis.
    pub fn token_increment_post_call(self: &Arc<Self>, request: &RequestRedis, increments: Vec<Increment>) -> bool {
        let Some(declared) = self.declare_token_increment(Send::PostCall(request), &increments) else {
            return false;
        };
        let store = Arc::clone(self);
        request.spawn(async move {
            let Err(error) = declared.settled().await else {
                return;
            };
            tracing::warn!(%error, "TTL preservation failed, falling back to regular pipeline");
            if let Err(error) = store.increment_many(&increments).await {
                tracing::warn!(%error, "in-memory increment failed");
            }
        });
        true
    }

    /// `PARALLEL_RELEASE_SCRIPT` on the request's post-call batch; the caller has already freed
    /// the slot in memory. `false` without Redis.
    pub fn parallel_release_post_call(&self, request: &RequestRedis, gauges: &[Scope], slot_id: &str) -> bool {
        let Some(declared) = self.declare_release(Send::PostCall(request), gauges, slot_id) else {
            return false;
        };
        request.spawn(async move {
            if let Err(error) = declared.settled().await {
                tracing::warn!(%error, "parallel_release_script failed, the slot stays released in memory only");
            }
        });
        true
    }

    /// `DualCache.async_increment_cache`: memory, then `INCRBYFLOAT` keeping an existing TTL. A
    /// Redis failure answers with the memory count.
    pub async fn increment(&self, increment: &Increment) -> Result<Number, Error> {
        let local = self.add_local(increment)?;
        let Some(remote) = self.remote.get() else {
            return Ok(local);
        };
        let ttl = increment.ttl.unwrap_or(remote.default_ttl);
        let total = remote
            .client
            .send(commands::incr_by_float(
                &increment.counter.key(),
                increment.amount.into(),
                Some(Expiry::KeepExisting(ttl)),
            ))
            .await;
        Ok(total.map(Number::Float).unwrap_or_else(|error| {
            tracing::warn!(%error, "Redis async_increment_cache failed, falling back to in-memory result");
            local
        }))
    }

    /// `DualCache.async_increment_cache_pipeline`: memory, then one pipeline of `INCRBYFLOAT`
    /// with `EXPIRE` per increment.
    pub async fn increment_many(&self, increments: &[Increment]) -> Result<(), Error> {
        increments
            .iter()
            .try_for_each(|increment| self.add_local(increment).map(drop))?;
        let Some(client) = self.remote.client() else {
            return Ok(());
        };
        let pipeline = RedisBatch::pipelined(client.clone());
        let totals = increments.iter().map(|increment| {
            pipeline.add(commands::incr_by_float(
                &increment.counter.key(),
                increment.amount.into(),
                increment.ttl.map(Expiry::Refresh),
            ))
        });
        let totals = Declared::all(totals.collect::<Vec<_>>()).await;
        if let Some(error) = totals.into_iter().find_map(Result::err) {
            tracing::warn!(%error, "Redis async_increment_cache_pipeline failed, falling back to in-memory result");
        }
        Ok(())
    }

    /// `DualCache.async_increment_cache_pipeline_post_call`: memory now, the Redis increments on
    /// the request's post-call batch.
    pub fn increment_many_post_call(&self, request: &RequestRedis, increments: &[Increment]) -> Result<(), Error> {
        increments
            .iter()
            .try_for_each(|increment| self.add_local(increment).map(drop))?;
        let Some(batch) = self.remote.batch(Send::PostCall(request)) else {
            return Ok(());
        };
        let declared = increments
            .iter()
            .map(|increment| scripts::counter_increment(&batch, increment))
            .collect::<Vec<_>>();
        request.spawn(async move {
            let outcomes = join_all(declared.into_iter().map(Declared::settled)).await;
            if let Some(error) = outcomes.into_iter().find_map(Result::err) {
                tracing::warn!(%error, "deferred Redis increment failed");
            }
        });
        Ok(())
    }

    fn declare_release(&self, send: Send<'_>, gauges: &[Scope], slot_id: &str) -> Option<Declared<Vec<i64>>> {
        Some(scripts::parallel_release(&self.batch(send)?, gauges, slot_id))
    }

    fn declare_token_increment(&self, send: Send<'_>, increments: &[Increment]) -> Option<Declared<Vec<f64>>> {
        Some(scripts::token_increment(&self.batch(send)?, increments))
    }

    /// `InMemoryCache.increment_cache`: `(current or 0) + amount` under one lock.
    fn add_local(&self, increment: &Increment) -> Result<Number, Error> {
        let key = increment.counter.key();
        let _guard = self.memory.guard();
        let total = match self.memory.get(&key)? {
            None => increment.amount,
            Some(Local::Number(current)) => current + increment.amount,
            Some(_) => return Err(litellm_cache::Error::InvalidEntry(None).into()),
        };
        self.memory.set(&key, Local::Number(total), increment.ttl)?;
        Ok(total)
    }

    fn batch(&self, send: Send<'_>) -> Option<RedisBatch> {
        self.remote.batch(send)
    }
}
